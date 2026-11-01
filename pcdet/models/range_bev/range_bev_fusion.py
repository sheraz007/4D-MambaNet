import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .range_cnn_encoder import RangeCNNEncoder


def _sinusoidal_xyz_encoding(xyz_n, num_bands):
    """
    xyz_n: [B, K, 3] values in roughly [-1, 1] per axis
    Returns: [B, K, 3 * 2 * num_bands]
    """
    B, K, _ = xyz_n.shape
    device, dtype = xyz_n.device, xyz_n.dtype
    out = []
    scales = 2.0 ** torch.arange(0, num_bands, device=device, dtype=dtype)
    for ax in range(3):
        v = xyz_n[..., ax:ax + 1]  # [B,K,1]
        for s in scales:
            x = v * s * math.pi
            out.append(torch.sin(x))
            out.append(torch.cos(x))
    return torch.cat(out, dim=-1)


class RangeBEVFusion(nn.Module):
    def __init__(self, model_cfg, bev_feature_dim):
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', False)
        if not self.enabled:
            return

        self.fusion_method = model_cfg.get('FUSION_METHOD', 'concat')
        self.range_image_size = model_cfg.get('RANGE_IMAGE_SIZE', [64, 512])
        self.fov = model_cfg.get('FOV', [10.0, -30.0])
        self.max_range = model_cfg.get('MAX_RANGE', 50.0)
        range_feature_dim = model_cfg.get('RANGE_FEATURE_DIM', 128)

        use_cnn = model_cfg.get('USE_CNN_ENCODER', False)
        if use_cnn:
            cnn_cfg = model_cfg.get('CNN_CONFIG', {})
            in_ch = cnn_cfg.get('IN_CHANNELS', 7)
            mid_ch = cnn_cfg.get('MID_CHANNELS', 32)
            out_ch = cnn_cfg.get('OUT_CHANNELS', 64)
            self.range_cnn = RangeCNNEncoder(in_ch, mid_ch, out_ch)
            proj_in = out_ch
        else:
            self.range_cnn = None
            proj_in = model_cfg.get('IN_CHANNELS', 7)

        self.range_pixel_proj = nn.Conv2d(proj_in, range_feature_dim, 1)

        if self.fusion_method == 'gated_residual':
            self.gate = nn.Sequential(
                nn.Conv2d(range_feature_dim + bev_feature_dim, 1, 1),
                nn.Sigmoid()
            )
            nn.init.constant_(self.gate[-2].bias, -3.0)
            self.delta_proj = nn.Conv2d(
                range_feature_dim, bev_feature_dim, 1)
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)
        elif self.fusion_method == 'transformer':
            transformer_cfg = self.model_cfg.get('TRANSFORMER_CONFIG', {})
            self.max_range_tokens = int(transformer_cfg.get('MAX_RANGE_TOKENS', 2048))
            self.bev_query_res = int(transformer_cfg.get('BEV_QUERY_RES', transformer_cfg.get('BEV_RES', 32)))
            self.d_model = int(transformer_cfg.get('D_MODEL', bev_feature_dim))
            self.nhead = int(transformer_cfg.get('NHEAD', 4))
            self.dropout = float(transformer_cfg.get('DROPOUT', 0.1))

            # Range tokens are 8 point channels + 1 density channel => 9
            self.token_proj = nn.Linear(9, self.d_model)

            # BEV queries are projected from bev_feature_dim (usually 128) to d_model
            self.bev_token_proj = nn.Linear(bev_feature_dim, self.d_model)
            self.out_proj = nn.Linear(self.d_model, bev_feature_dim)

            self.transformer_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.nhead,
                dropout=self.dropout,
                batch_first=True,
            )

            # Learned 2D positional embeddings (separate row/col embeddings)
            H_r, W_r = self.range_image_size
            self.range_row_embed = nn.Embedding(H_r, self.d_model)
            self.range_col_embed = nn.Embedding(W_r, self.d_model)

            R = self.bev_query_res
            self.bev_row_embed = nn.Embedding(R, self.d_model)
            self.bev_col_embed = nn.Embedding(R, self.d_model)

            # Learnable residual scaling
            self.transformer_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

            # Optional metric 3D positional encoding on range tokens (Scenario B: "geometric blindness")
            # XYZ_PE_MODE: 'none' | 'concat' (Linear on normalized xyz) | 'sinusoidal' (Fourier features + Linear)
            self.xyz_pe_mode = str(transformer_cfg.get('XYZ_PE_MODE', 'none')).lower()
            self.xyz_pe_bands = int(transformer_cfg.get('XYZ_PE_BANDS', 8))
            pc = transformer_cfg.get('POINT_CLOUD_RANGE_FOR_XYZ_PE', [0.0, -25.6, -3.0, 51.2, 25.6, 2.0])
            self.register_buffer(
                '_xyz_pc_min', torch.tensor([pc[0], pc[1], pc[2]], dtype=torch.float32), persistent=False,
            )
            self.register_buffer(
                '_xyz_pc_max', torch.tensor([pc[3], pc[4], pc[5]], dtype=torch.float32), persistent=False,
            )
            if self.xyz_pe_mode == 'concat':
                self.xyz_proj = nn.Linear(3, self.d_model)
            elif self.xyz_pe_mode == 'sinusoidal':
                sin_dim = 3 * 2 * self.xyz_pe_bands
                self.xyz_sin_proj = nn.Linear(sin_dim, self.d_model)
            else:
                self.xyz_pe_mode = 'none'

    def _normalize_xyz_for_pe(self, xyz, device, dtype):
        """Map raw xyz (meters) to [-1, 1] using fixed point cloud range."""
        lo = self._xyz_pc_min.to(device=device, dtype=dtype)
        hi = self._xyz_pc_max.to(device=device, dtype=dtype)
        return 2.0 * (xyz - lo) / (hi - lo).clamp(min=1e-6) - 1.0

    def _get_aggregated_range_image(self, batch_dict):
        """
        Transformer-only dense range image with mean aggregation + density channel.

        Returns:
            range_img9: [B, 9, H_r, W_r]
                channels: [x,y,z,rcs,v_r,v_r_comp,time,tem_idx, density]
        """
        points = batch_dict['points']
        batch_size = batch_dict['batch_size']
        H, W = self.range_image_size

        range_images = []
        for b in range(batch_size):
            mask = points[:, 0] == b
            pts = points[mask, 1:]  # [N, F], F should be 8 in your setup
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            fov_up = self.fov[0] * (3.14159 / 180.0)
            fov_down = self.fov[1] * (3.14159 / 180.0)
            fov_total = fov_up - fov_down

            r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-6)
            pitch = torch.asin((z / r).clamp(-1, 1))
            yaw = torch.atan2(y, x)
            row = 1.0 - (pitch - fov_down) / fov_total
            col = 0.5 * (1.0 - yaw / 3.14159)
            row = (row * H).long().clamp(0, H - 1)
            col = (col * W).long().clamp(0, W - 1)

            linear = row * W + col  # [N]
            HW = H * W
            F = pts.shape[1]

            # Sum aggregation and density (counts) per pixel
            sums = torch.zeros((HW, F), dtype=pts.dtype, device=pts.device)
            counts = torch.zeros((HW,), dtype=pts.dtype, device=pts.device)
            if linear.numel() > 0:
                ones = torch.ones((linear.numel(),), dtype=pts.dtype, device=pts.device)
                counts.index_add_(0, linear, ones)
                sums.index_add_(0, linear, pts)

            counts_clamped = counts.clamp(min=1.0)
            mean = sums / counts_clamped.unsqueeze(1)  # [HW, F]

            mean_hwf = mean.view(H, W, F).permute(2, 0, 1).contiguous()  # [F, H, W]
            density_hw = counts.view(H, W).unsqueeze(0)  # [1, H, W]
            range_img9 = torch.cat([mean_hwf, density_hw], dim=0)  # [9, H, W]
            range_images.append(range_img9)

        return torch.stack(range_images, dim=0)  # [B, 9, H, W]

    def _extract_sparse_tokens(self, range_img9):
        """
        Transformer-only deterministic sparse token extraction from [B, 9, H, W].

        Returns:
            token_feat: [B, K, 9] padded with zeros
            row_idx:    [B, K] long
            col_idx:    [B, K] long
            key_padding_mask: [B, K] bool (True = ignore)
        """
        B, C, H, W = range_img9.shape
        assert C == 9
        K = self.max_range_tokens

        density = range_img9[:, 8, :, :]  # [B, H, W]
        density_flat = density.reshape(B, -1)  # [B, HW]

        token_feat = torch.zeros((B, K, 9), dtype=range_img9.dtype, device=range_img9.device)
        row_idx = torch.zeros((B, K), dtype=torch.long, device=range_img9.device)
        col_idx = torch.zeros((B, K), dtype=torch.long, device=range_img9.device)
        key_padding_mask = torch.ones((B, K), dtype=torch.bool, device=range_img9.device)

        HW = H * W
        for b in range(B):
            active = density_flat[b] > 0
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)  # [M]
            M = active_idx.numel()
            if M == 0:
                continue

            dens = density_flat[b, active_idx]  # [M]
            if M > K:
                # Top-K by density desc, then linear index asc as tie-break
                # Build a composite key that sorts deterministically.
                # key = dens*(HW+1) + (HW - idx) => higher dens first; for equal dens, smaller idx => larger (HW-idx).
                idx_float = active_idx.to(dens.dtype)
                key = dens * (HW + 1.0) + (float(HW) - idx_float)
                order = torch.argsort(key, descending=True)
                sel = active_idx[order[:K]]
            else:
                sel = active_idx

            k_sel = sel.numel()
            if k_sel == 0:
                continue

            rows = (sel // W).to(torch.long)
            cols = (sel % W).to(torch.long)

            feats = range_img9[b, :, rows, cols].transpose(0, 1).contiguous()  # [k_sel, 9]
            token_feat[b, :k_sel] = feats
            row_idx[b, :k_sel] = rows
            col_idx[b, :k_sel] = cols
            key_padding_mask[b, :k_sel] = False

        return token_feat, row_idx, col_idx, key_padding_mask

    def _forward_transformer(self, batch_dict, bev_features):
        """
        Transformer-only fusion. Reads/writes batch_dict['spatial_features'] only.
        """
        # Dense aggregated range with density channel
        range_img9 = self._get_aggregated_range_image(batch_dict)  # [B, 9, H_r, W_r]

        # Sparse token extraction
        token_feat, range_row_idx, range_col_idx, key_padding_mask = self._extract_sparse_tokens(range_img9)
        # token_feat: [B, K, 9]

        B, C_bev, H_bev, W_bev = bev_features.shape
        R = self.bev_query_res

        # Downsample BEV to fixed query resolution
        bev_small = F.adaptive_avg_pool2d(bev_features, (R, R))  # [B, C, R, R]
        bev_tokens = bev_small.permute(0, 2, 3, 1).reshape(B, R * R, C_bev)  # [B, Q, C]
        bev_tokens = self.bev_token_proj(bev_tokens)  # [B, Q, d_model]

        # BEV positional embeddings for fixed R x R grid
        device = bev_features.device
        grid_rows = torch.arange(R, device=device).view(R, 1).expand(R, R).reshape(-1)  # [Q]
        grid_cols = torch.arange(R, device=device).view(1, R).expand(R, R).reshape(-1)  # [Q]
        bev_pos = self.bev_row_embed(grid_rows) + self.bev_col_embed(grid_cols)  # [Q, d_model]
        bev_tokens = bev_tokens + bev_pos.unsqueeze(0)  # [B, Q, d_model]

        # Range token projection + positional embeddings
        range_tokens = self.token_proj(token_feat)  # [B, K, d_model]
        range_pos = self.range_row_embed(range_row_idx) + self.range_col_embed(range_col_idx)  # [B, K, d_model]
        range_tokens = range_tokens + range_pos

        # Optional absolute 3D grounding (first 3 channels of token_feat = mean x,y,z in meters)
        if getattr(self, 'xyz_pe_mode', 'none') == 'concat':
            xyz = token_feat[..., :3]
            xyz_n = self._normalize_xyz_for_pe(xyz, token_feat.device, token_feat.dtype)
            range_tokens = range_tokens + self.xyz_proj(xyz_n)
        elif getattr(self, 'xyz_pe_mode', 'none') == 'sinusoidal':
            xyz = token_feat[..., :3]
            xyz_n = self._normalize_xyz_for_pe(xyz, token_feat.device, token_feat.dtype)
            sin_e = _sinusoidal_xyz_encoding(xyz_n, self.xyz_pe_bands)
            range_tokens = range_tokens + self.xyz_sin_proj(sin_e)

        # Cross-attention: BEV queries attend to Range keys/values
        attn_out, _ = self.transformer_attn(
            query=bev_tokens,
            key=range_tokens,
            value=range_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # [B, Q, d_model]

        # Project back to BEV channel dim and reshape to [B, C, R, R]
        out_tokens = self.out_proj(attn_out)  # [B, Q, C_bev]
        out_small = out_tokens.reshape(B, R, R, C_bev).permute(0, 3, 1, 2).contiguous()  # [B, C, R, R]

        # Upsample back to original BEV resolution
        out_up = F.interpolate(out_small, size=(H_bev, W_bev), mode='bilinear', align_corners=False)  # [B, C, H, W]
        alpha = self.transformer_alpha.to(out_up.dtype)
        out = bev_features + alpha * out_up
        return out

    def project_to_range_image(self, batch_dict):
        points = batch_dict['points']
        batch_size = batch_dict['batch_size']
        H, W = self.range_image_size

        pc_range = [0.0, -25.6, -3.0, 51.2, 25.6, 2.0]
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        x_range = x_max - x_min
        y_range = y_max - y_min

        range_images = []
        for b in range(batch_size):
            mask = points[:, 0] == b
            pts = points[mask, 1:]

            x, y = pts[:, 0], pts[:, 1]

            col = ((x - x_min) / x_range * W).long().clamp(0, W - 1)
            row = ((y - y_min) / y_range * H).long().clamp(0, H - 1)

            img = torch.zeros(
                pts.shape[1], H, W,
                dtype=pts.dtype, device=pts.device)
            img[:, row, col] = pts.T

            img[0].clamp_(-55.0, 55.0).div_(55.0)
            img[1].clamp_(-30.0, 30.0).div_(30.0)
            img[2].clamp_(-4.0,  4.0).div_(4.0)
            img[3].clamp_(-30.0, 60.0).add_(30.0).div_(90.0)
            img[4].clamp_(-25.0, 25.0).div_(25.0)
            img[5].clamp_(-25.0, 25.0).div_(25.0)
            img[6].clamp_(0.0,   1.0)

            range_images.append(img)

        return torch.stack(range_images, dim=0)

    def forward(self, batch_dict):
        if not self.enabled:
            return batch_dict
        
        bev_features = batch_dict['spatial_features']
        
        if not hasattr(self, '_debug_done'):
            self._debug_done = True
            print(f"[RangeBEVFusion] bev_features shape: {bev_features.shape}")
            print(f"[RangeBEVFusion] fusion_method: {self.fusion_method}")

        if self.fusion_method == 'transformer':
            out = self._forward_transformer(batch_dict, bev_features)
            batch_dict['spatial_features'] = out
            return batch_dict

        range_image = self.project_to_range_image(batch_dict)
        if self.range_cnn is not None:
            range_image = self.range_cnn(range_image)

        range_feats = self.range_pixel_proj(range_image)

        H_bev = bev_features.shape[2]
        W_bev = bev_features.shape[3]
        range_bev = F.adaptive_avg_pool2d(range_feats, (H_bev, W_bev))
        
        if self.fusion_method == 'gated_residual':
            combined = torch.cat([bev_features, range_bev], dim=1)
            alpha = self.gate(combined)
            delta = self.delta_proj(range_bev)
            
            if not hasattr(self, '_gate_debug'):
                self._gate_debug = True
                print(f"[RangeBEVFusion] alpha mean: {alpha.mean().item():.4f}")
                print(f"[RangeBEVFusion] alpha std:  {alpha.std().item():.4f}")
                if alpha.mean().item() < 0.05 or alpha.mean().item() > 0.95:
                    print("[RangeBEVFusion] WARNING: gate may be collapsed")
                else:
                    print("[RangeBEVFusion] gate looks healthy")
            
            out = bev_features + alpha * delta
        
        elif self.fusion_method == 'concat':
            out = torch.cat([bev_features, range_bev], dim=1)
        
        else:
            out = bev_features
        
        batch_dict['spatial_features'] = out
        return batch_dict

