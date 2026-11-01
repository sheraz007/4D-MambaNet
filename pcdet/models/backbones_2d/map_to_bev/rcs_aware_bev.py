import torch
import torch.nn as nn


class RCSAwareBEVCompression(nn.Module):
    """Optional RCS-aware dense BEV fusion (paper Table 7 ablation only).

    Builds a 2-channel BEV map from point-wise RCS (mean / max per cell) and
    fuses it with dense BEV features through 1x1 convolutions. Not used by the
    main 4D-MambaNet model; enable only via ``ablation_rcs_bev/*`` configs that
    study the RCS injection stage.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.detach = self.model_cfg.get('DETACH', False)

        self.point_cloud_range = self.model_cfg.get(
            'POINT_CLOUD_RANGE', [0, -25.6, -3, 51.2, 25.6, 2]
        )
        self.base_voxel_size = self.model_cfg.get('BASE_VOXEL_SIZE', [0.05, 0.05])
        self.rcs_out_channels = int(self.model_cfg.get('RCS_OUT_CHANNELS', 16))
        # After collate, points are [batch_idx, <encoded channels...>]. VoD uses RCS at index 4;
        # TJ4D (absolute_coordinates_encoding) is [bs, x, y, z, vr, range, power, alpha, beta, vrc] -> vrc at 9.
        self.rcs_feature_index = int(self.model_cfg.get('RCS_FEATURE_INDEX', 4))
        # real: submitted map. shuffled / zeros keep the same encoder parameters.
        self.rcs_map_mode = str(self.model_cfg.get('RCS_MAP', 'real')).lower()
        self.rcs_map_seed = int(self.model_cfg.get('RCS_MAP_SEED', self.model_cfg.get('SEED', 0)))
        if self.rcs_map_mode not in ('real', 'shuffled', 'zeros'):
            raise ValueError(f'RCS_MAP must be real, shuffled, or zeros, got {self.rcs_map_mode}')

        self.rcs_encoder = nn.Sequential(
            nn.Conv2d(2, self.rcs_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.rcs_out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(
                self.num_bev_features + self.rcs_out_channels,
                self.num_bev_features,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.num_bev_features, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def _build_rcs_map(self, points, batch_size, h_bev, w_bev, stride, device, dtype):
        """
        Build RCS-based spatial map from raw radar points.
        points format: [bs, x, y, z, ...] with RCS-like channel at model_cfg RCS_FEATURE_INDEX (default 4).
        Output: (B, 2, H, W)
          channel 0: mean RCS per BEV cell
          channel 1: max RCS per BEV cell
        """
        rcs_map = torch.zeros((batch_size, 2, h_bev, w_bev),
                              device=device, dtype=dtype)

        if points is None or points.shape[0] == 0:
            return rcs_map

        points = points.to(device=device)
        bs_idx = points[:, 0].long()
        x   = points[:, 1]
        y   = points[:, 2]
        rcs = points[:, self.rcs_feature_index]

        pcx0, pcy0 = self.point_cloud_range[0], self.point_cloud_range[1]
        vx = float(self.base_voxel_size[0]) * float(stride)
        vy = float(self.base_voxel_size[1]) * float(stride)

        ix = torch.floor((x - pcx0) / vx).long()
        iy = torch.floor((y - pcy0) / vy).long()

        valid = (
            (bs_idx >= 0) & (bs_idx < batch_size) &
            (ix >= 0) & (ix < w_bev) &
            (iy >= 0) & (iy < h_bev)
        )
        if valid.sum() == 0:
            return rcs_map

        bs_idx = bs_idx[valid]
        ix     = ix[valid]
        iy     = iy[valid]
        rcs    = rcs[valid].to(dtype=dtype)

        linear    = bs_idx * (h_bev * w_bev) + iy * w_bev + ix
        flat_size = batch_size * h_bev * w_bev

        rcs_sum   = torch.zeros(flat_size, device=device, dtype=dtype)
        rcs_count = torch.zeros(flat_size, device=device, dtype=dtype)
        rcs_max   = torch.zeros(flat_size, device=device, dtype=dtype)

        rcs_sum.index_add_(0, linear, rcs)
        rcs_count.index_add_(0, linear, torch.ones_like(rcs))
        rcs_max.index_add_(0, linear, rcs)

        rcs_mean = rcs_sum / (rcs_count + 1e-6)
        rcs_map[:, 0] = rcs_mean.view(batch_size, h_bev, w_bev)
        rcs_map[:, 1] = rcs_max.view(batch_size, h_bev, w_bev)

        if self.rcs_map_mode == 'real':
            return rcs_map
        if self.rcs_map_mode == 'zeros':
            return torch.zeros_like(rcs_map)

        occupied = (rcs_count.view(batch_size, h_bev, w_bev) > 0)
        return self._shuffle_occupied_cells(rcs_map, occupied)

    def _shuffle_occupied_cells(self, rcs_map, occupied):
        """Permute non-empty cells inside each sample. Empty cells stay empty.

        Both channels move together, so each cell keeps its (mean, max) pair.
        Generator seed is cfg seed + batch index.
        """
        out = rcs_map.clone()
        for b in range(rcs_map.shape[0]):
            cells = occupied[b].nonzero(as_tuple=False)
            n = cells.shape[0]
            if n <= 1:
                continue
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.rcs_map_seed + int(b))
            perm = torch.randperm(n, generator=generator)
            src = rcs_map[b, :, cells[:, 0], cells[:, 1]].clone()
            dst = cells[perm]
            out[b, :, dst[:, 0], dst[:, 1]] = src
        return out

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        if self.detach:
            spatial_features = spatial_features.detach()

        batch_size, _, h_bev, w_bev = spatial_features.shape
        stride = int(batch_dict.get('encoded_spconv_tensor_stride', 8))

        points = batch_dict.get('points', None)
        rcs_map = self._build_rcs_map(
            points=points,
            batch_size=batch_size,
            h_bev=h_bev,
            w_bev=w_bev,
            stride=stride,
            device=spatial_features.device,
            dtype=spatial_features.dtype,
        )

        rcs_features = self.rcs_encoder(rcs_map)
        fused = self.fuse_conv(
            torch.cat([spatial_features, rcs_features], dim=1)
        )

        batch_dict['spatial_features']        = fused
        batch_dict['spatial_features_2d']     = fused
        batch_dict['spatial_features_stride'] = stride

        if self.model_cfg.get('OUTPUT_BEV_NON_EMPTY_MASK', False):
            bev_non_empty_indices = encoded_spconv_tensor.indices.long()
            bev_non_empty_mask = torch.zeros(
                [fused.shape[0], *fused.shape[-2:]],
                dtype=torch.bool, device=fused.device
            )
            bev_non_empty_mask[
                bev_non_empty_indices[:, 0],
                bev_non_empty_indices[:, 1],
                bev_non_empty_indices[:, 2]
            ] = True
            batch_dict['bev_non_empty_indices'] = bev_non_empty_indices
            batch_dict['bev_non_empty_mask']    = bev_non_empty_mask

        return batch_dict
