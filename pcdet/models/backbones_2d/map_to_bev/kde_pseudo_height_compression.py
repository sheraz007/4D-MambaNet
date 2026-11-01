import torch
import torch.nn as nn


class KDEPseudoHeightCompression(nn.Module):
    """
    KDE branch on top of pillar BEV features:
      1) Build a KDE-like density/stat map from radar points in BEV.
      2) Encode KDE stats to feature channels.
      3) Concatenate with pillar BEV features and fuse by 1x1 conv.
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

        self.kde_out_channels = int(self.model_cfg.get('KDE_OUT_CHANNELS', 16))
        self.sigma_z = float(self.model_cfg.get('KDE_SIGMA_Z', 1.0))
        self.sigma_vr = float(self.model_cfg.get('KDE_SIGMA_VR', 1.0))
        self.sigma_rcs = float(self.model_cfg.get('KDE_SIGMA_RCS', 5.0))

        self.kde_encoder = nn.Sequential(
            nn.Conv2d(3, self.kde_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.kde_out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.num_bev_features + self.kde_out_channels, self.num_bev_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_bev_features, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def _build_kde_stats_map(self, points, batch_size, h_bev, w_bev, stride, device, dtype):
        """
        points format in this project: [bs, x, y, z, rcs, vr, v_r_comp, time, ...]
        We use x,y,z,vr,rcs for KDE-like weighting in spatial + Doppler space.
        """
        kde_stats = torch.zeros((batch_size, 3, h_bev, w_bev), device=device, dtype=dtype)

        if points is None or points.shape[0] == 0:
            return kde_stats

        points = points.to(device=device)
        bs_idx = points[:, 0].long()
        x = points[:, 1]
        y = points[:, 2]
        z = points[:, 3]
        rcs = points[:, 4]
        vr = points[:, 5]

        pcx0, pcy0, pcz0, pcx1, pcy1, pcz1 = self.point_cloud_range
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
            return kde_stats

        bs_idx = bs_idx[valid]
        ix = ix[valid]
        iy = iy[valid]
        z = z[valid]
        vr = vr[valid]
        rcs = rcs[valid]

        z_center = 0.5 * (pcz0 + pcz1)
        z_term = ((z - z_center) / self.sigma_z) ** 2
        vr_term = (vr / self.sigma_vr) ** 2
        rcs_term = (rcs / self.sigma_rcs) ** 2
        weight = torch.exp(-0.5 * (z_term + vr_term + rcs_term)).to(dtype=dtype)

        linear = bs_idx * (h_bev * w_bev) + iy * w_bev + ix
        flat_size = batch_size * h_bev * w_bev

        density = torch.zeros(flat_size, device=device, dtype=dtype)
        doppler_energy = torch.zeros(flat_size, device=device, dtype=dtype)
        rcs_energy = torch.zeros(flat_size, device=device, dtype=dtype)

        density.index_add_(0, linear, weight)
        doppler_energy.index_add_(0, linear, weight * (vr.abs().to(dtype=dtype)))
        rcs_energy.index_add_(0, linear, weight * (rcs.abs().to(dtype=dtype)))

        kde_stats[:, 0] = density.view(batch_size, h_bev, w_bev)
        kde_stats[:, 1] = doppler_energy.view(batch_size, h_bev, w_bev)
        kde_stats[:, 2] = rcs_energy.view(batch_size, h_bev, w_bev)
        return kde_stats

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()  # [B, C, H, W]
        if self.detach:
            spatial_features = spatial_features.detach()

        batch_size, _, h_bev, w_bev = spatial_features.shape
        stride = int(batch_dict.get('encoded_spconv_tensor_stride', 8))

        points = batch_dict.get('points', None)
        kde_stats = self._build_kde_stats_map(
            points=points,
            batch_size=batch_size,
            h_bev=h_bev,
            w_bev=w_bev,
            stride=stride,
            device=spatial_features.device,
            dtype=spatial_features.dtype,
        )
        kde_features = self.kde_encoder(kde_stats)

        fused_spatial_features = self.fuse_conv(torch.cat([spatial_features, kde_features], dim=1))

        batch_dict['kde_features'] = kde_features
        batch_dict['spatial_features'] = fused_spatial_features
        batch_dict['spatial_features_2d'] = fused_spatial_features
        batch_dict['spatial_features_stride'] = stride

        if self.model_cfg.get('OUTPUT_BEV_NON_EMPTY_MASK', False):
            bev_non_empty_indices = encoded_spconv_tensor.indices.long()
            bev_non_empty_mask = torch.zeros(
                [fused_spatial_features.shape[0], *fused_spatial_features.shape[-2:]],
                dtype=torch.bool,
                device=fused_spatial_features.device
            )
            bev_non_empty_mask[bev_non_empty_indices[:, 0], bev_non_empty_indices[:, 1], bev_non_empty_indices[:, 2]] = True
            batch_dict['bev_non_empty_indices'] = bev_non_empty_indices
            batch_dict['bev_non_empty_mask'] = bev_non_empty_mask

        return batch_dict
