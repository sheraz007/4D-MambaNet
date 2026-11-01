# Based on Radar M3-Net (Y. Yang et al., Expert Systems with Applications, 2025). Released with permission.
import torch
from ....utils.spconv_utils import replace_feature
from .rcs_aware_bev import RCSAwareBEVCompression


class RCSAwareBEVCompressionSparse(RCSAwareBEVCompression):
    """Sparse-tensor variant of RCS-aware BEV fusion (Table 7 ablation only).

    Writes fused features back onto ``encoded_spconv_tensor`` for temporal
    fusion / sparse heads. Channel 1 stores the true per-cell maximum RCS
    (scatter_reduce), not a summed accumulation.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.detach = False

    def _build_rcs_map(self, points, batch_size, h_bev, w_bev, stride, device, dtype):
        rcs_map = torch.zeros((batch_size, 2, h_bev, w_bev), device=device, dtype=dtype)
        if points is None or points.shape[0] == 0:
            return rcs_map
        points = points.to(device=device)
        bs_idx = points[:, 0].long(); x = points[:, 1]; y = points[:, 2]
        rcs = points[:, self.rcs_feature_index]
        pcx0, pcy0 = self.point_cloud_range[0], self.point_cloud_range[1]
        vx = float(self.base_voxel_size[0]) * float(stride); vy = float(self.base_voxel_size[1]) * float(stride)
        ix = torch.floor((x - pcx0) / vx).long(); iy = torch.floor((y - pcy0) / vy).long()
        valid = (bs_idx >= 0) & (bs_idx < batch_size) & (ix >= 0) & (ix < w_bev) & (iy >= 0) & (iy < h_bev)
        if valid.sum() == 0:
            return rcs_map
        bs_idx, ix, iy, rcs = bs_idx[valid], ix[valid], iy[valid], rcs[valid].to(dtype=dtype)
        linear = bs_idx * (h_bev * w_bev) + iy * w_bev + ix
        n = batch_size * h_bev * w_bev
        rcs_sum = torch.zeros(n, device=device, dtype=dtype).index_add_(0, linear, rcs)
        rcs_cnt = torch.zeros(n, device=device, dtype=dtype).index_add_(0, linear, torch.ones_like(rcs))
        rcs_max = torch.full((n,), float('-inf'), device=device, dtype=dtype)
        rcs_max = rcs_max.scatter_reduce(0, linear, rcs, reduce='amax', include_self=True)
        occ = rcs_cnt > 0
        rcs_max = torch.where(occ, rcs_max, torch.zeros_like(rcs_max))   # empty cells = 0, as before
        rcs_map[:, 0] = (rcs_sum / (rcs_cnt + 1e-6)).view(batch_size, h_bev, w_bev)
        rcs_map[:, 1] = rcs_max.view(batch_size, h_bev, w_bev)
        if self.rcs_map_mode == 'real':
            return rcs_map
        if self.rcs_map_mode == 'zeros':
            return torch.zeros_like(rcs_map)
        return self._shuffle_occupied_cells(rcs_map, occ.view(batch_size, h_bev, w_bev))

    def forward(self, batch_dict):
        batch_dict = super().forward(batch_dict)
        encoded = batch_dict['encoded_spconv_tensor']
        feats2d = batch_dict['spatial_features_2d']
        idx = encoded.indices.long()
        if idx.shape[1] != 3:
            raise RuntimeError(f'encoded_spconv_tensor indices must be (b, y, x), got {tuple(idx.shape)}')
        fused = feats2d.permute(0, 2, 3, 1)[idx[:, 0], idx[:, 1], idx[:, 2]]
        batch_dict['encoded_spconv_tensor'] = replace_feature(encoded, fused.contiguous())
        return batch_dict
