import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv

class UniConv(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.downsample_times = 8
        self.voxel_size = 0.05
        self.point_cloud_range = torch.tensor([0, -25.6, -3, 51.2, 25.6, 2]).cuda()

        self.sta_con_in_bev = model_cfg.get('sta_con_in_bev', False)
        self.input_channels = model_cfg.get('input_channels', 128)
        output_channels = model_cfg.get('output_channels', 128)
        self.xy_ks = model_cfg.get('xy_ks', 7)
        self.z_ks = model_cfg.get('z_ks', 3)

        self.layer_num = model_cfg.get('layer_num', 1)
        if self.layer_num > 1:
            self.st_conv_list = nn.Sequential()
            for i in range(self.layer_num):
                # if i == self.layer_num - 1:
                #     pad_z = 0
                # else:
                pad_z = self.z_ks//2
                st_conv = spconv.SparseSequential(
                    spconv.SparseConv3d(self.input_channels, output_channels, [self.z_ks, self.xy_ks, self.xy_ks], stride=1, padding=[pad_z, self.xy_ks//2, self.xy_ks//2], bias=False, indice_key='st_conv_{}'.format(i)),
                    nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
                self.st_conv_list.append(st_conv)
        else:
            self.st_conv = spconv.SparseSequential(
                spconv.SparseConv3d(self.input_channels, output_channels, [self.z_ks, self.xy_ks, self.xy_ks], stride=1, padding=[0, self.xy_ks//2, self.xy_ks//2], bias=False, indice_key='st_conv'),
                nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def grid2point(self, voxel_coords): # tensor
        '''
        spatial_indice(bs,zyx)---point(bs, xyz)
        '''
        voxel_centers = voxel_coords[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = self.voxel_size * self.downsample_times
        pc_range = self.point_cloud_range[0:3].float()
        #voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        # mao check一下这里应该是不用加0.5的
        voxel_centers = (voxel_centers) * voxel_size + pc_range
        voxel_centers = torch.cat((voxel_coords[:, 0:1], voxel_centers), dim=1)
        return voxel_centers
        
    def point2grid(self, points): # tensor
        if points.shape[-1] == 3:
            return (points - self.point_cloud_range[[0,1,2]]) / (self.voxel_size * self.downsample_times)
        else:
            grid = (points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / (self.voxel_size * self.downsample_times)
            grid = torch.cat((points[:, 0:1], grid), dim=1)
        return grid

    def static_com(self, radar2_radar1, point_cur): # point_cur:=bs xyz
        # sta_com
        if isinstance(point_cur, np.ndarray) == False:
            point_cur = point_cur.cpu().numpy()
        radar1_radar2 = np.linalg.inv(radar2_radar1)
        prev_point = np.concatenate((point_cur[:,1:4], np.ones((point_cur.shape[0], 1))), axis=1) @ radar1_radar2.T
        prev_point = torch.tensor(prev_point[:, :-1]).cuda()
        prev_point = torch.cat((torch.tensor(point_cur[:,0:1]).cuda(), prev_point), dim=1)
        return prev_point
    
    def gen_before_grid_from_cur(self, grid_cur_frame, radar2_radar1):
        point_cur = self.grid2point(grid_cur_frame[:, [0, 3, 2, 1]])
        prev_point = self.static_com(radar2_radar1, point_cur)
        prev_gird = self.point2grid(prev_point)
        prev_grid = prev_gird[:,1:3] # xyz
        return prev_grid

    def gen_cur_grid(self, w, h):
        xs = torch.linspace(
            0, w - 1, w).view(1, w).expand(h, w).cuda()
        ys = torch.linspace(
            0, h - 1, h).view(h, 1).expand(h, w).cuda()
        grid_cur_frame = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid_cur_frame = grid_cur_frame.view(-1, 3)
        grid_cur_frame = torch.cat((torch.zeros((grid_cur_frame.shape[0], 1)).cuda(), grid_cur_frame), dim=1)
        return grid_cur_frame
    
    def get_before_feat(self, radar2_radar1, before_feat):
        w, h = before_feat.shape[-1], before_feat.shape[-2]
        grid_cur_frame = self.gen_cur_grid(w, h)
        prev_grid = self.gen_before_grid_from_cur(grid_cur_frame, radar2_radar1)
        prev_grid[:, 0] = prev_grid[:, 0] * 2 / (w - 1) - 1 # only one batch data input
        prev_grid[:, 1] = prev_grid[:, 1] * 2 / (h - 1) - 1 # only one batch data input
        prev_grid = prev_grid.view(1, h, w, 2)
        output = F.grid_sample(before_feat.unsqueeze(0), prev_grid.to(before_feat.dtype), align_corners=True)
        return output

    def forward(self, data_dict):
        prev_feats = data_dict['before_feat_list']
        sparse_tensor = data_dict['encoded_spconv_tensor']
        
        indices_prev = []
        feat_prev_list = []
        indices_cur = sparse_tensor.indices
        indices_cur = torch.cat([indices_cur[:, 0:1], torch.zeros((indices_cur.shape[0], 1), dtype=indices_cur.dtype, device=indices_cur.device), indices_cur[:, 1:3]], dim=1)
        
        indices_prev_list = []

        for k, prev_feat in enumerate(prev_feats):
            feat_list = []
            indice_list = []
            before_feat = prev_feat.dense()

            for bs in range(data_dict['batch_size']):
                feat_prev = self.get_before_feat(data_dict['radar2_radar1_list'][bs][k], before_feat[bs]) # return bs c h w

                feat_prev = feat_prev[0]
                non_empty_indice = feat_prev.any(0).nonzero() # h w
                non_empty_feats = feat_prev[:, non_empty_indice[:, 0], non_empty_indice[:, 1]]
                feat_list.append(non_empty_feats.permute(1, 0))
                indice = torch.cat((bs * torch.ones((non_empty_indice.shape[0], 1),device=before_feat.device), non_empty_indice), dim=1)
                indice_list.append(indice)

            feat_prev = torch.cat(feat_list, dim=0)
            indices_prev = torch.cat(indice_list, dim=0)
            feat_prev_list.append(feat_prev)
            # add prev height
            indices_prev_list.append(torch.cat([indices_prev[:, 0:1], (k + 1) + torch.zeros((indices_prev.shape[0], 1), dtype=indices_prev.dtype, device=indices_prev.device), indices_prev[:, 1:]], dim=1))

        indices_all = torch.cat([indices_cur, *indices_prev_list], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat([sparse_tensor.features, *feat_prev_list], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = spconv.SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[len(prev_feats)+1, *sparse_tensor.spatial_shape],
            batch_size=sparse_tensor.batch_size
        )

        if self.layer_num > 1:
            for lay_id in range(self.layer_num):
                st_tensor = self.st_conv_list[lay_id](st_tensor)
        else:
            st_tensor = self.st_conv(st_tensor)

        st_tensor = self.bev_out(st_tensor)

        data_dict['encoded_spconv_tensor'] = st_tensor

        return data_dict


class UniConvDoppler(UniConv):
    def forward(self, data_dict):
        prev_feats = data_dict['before_feat_list']
        sparse_tensor = data_dict['encoded_spconv_tensor']
        
        indices_prev = []
        feat_prev_list = []
        indices_cur = sparse_tensor.indices
        indices_cur = torch.cat([
            indices_cur[:, 0:1],
            torch.zeros((indices_cur.shape[0], 1),
                        dtype=indices_cur.dtype,
                        device=indices_cur.device),
            indices_cur[:, 1:3]
        ], dim=1)
        
        indices_prev_list = []

        # Build Doppler confidence map from current frame points
        # vrc channel 5 (0-indexed after batch dim): absolute Doppler
        # High |vrc| = moving object = previous frame position unreliable
        points = data_dict['points']  # [N, 8] bs,x,y,z,RCS,vr,vrc,t
        batch_size = data_dict['batch_size']
        w_bev = sparse_tensor.spatial_shape[-1]
        h_bev = sparse_tensor.spatial_shape[-2]
        voxel_stride = self.voxel_size * self.downsample_times
        pc_range = self.point_cloud_range

        # Build per-batch motion confidence map [bs, 1, H, W]
        # Value near 0 = moving (unreliable temporal) 
        # Value near 1 = static (reliable temporal)
        confidence_maps = []
        for bs in range(batch_size):
            mask = points[:, 0] == bs
            pts = points[mask]

            # Start with full confidence everywhere
            conf_map = torch.ones(
                1, h_bev, w_bev,
                dtype=sparse_tensor.features.dtype,
                device=sparse_tensor.features.device)

            if pts.shape[0] > 0:
                x = pts[:, 1]
                y = pts[:, 2]
                vrc = pts[:, 6]

                # Moving points: |vrc| > 0.5 m/s
                moving_mask = vrc.abs() > 0.5
                if moving_mask.sum() > 0:
                    x_mov = pts[moving_mask, 1]
                    y_mov = pts[moving_mask, 2]
                    vrc_mov = pts[moving_mask, 6]

                    # Confidence inversely proportional to |vrc|
                    # |vrc|=0.5 -> conf=1.0, |vrc|=5.0 -> conf=0.1
                    motion_conf = 1.0 / (1.0 + vrc_mov.abs())

                    col = ((x_mov - pc_range[0]) / voxel_stride
                           ).long().clamp(0, w_bev - 1)
                    row = ((y_mov - pc_range[1]) / voxel_stride
                           ).long().clamp(0, h_bev - 1)

                    # Write minimum confidence at each moving cell
                    for idx in range(len(row)):
                        r = row[idx].item()
                        c = col[idx].item()
                        conf_map[0, r, c] = min(
                            conf_map[0, r, c].item(),
                            motion_conf[idx].item())

            confidence_maps.append(conf_map)

        if not hasattr(self, '_doppler_debug'):
            self._doppler_debug = True
            total_moving = sum(
                (1.0 - c).gt(0.1).sum().item()
                for c in confidence_maps)
            print(f"[DopplerUniConv] moving BEV cells: {total_moving}")
            print(f"[DopplerUniConv] batch_size: {batch_size}")

        for k, prev_feat in enumerate(prev_feats):
            feat_list = []
            indice_list = []
            before_feat = prev_feat.dense()

            for bs in range(batch_size):
                # Standard ego-motion warp (unchanged from UniConv)
                feat_prev = self.get_before_feat(
                    data_dict['radar2_radar1_list'][bs][k],
                    before_feat[bs])
                feat_prev = feat_prev[0]  # [C, H, W]

                # Apply Doppler confidence weighting:
                # Moving regions get downweighted in temporal features
                conf = confidence_maps[bs]  # [1, H, W]
                feat_prev = feat_prev * conf

                non_empty_indice = feat_prev.any(0).nonzero()
                non_empty_feats = feat_prev[
                    :, non_empty_indice[:, 0],
                    non_empty_indice[:, 1]]
                feat_list.append(non_empty_feats.permute(1, 0))
                indice = torch.cat((
                    bs * torch.ones(
                        (non_empty_indice.shape[0], 1),
                        device=before_feat.device),
                    non_empty_indice), dim=1)
                indice_list.append(indice)

            feat_prev = torch.cat(feat_list, dim=0)
            indices_prev = torch.cat(indice_list, dim=0)
            feat_prev_list.append(feat_prev)
            indices_prev_list.append(torch.cat([
                indices_prev[:, 0:1],
                (k + 1) + torch.zeros(
                    (indices_prev.shape[0], 1),
                    dtype=indices_prev.dtype,
                    device=indices_prev.device),
                indices_prev[:, 1:]
            ], dim=1))

        indices_all = torch.cat(
            [indices_cur, *indices_prev_list], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat(
            [sparse_tensor.features, *feat_prev_list], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = spconv.SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[
                len(prev_feats)+1,
                *sparse_tensor.spatial_shape],
            batch_size=sparse_tensor.batch_size)

        if self.layer_num > 1:
            for lay_id in range(self.layer_num):
                st_tensor = self.st_conv_list[lay_id](st_tensor)
        else:
            st_tensor = self.st_conv(st_tensor)

        st_tensor = self.bev_out(st_tensor)
        data_dict['encoded_spconv_tensor'] = st_tensor
        return data_dict

        """
        # Build Doppler motion map from current frame points
        # vrc is channel 5 (0-indexed) of points: [bs,x,y,z,RCS,vr,vrc,t]
        # vrc = absolute radial Doppler velocity in m/s
        points = data_dict['points']  # [N, 8]
        batch_size = data_dict['batch_size']
        w_bev = sparse_tensor.spatial_shape[-1]
        h_bev = sparse_tensor.spatial_shape[-2]
        voxel_stride = self.voxel_size * self.downsample_times
        pc_range = self.point_cloud_range

        # Build per-batch Doppler BEV map [bs, 2, H, W]
        # channel 0 = mean vx (x-component of motion)
        # channel 1 = mean vy (y-component of motion)
        doppler_maps = []
        for bs in range(batch_size):
            mask = points[:, 0] == bs
            pts = points[mask]  # [N_bs, 8]
            
            doppler_map = torch.zeros(
                2, h_bev, w_bev,
                dtype=sparse_tensor.features.dtype,
                device=sparse_tensor.features.device)
            count_map = torch.zeros(
                h_bev, w_bev,
                dtype=sparse_tensor.features.dtype,
                device=sparse_tensor.features.device)

            if pts.shape[0] > 0:
                x = pts[:, 1]
                y = pts[:, 2]
                vrc = pts[:, 6]  # absolute Doppler velocity

                # Only use moving points: |vrc| > 0.5 m/s
                moving_mask = vrc.abs() > 0.5
                if moving_mask.sum() > 0:
                    pts_mov = pts[moving_mask]
                    x_mov = pts_mov[:, 1]
                    y_mov = pts_mov[:, 2]
                    vrc_mov = pts_mov[:, 6]

                    # Convert Doppler radial velocity to xy components
                    # vrc is radial — project onto x and y axes
                    r = torch.sqrt(x_mov**2 + y_mov**2).clamp(min=1e-6)
                    vx = vrc_mov * (x_mov / r)
                    vy = vrc_mov * (y_mov / r)

                    # Map to BEV grid indices
                    col = ((x_mov - pc_range[0]) / voxel_stride
                           ).long().clamp(0, w_bev - 1)
                    row = ((y_mov - pc_range[1]) / voxel_stride
                           ).long().clamp(0, h_bev - 1)

                    # Accumulate velocity at each BEV cell
                    doppler_map[0].index_put_(
                        (row, col), vx, accumulate=True)
                    doppler_map[1].index_put_(
                        (row, col), vy, accumulate=True)
                    count_map.index_put_(
                        (row, col),
                        torch.ones_like(vx), accumulate=True)

                    # Average where multiple points hit same cell
                    valid = count_map > 0
                    doppler_map[0][valid] /= count_map[valid]
                    doppler_map[1][valid] /= count_map[valid]

            doppler_maps.append(doppler_map)

        for k, prev_feat in enumerate(prev_feats):
            feat_list = []
            indice_list = []
            before_feat = prev_feat.dense()

            # dt: time between frames in seconds
            # VoD radar runs at ~10Hz so dt ≈ 0.1s per frame
            dt = 0.1 * (k + 1)

            for bs in range(batch_size):
                # Step 1: standard ego-motion warp (unchanged)
                feat_prev = self.get_before_feat(
                    data_dict['radar2_radar1_list'][bs][k],
                    before_feat[bs])
                feat_prev = feat_prev[0]  # [C, H, W]

                # Step 2: Doppler correction for moving regions
                # Compute pixel displacement from Doppler velocity
                dmap = doppler_maps[bs]  # [2, H, W]
                vx_map = dmap[0]  # [H, W]
                vy_map = dmap[1]  # [H, W]

                # displacement in BEV pixels
                dx = (vx_map * dt / voxel_stride)  # [H, W]
                dy = (vy_map * dt / voxel_stride)  # [H, W]

                # Only apply where motion is significant
                motion_mag = (dx**2 + dy**2).sqrt()
                motion_mask = motion_mag > 0.5  # > 0.5 pixel shift

                if motion_mask.sum() > 0:
                    # Get current grid coordinates
                    h, w = feat_prev.shape[-2], feat_prev.shape[-1]
                    
                    # Create sampling grid with Doppler correction
                    xs = torch.linspace(0, w-1, w, device=feat_prev.device)
                    ys = torch.linspace(0, h-1, h, device=feat_prev.device)
                    # Older PyTorch versions may not support `indexing=` in torch.meshgrid.
                    grid_y, grid_x = torch.meshgrid(ys, xs)
                    
                    # Apply Doppler displacement to moving regions
                    grid_x_corrected = grid_x - dx
                    grid_y_corrected = grid_y - dy
                    
                    # Normalize to [-1, 1] for grid_sample
                    grid_x_norm = grid_x_corrected * 2 / (w - 1) - 1
                    grid_y_norm = grid_y_corrected * 2 / (h - 1) - 1
                    
                    # Only apply correction in motion regions
                    # blend: moving regions use corrected grid,
                    # static regions keep original ego-warped features
                    mask_float = motion_mask.float().unsqueeze(0)
                    
                    sampling_grid = torch.stack(
                        [grid_x_norm, grid_y_norm], dim=-1
                    ).unsqueeze(0)  # [1, H, W, 2]
                    
                    feat_doppler = F.grid_sample(
                        before_feat[bs].unsqueeze(0),
                        sampling_grid.to(before_feat.dtype),
                        align_corners=True
                    )[0]  # [C, H, W]
                    
                    # Blend: use Doppler-corrected features
                    # only in motion regions
                    feat_prev = (feat_prev * (1 - mask_float) +
                                 feat_doppler * mask_float)

                non_empty_indice = feat_prev.any(0).nonzero()
                non_empty_feats = feat_prev[
                    :, non_empty_indice[:, 0], non_empty_indice[:, 1]]
                feat_list.append(non_empty_feats.permute(1, 0))
                indice = torch.cat((
                    bs * torch.ones(
                        (non_empty_indice.shape[0], 1),
                        device=before_feat.device),
                    non_empty_indice), dim=1)
                indice_list.append(indice)

            feat_prev = torch.cat(feat_list, dim=0)
            indices_prev = torch.cat(indice_list, dim=0)
            feat_prev_list.append(feat_prev)
            indices_prev_list.append(torch.cat([
                indices_prev[:, 0:1],
                (k + 1) + torch.zeros(
                    (indices_prev.shape[0], 1),
                    dtype=indices_prev.dtype,
                    device=indices_prev.device),
                indices_prev[:, 1:]
            ], dim=1))

        indices_all = torch.cat([indices_cur, *indices_prev_list], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat(
            [sparse_tensor.features, *feat_prev_list], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = spconv.SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[len(prev_feats)+1, *sparse_tensor.spatial_shape],
            batch_size=sparse_tensor.batch_size
        )

        if self.layer_num > 1:
            for lay_id in range(self.layer_num):
                st_tensor = self.st_conv_list[lay_id](st_tensor)
        else:
            st_tensor = self.st_conv(st_tensor)

        st_tensor = self.bev_out(st_tensor)
        data_dict['encoded_spconv_tensor'] = st_tensor

        # One-time debug print
        if not hasattr(self, '_doppler_debug'):
            self._doppler_debug = True
            total_moving = sum(
                (d[0]**2 + d[1]**2).sqrt().gt(0.1).sum().item()
                for d in doppler_maps)
            print(f"[DopplerUniConv] moving BEV cells: {total_moving}")
            print(f"[DopplerUniConv] batch_size: {batch_size}")
            print(f"[DopplerUniConv] prev_frames: {len(prev_feats)}")

        # DEAD_CODE_REMOVED
        """
