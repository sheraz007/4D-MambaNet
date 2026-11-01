import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv

class UniConv_forward(nn.Module):
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
        
    def point2grid_floor(self, points): # tensor
        '''
        point(bs, xyz)---spatial_indice(bs,xyz)
        '''
        grid = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / (self.voxel_size * self.downsample_times))
        grid = torch.cat((points[:, 0:1], grid), dim=1)
        return grid

    def static_com_21(self, radar2_radar1, point_prev): # point_cur:=bs xyz
        # sta_com
        if isinstance(point_prev, np.ndarray) == False:
            point_prev = point_prev.cpu().numpy()
        # radar1_radar2 = np.linalg.inv(radar2_radar1)
        point_prev_cur = np.concatenate((point_prev[:,1:4], np.ones((point_prev.shape[0], 1))), axis=1) @ radar2_radar1.T
        point_prev_cur = torch.tensor(point_prev_cur[:, :-1]).cuda()
        point_prev_cur = torch.cat((torch.tensor(point_prev[:,0:1]).cuda(), point_prev_cur), dim=1)
        return point_prev_cur
        
    def get_before_feat(self, radar2_radar1, before_feat_indices):
        before_grid = torch.cat([before_feat_indices[:, 0:1], torch.zeros((before_feat_indices.shape[0], 1), dtype=before_feat_indices.dtype, device=before_feat_indices.device), before_feat_indices[:, 1:]], dim=1) #bs zyx
        before_point = self.grid2point(before_grid)
        before_point_cur = self.static_com_21(radar2_radar1, before_point)
        before_grid_cur = self.point2grid_floor(before_point_cur) # bs xyz
        before_indices_cur = before_grid_cur[:,[0,2,1]] # bs yx
        return before_indices_cur

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
            # before_feat = prev_feat.dense()

            for bs in range(data_dict['batch_size']):
                indices = prev_feat.indices
                before_feat_indices = indices[indices[:,0] == bs] # bs yx

                non_empty_indice = self.get_before_feat(data_dict['radar2_radar1_list'][bs][k], before_feat_indices) # return bs c h w
                non_empty_feats = prev_feat.features
                feat_list.append(non_empty_feats)
                indice_list.append(non_empty_indice)

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
