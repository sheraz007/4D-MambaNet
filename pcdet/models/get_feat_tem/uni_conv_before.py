import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv

class UniConv_before(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.downsample_times = 8
        self.voxel_size = 0.05
        self.point_cloud_range = torch.tensor([0, -25.6, -3, 51.2, 25.6, 2]).cuda()

        self.sta_con_in_bev = model_cfg.get('sta_con_in_bev', False)
        input_channels = model_cfg.get('input_channels', 128)
        output_channels = model_cfg.get('output_channels', 128)
        self.xy_ks = model_cfg.get('xy_ks', 7)
        self.z_ks = model_cfg.get('z_ks', 3)
        
        self.st_conv = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, output_channels, [self.z_ks, self.xy_ks, self.xy_ks], stride=1, padding=[0, self.xy_ks//2, self.xy_ks//2], bias=False, indice_key='st_conv'),
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

    def forward(self, data_dict):
        prev_feats = data_dict['before_feat_list']
        sparse_tensor = data_dict['encoded_spconv_tensor']
        
        indices_prev = []
        indices_cur = sparse_tensor.indices
        indices_cur = torch.cat([indices_cur[:, 0:1], torch.zeros((indices_cur.shape[0], 1), dtype=indices_cur.dtype, device=indices_cur.device), indices_cur[:, 1:3]], dim=1)
        for k, prev_feat in enumerate(prev_feats):
            indices = prev_feat.indices
            indices = torch.cat([indices[:, 0:1], (k+1)+torch.zeros((indices.shape[0], 1), dtype=indices.dtype, device=indices.device), indices[:, 1:3]], dim=1)
            indices_prev.append(indices)

        indices_all = torch.cat([indices_cur, *indices_prev], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat([sparse_tensor.features, *[prev_feat.features for prev_feat in prev_feats]], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = spconv.SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[len(prev_feats)+1, *sparse_tensor.spatial_shape],
            batch_size=sparse_tensor.batch_size
        )
        st_tensor = self.st_conv(st_tensor)
        st_tensor = self.bev_out(st_tensor)

        data_dict['encoded_spconv_tensor'] = st_tensor
        return data_dict
