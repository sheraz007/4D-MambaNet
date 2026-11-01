import numpy as np
import torch
import torch.nn as nn
from ...utils.spconv_utils import replace_feature, spconv

class BEV_concat(nn.Module):
    def __init__(self, model_cfg, input_channels=128):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = input_channels
        self.tem_num = model_cfg.get('tem_num', 3)

        self.temporal_conv_list = nn.ModuleList()
        for i in range(self.tem_num):
            conv_module = spconv.SparseSequential(
                    spconv.SparseConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=True),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU(True),
                )
            self.temporal_conv_list.append(conv_module)
                
    def get_unique(self, features_cat, indices_cat, spatial_shape, batch_size):
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        return x_out
        
    def forward(self, data_dict):
        out_all_frames = data_dict['encoded_spconv_tensor']
        
        time_feature_list = []
        time_feature_list.append(out_all_frames)
        time_feature_list += data_dict['before_feat_list']
        
        all_time_feat = []
        
        for i in range(len(time_feature_list)):
            cur_time_feature = self.temporal_conv_list[i](time_feature_list[i])
            all_time_feat.append(cur_time_feature)

        time_aug_feat_features = torch.cat([all_time_feat_cur.features for all_time_feat_cur in all_time_feat])
        time_aug_feat_indices = torch.cat([all_time_feat_cur.indices for all_time_feat_cur in all_time_feat])
        time_aug_feat = self.get_unique(time_aug_feat_features, time_aug_feat_indices, out_all_frames.spatial_shape, out_all_frames.batch_size)
        data_dict['encoded_spconv_tensor'] = time_aug_feat
        return data_dict