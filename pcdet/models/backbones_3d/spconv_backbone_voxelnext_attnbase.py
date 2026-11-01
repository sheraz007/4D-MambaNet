import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import voxel_aggregation_utils
from ...utils.spconv_utils import replace_feature, spconv
from .attn_layer.VoxelAttn_Layer import VoxelAttn_Layer
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .attn_layer.ptv3.point_transformer_v3 import PointTransformerV3

class VoxelResBackBone8xVoxelNeXtattn(VoxelResBackBone8xVoxelNeXt):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        self.model_cfg = model_cfg
        self.attn_type = model_cfg.get('attn_type', 'ptv3')
        self.attn_loc = model_cfg.get('attn_loc', [7])
        attn_channels = model_cfg.get('attn_channels', [32, 32, 64, 128, 128, 128, 128])
        self.lay_num = model_cfg.get('lay_num', 2)

        self.use_lay_embed = model_cfg.get('use_lay_embed', False)

        out_filters = attn_channels[-1]
        num_layer_emb = self.lay_num + 1
        self.scale_embeddings = nn.Embedding(num_layer_emb, out_filters, _weight=torch.zeros(num_layer_emb, out_filters))

        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # 子流形卷积

        self.num_point_features = out_channel
       
        self.voxel_attn = nn.ModuleList()
        for i in range(len(self.attn_loc)):
            cur_attn_loc = self.attn_loc[i]
            cur_lay_attn = VoxelAttn_Layer(
                                attn_type=self.attn_type,
                                enc_depths=(2,),
                                enc_channels=(attn_channels[cur_attn_loc-1],),
                                enc_num_head=(8,),
                                enc_patch_size=(48,),
                                enable_flash=False,
                                indice_key='attn_{}'.format(i)
                            )
            self.voxel_attn.append(cur_lay_attn)

    def l2b(self, points):
        batch = int(points[-1,0]) + 1
        point_len = []
        point_list = []
        for i in range(batch):
            cur_batch_point = points[points[:,0] == i][:,1:]
            point_list.append(cur_batch_point)
            point_len.append(cur_batch_point.shape[0])
        max_len = max(point_len)
        batch_point = torch.zeros((batch, max_len, points.shape[-1] - 1)).cuda()
        for i in range(batch):
            batch_point[i, :point_len[i], :] = point_list[i]
        return batch_point

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        cur_lay_id = 0

        # mamba test
        debug_mamba = False
        if debug_mamba:
            batch_dict_p = self.l2b(batch_dict['points'])
            x = self.voxel_attn[cur_lay_id](batch_dict_p)

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x) # [16, 41, 1600, 1408]

        x_conv2 = self.conv2(x_conv1) # [32, 21, 800, 704]
        if 2 in self.attn_loc:
            x_conv2 = self.voxel_attn[cur_lay_id](x_conv2)
            cur_lay_id += 1

        x_conv3 = self.conv3(x_conv2) # [64, 11, 400, 352]
        if 3 in self.attn_loc:
            x_conv3 = self.voxel_attn[cur_lay_id](x_conv3)
            cur_lay_id += 1

        x_conv4 = self.conv4(x_conv3) # [128, 6, 200, 176]


        x_conv5 = self.conv5(x_conv4) # [128, 3, 100, 88]
        if 5 in self.attn_loc:
            x_conv5 = self.voxel_attn[cur_lay_id](x_conv5)
            cur_lay_id += 1

        x_conv6 = self.conv6(x_conv5) # [128, 2, 50, 44]
        if 6 in self.attn_loc:
            x_conv6 = self.voxel_attn[cur_lay_id](x_conv6)
            cur_lay_id += 1

        if self.model_cfg.get('DEBUG', False):
            pts:torch.Tensor = batch_dict['points'][:, 1:4]
            # inverse aug   # 将数据增强后的点云变为增强前
            pts -= batch_dict['lidar_aug_matrix'][0][:3, 3]
            pts = torch.inverse(batch_dict['lidar_aug_matrix'][0][:3, :3]).matmul(
                pts.transpose(1, 0)
            ).T
            pts = torch.hstack((pts, torch.ones([pts.shape[0], 1]).to(pts.device)))
            pts = torch.matmul(pts, batch_dict['trans_lidar_to_cam'][0].T)
            pts_2d_hom = torch.matmul(pts, batch_dict['trans_cam_to_img'][0].T)
            pts_img = (pts_2d_hom[:, 0:2].T / pts[:, 2]).T  # (N, 2)
            pts_img = pts_img.cpu().numpy()

            x_conv4_dense = x_conv4.dense().detach()[0].sum(dim=(0,1))
            x_conv5_dense = x_conv5.dense().detach()[0].sum(dim=(0,1))
            x_conv6_dense = x_conv6.dense().detach()[0].sum(dim=(0,1))
            import matplotlib.pyplot as plt
            plt.subplot(3,1,1)
            plt.imshow(x_conv4_dense.cpu().numpy())
            plt.subplot(3,1,2)
            plt.imshow(x_conv5_dense.cpu().numpy())
            plt.subplot(3,1,3)
            plt.imshow(x_conv6_dense.cpu().numpy())
            plt.savefig('a.png')
    
        # 将x_conv4,x_conv5,x_conv6的特征合并起来
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        if self.lay_num >= 3:
            x_conv7.indices[:, 1:] *= 8
        if self.lay_num >= 4:
            x_conv8.indices[:, 1:] *= 16
            
        if self.use_lay_embed:
            x_conv4 = x_conv4.replace_feature(x_conv4.features + self.scale_embeddings(torch.tensor(0).long().cuda()))
            if self.lay_num >= 1:
                x_conv5 = x_conv5.replace_feature(x_conv5.features + self.scale_embeddings(torch.tensor(1).long().cuda()))
            if self.lay_num >= 2:
                x_conv6 = x_conv6.replace_feature(x_conv6.features + self.scale_embeddings(torch.tensor(2).long().cuda()))
            if self.lay_num >= 3:
                x_conv7 = x_conv7.replace_feature(x_conv7.features + self.scale_embeddings(torch.tensor(3).long().cuda()))
            if self.lay_num >= 4:
                x_conv8 = x_conv8.replace_feature(x_conv8.features + self.scale_embeddings(torch.tensor(4).long().cuda()))                

        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        if 7 in self.attn_loc:
            x_conv4 = self.voxel_attn[cur_lay_id](x_conv4)

        out = self.bev_out(x_conv4)
        # 2d稀疏卷积[200, 176]
        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
