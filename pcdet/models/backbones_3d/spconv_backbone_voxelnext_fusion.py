from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import voxel_aggregation_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 子流形+bn+relu
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        # 子流形+bn
        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)
        # 拼接+relu
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xVoxelNeXt(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # 子流形卷积
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block
        # 第一层 两个残差块
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )
        # 第二层(/2) 稀疏卷积下采样 后接两个残差块
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        # 第三层(/4) 稀疏卷积下采样 后接两个残差块
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )
        # 第四层(/8) 稀疏卷积下采样 后接两个残差块
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )
        # 第五层(/16) 稀疏卷积下采样 后接两个残差块
        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        # 第六层(/32) 稀疏卷积下采样 后接两个残差块
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

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
        x_conv3 = self.conv3(x_conv2) # [64, 11, 400, 352]
        x_conv4 = self.conv4(x_conv3) # [128, 6, 200, 176]
        x_conv5 = self.conv5(x_conv4) # [128, 3, 100, 88]
        x_conv6 = self.conv6(x_conv5) # [128, 2, 50, 44]

        centroids_all, centroid_voxel_idxs_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(batch_dict['points'],
            self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS,
            batch_dict['multi_scale_3d_strides'],
            self.voxel_size,
            self.point_cloud_range)
        point_features = {}
        point_coords = {}
        for feature_location in self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS:
            centroids = centroids_all[feature_location][:, :4]
            centroid_voxel_idxs = centroid_voxel_idxs_all[feature_location]
            x_conv = batch_dict['multi_scale_3d_features'][feature_location]

            overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroid_voxel_idxs, x_conv)

            x_conv_features = torch.zeros((centroids.shape[0], x_conv.features.shape[-1]), dtype=x_conv.features.dtype, device=centroids.device)
            x_conv_features[overlapping_voxel_feature_nonempty_mask] = x_conv.features[overlapping_voxel_feature_indices_nonempty]
            point_coords[feature_location] = centroids[overlapping_voxel_feature_nonempty_mask]
            coords = centroids[overlapping_voxel_feature_nonempty_mask]
            if feature_location==self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS[0]:
                point_features_fuse = self.crossattention_head(batch_dict, point_features=x_conv_features[overlapping_voxel_feature_nonempty_mask], point_coords=coords, layer_name="layer1")
                point_features[feature_location] = point_features_fuse
            else:
                point_features[feature_location] = x_conv_features[overlapping_voxel_feature_nonempty_mask]

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

        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

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

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        assert input_channels == 7 or input_channels == 11
        if input_channels == 7:
            input_channels = 4  # x y z rcs
        else:
            input_channels = 8 # x y z rcs score*4
        input_channels_velo = 3
        self.input_channels = input_channels
        self.input_channels_velo = input_channels_velo
        # 子流形卷积
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        # 速度组
        self.conv_input_velo = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels_velo, channels[0], 3, padding=1, bias=False, indice_key='subm1_velo'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        block = post_act_block
        # 第一层 两个残差块
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv1_velo = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1_velo'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1_velo'),
        )

        # 第二层(/2) 稀疏卷积下采样 后接两个残差块
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv2_velo = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2_velo', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2_velo'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2_velo'),
        )

        # 第三层(/4) 稀疏卷积下采样 后接两个残差块
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv3_velo = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3_velo', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3_velo'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3_velo'),
        )

        # 第四层(/8) 稀疏卷积下采样 后接两个残差块
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv4_velo = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4_velo', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4_velo'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4_velo'),
        )

        # 第五层(/16) 稀疏卷积下采样 后接两个残差块
        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv5_velo = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5_velo', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5_velo'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5_velo'),
        )

        # 第六层(/32) 稀疏卷积下采样 后接两个残差块
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv6_velo = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6_velo', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6_velo'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6_velo'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

        self.group_fuse = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel*2, out_channel, 1, stride=1, padding=0, bias=False, indice_key='subm_fuse'),
            norm_fn(out_channel),
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
        voxel_features_all, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        ################################ x y z rcs group #####################################
        voxel_features = voxel_features_all[:, :self.input_channels]
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x) # [16, 41, 1600, 1408]
        x_conv2 = self.conv2(x_conv1) # [32, 21, 800, 704]
        x_conv3 = self.conv3(x_conv2) # [64, 11, 400, 352]
        x_conv4 = self.conv4(x_conv3) # [128, 6, 200, 176]
        x_conv5 = self.conv5(x_conv4) # [128, 3, 100, 88]
        x_conv6 = self.conv6(x_conv5) # [128, 2, 50, 44]
        # 将x_conv4,x_conv5,x_conv6的特征合并起来
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)
        #########################################################################################

        ################################ vr vcomp time group #####################################
        voxel_features_velo = voxel_features_all[:, self.input_channels:]
        input_sp_tensor_velo = spconv.SparseConvTensor(
            features=voxel_features_velo,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_velo = self.conv_input_velo(input_sp_tensor_velo)

        x_conv1_velo = self.conv1_velo(x_velo) # [16, 41, 1600, 1408]
        x_conv2_velo = self.conv2_velo(x_conv1_velo) # [32, 21, 800, 704]
        x_conv3_velo = self.conv3_velo(x_conv2_velo) # [64, 11, 400, 352]
        x_conv4_velo = self.conv4_velo(x_conv3_velo) # [128, 6, 200, 176]
        x_conv5_velo = self.conv5_velo(x_conv4_velo) # [128, 3, 100, 88]
        x_conv6_velo = self.conv6_velo(x_conv5_velo) # [128, 2, 50, 44]
        # 将x_conv4,x_conv5,x_conv6的特征合并起来
        x_conv5_velo.indices[:, 1:] *= 2
        x_conv6_velo.indices[:, 1:] *= 4
        x_conv4_velo = x_conv4_velo.replace_feature(torch.cat([x_conv4_velo.features, x_conv5_velo.features, x_conv6_velo.features]))
        x_conv4_velo.indices = torch.cat([x_conv4_velo.indices, x_conv5_velo.indices, x_conv6_velo.indices])
        out_velo = self.bev_out(x_conv4_velo)
        #########################################################################################

        # fuse two group
        assert out.features.shape[0] == out_velo.features.shape[0]
        out = replace_feature(out, torch.cat([out.features, out_velo.features], dim=1))
        out = self.group_fuse(out)

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


    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.input_channels = 7
        self.input_channels_velo = input_channels
        # 子流形卷积
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        # 速度组
        self.conv_input_velo = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels_velo, channels[0], 3, padding=1, bias=False, indice_key='subm1_velo'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        block = post_act_block
        # 第一层 两个残差块
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv1_velo = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1_velo'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1_velo'),
        )

        # 第二层(/2) 稀疏卷积下采样 后接两个残差块
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv2_velo = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2_velo', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2_velo'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2_velo'),
        )

        # 第三层(/4) 稀疏卷积下采样 后接两个残差块
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv3_velo = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3_velo', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3_velo'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3_velo'),
        )

        # 第四层(/8) 稀疏卷积下采样 后接两个残差块
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv4_velo = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4_velo', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4_velo'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4_velo'),
        )

        # 第五层(/16) 稀疏卷积下采样 后接两个残差块
        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv5_velo = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5_velo', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5_velo'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5_velo'),
        )

        # 第六层(/32) 稀疏卷积下采样 后接两个残差块
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv6_velo = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6_velo', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6_velo'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6_velo'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

        self.group_fuse = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel*2, out_channel, 1, stride=1, padding=0, bias=False, indice_key='subm_fuse'),
            norm_fn(out_channel),
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
    
    def bev_out2d(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        spatial_shape = x_conv.spatial_shape

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
        voxel_features_all, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        ################################ x y z rcs group #####################################
        mask = voxel_features_all[:,-1].long()==1
        voxel_features = voxel_features_all[mask][:, :self.input_channels]
        voxel_coords_main = voxel_coords[mask]
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords_main.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x) # [16, 41, 1600, 1408]
        x_conv2 = self.conv2(x_conv1) # [32, 21, 800, 704]
        x_conv3 = self.conv3(x_conv2) # [64, 11, 400, 352]
        x_conv4 = self.conv4(x_conv3) # [128, 6, 200, 176]
        x_conv5 = self.conv5(x_conv4) # [128, 3, 100, 88]
        x_conv6 = self.conv6(x_conv5) # [128, 2, 50, 44]
        # 将x_conv4,x_conv5,x_conv6的特征合并起来
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])        
        #########################################################################################

        ################################ vr vcomp time group #####################################
        mask_sub = (voxel_features_all[:,-1].long()==0)|(voxel_features_all[:,-1].long()==-1)
        voxel_features_velo = voxel_features_all[mask_sub][:, :self.input_channels_velo]
        voxel_coords_sub = voxel_coords[mask_sub]
        if voxel_features_velo.numel() != 0:
            input_sp_tensor_velo = spconv.SparseConvTensor(
                features=voxel_features_velo,
                indices=voxel_coords_sub.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x_velo = self.conv_input_velo(input_sp_tensor_velo)

            x_conv1_velo = self.conv1_velo(x_velo) # [16, 41, 1600, 1408]
            x_conv2_velo = self.conv2_velo(x_conv1_velo) # [32, 21, 800, 704]
            x_conv3_velo = self.conv3_velo(x_conv2_velo) # [64, 11, 400, 352]
            x_conv4_velo = self.conv4_velo(x_conv3_velo) # [128, 6, 200, 176]
            x_conv5_velo = self.conv5_velo(x_conv4_velo) # [128, 3, 100, 88]
            x_conv6_velo = self.conv6_velo(x_conv5_velo) # [128, 2, 50, 44]
            # 将x_conv4,x_conv5,x_conv6的特征合并起来
            x_conv5_velo.indices[:, 1:] *= 2
            x_conv6_velo.indices[:, 1:] *= 4
            x_conv4_velo = x_conv4_velo.replace_feature(torch.cat([x_conv4_velo.features, x_conv5_velo.features, x_conv6_velo.features]))
            x_conv4_velo.indices = torch.cat([x_conv4_velo.indices, x_conv5_velo.indices, x_conv6_velo.indices])
        else:
            x_conv4_velo = spconv.SparseConvTensor(
                features=torch.zeros([0, x_conv4.features.shape[1]], dtype=x_conv4.features.dtype, device=x_conv4.features.device),
                indices=torch.zeros([0, x_conv4.indices.shape[1]], dtype=torch.int32, device=x_conv4.indices.device),
                spatial_shape=x_conv4.spatial_shape,
                batch_size=batch_size
            )
        #########################################################################################

        # fuse
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv4_velo.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv4_velo.indices])

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