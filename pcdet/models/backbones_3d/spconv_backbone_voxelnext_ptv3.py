# Based on Radar M3-Net (Y. Yang et al., Expert Systems with Applications, 2025). Released with permission.
import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import voxel_aggregation_utils
from ...utils.spconv_utils import replace_feature, spconv
from ..sparse_attention.point_transformer_v3 import VoxelAttnPTv3

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


class VoxelResBackBone8xVoxelNeXtPTv3(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.attn_loc = model_cfg.get('attn_loc', ['xcout'])

        self.mamba_loc = list(model_cfg.get('mamba_loc', []) or [])
        _mamba_requested = list(self.mamba_loc)
        if _mamba_requested:
            from pcdet.models.backbones_3d.mamba_voxel_block import MambaVoxelBlock
            _mamba_channels = {'xc3': 64, 'xc4': 128, 'xcout': 128}
            _use_fg = model_cfg.get('use_fg_scoring', False)
            _mamba_cfg = model_cfg.get('MAMBA_CFG', {}) or {}
            _stages = _mamba_cfg.get('MAMBA_STAGES', ['xc3', 'xc4', 'xcout'])
            if _stages is None:
                _stages = ['xc3', 'xc4', 'xcout']
            _stages = [str(s) for s in _stages]
            _unknown = [s for s in _stages if s not in _mamba_channels]
            if _unknown:
                raise ValueError(f'unsupported MAMBA_STAGES: {_unknown}')
            _stage_set = set(_stages)
            # Stages left out are absent: no block, no residual, no LayerNorm.
            self.mamba_loc = [loc for loc in _mamba_requested if loc in _stage_set]
            if self.mamba_loc:
                self.mamba_blocks = nn.ModuleDict({
                    loc: MambaVoxelBlock(
                        d_model=_mamba_channels[loc],
                        use_fg_scoring=_use_fg,
                        per_frame_sequence=_mamba_cfg.get('PER_FRAME_SEQUENCE', False),
                        scan_order=_mamba_cfg.get('SCAN_ORDER', 'native'),
                        scan_seed=_mamba_cfg.get('SCAN_SEED', 0),
                        bidirectional=_mamba_cfg.get('BIDIRECTIONAL', False),
                        mixer=_mamba_cfg.get('MIXER', 'mamba'),
                    )
                    for loc in self.mamba_loc
                    if loc in _mamba_channels
                })

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        input_channels = model_cfg.get('INPUT_CHANNELS', input_channels)
        # Explicit INPUT_CHANNELS counts the raw mean (scale 0) plus PFN outputs.
        # Dropping RCS shrinks only that raw mean by one channel.
        if model_cfg.get('INPUT_CHANNELS', None) is not None and not bool(model_cfg.get('RCS_IN_INPUT', True)):
            input_channels = int(input_channels) - 1
        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        self.lay_num = model_cfg.get('lay_num', 2)

        self.use_lay_embed = model_cfg.get('use_lay_embed', False)
        out_filters = channels[-1]
        num_layer_emb = self.lay_num + 1
        self.scale_embeddings = nn.Embedding(num_layer_emb, out_filters, _weight=torch.zeros(num_layer_emb, out_filters))

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

        self.conv7 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv7', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res7'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res7'),
        )
        self.conv8 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv8', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res8'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res8'),
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

        if 'xc1' in self.attn_loc or 'xc4' in self.attn_loc or 'xcout' in self.attn_loc:
            self.voxel_attn = nn.ModuleList([
                VoxelAttnPTv3(
                    enc_depths=(2,),
                    enc_channels=(32,),
                    enc_num_head=(8,),
                    enc_patch_size=(48,),
                    enable_flash=False,
                    indice_key='ptv3_1'
                ),
                VoxelAttnPTv3(
                    enc_depths=(2,),
                    enc_channels=(128,),
                    enc_num_head=(8,),
                    enc_patch_size=(48,),
                    enable_flash=False,
                    indice_key='ptv3_4'
                ),
                VoxelAttnPTv3(
                    enc_depths=(2,),
                    enc_channels=(128,),
                    enc_num_head=(8,),
                    enc_patch_size=(48,),
                    enable_flash=False,
                    indice_key='ptv3_out',
                )
            ])

        if 'xc2' in self.attn_loc or 'xc3' in self.attn_loc or 'xc5' in self.attn_loc or 'xc6' in self.attn_loc:
            self.voxel_attn_attnloc = nn.ModuleList([
            VoxelAttnPTv3(
                enc_depths=(2,),
                enc_channels=(32,),
                enc_num_head=(8,),
                enc_patch_size=(48,),
                enable_flash=False,
                indice_key='ptv3_2'
            ),
            VoxelAttnPTv3(
                enc_depths=(2,),
                enc_channels=(64,),
                enc_num_head=(8,),
                enc_patch_size=(48,),
                enable_flash=False,
                indice_key='ptv3_3'
            ),
            VoxelAttnPTv3(
                enc_depths=(2,),
                enc_channels=(128,),
                enc_num_head=(8,),
                enc_patch_size=(48,),
                enable_flash=False,
                indice_key='ptv3_5'
            ),
            VoxelAttnPTv3(
                enc_depths=(2,),
                enc_channels=(128,),
                enc_num_head=(8,),
                enc_patch_size=(48,),
                enable_flash=False,
                indice_key='ptv3_6'
            )
        ])

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
        if 'xc1' in self.attn_loc:
            x_conv1 = self.voxel_attn[0](x_conv1)

        x_conv2 = self.conv2(x_conv1) # [32, 21, 800, 704]
        if 'xc2' in self.attn_loc:
            x_conv2 = self.voxel_attn_attnloc[0](x_conv2)

        x_conv3 = self.conv3(x_conv2) # [64, 11, 400, 352]
        if 'xc3' in self.attn_loc:
            x_conv3 = self.voxel_attn_attnloc[1](x_conv3)

        if 'xc3' in self.mamba_loc:
            x_conv3 = self.mamba_blocks['xc3'](x_conv3)

        x_conv4 = self.conv4(x_conv3) # [128, 6, 200, 176]
        if 'xc4' in self.attn_loc:
            x_conv4 = self.voxel_attn[1](x_conv4)

        if 'xc4' in self.mamba_loc:
            x_conv4 = self.mamba_blocks['xc4'](x_conv4)

        x_conv5 = self.conv5(x_conv4) # [128, 3, 100, 88]
        if 'xc5' in self.attn_loc:
            x_conv5 = self.voxel_attn_attnloc[2](x_conv5)

        x_conv6 = self.conv6(x_conv5) # [128, 2, 50, 44]
        if 'xc6' in self.attn_loc:
            x_conv6 = self.voxel_attn_attnloc[3](x_conv6)

        if self.lay_num >= 3:
            x_conv7 = self.conv7(x_conv6)
        if self.lay_num >= 4:
            x_conv8 = self.conv8(x_conv7)

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
        if self.lay_num == 0:
            pass

        if self.lay_num == 1:
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features]))
            x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices])
        if self.lay_num == 2:
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        if self.lay_num == 3:
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features, x_conv7.features]))
            x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices, x_conv7.indices])
        if self.lay_num == 4:
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features, x_conv7.features, x_conv8.features]))
            x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices, x_conv7.indices, x_conv8.indices])
                    
        if 'xcout' in self.attn_loc:
            x_conv4 = self.voxel_attn[2](x_conv4)

        if 'xcout' in self.mamba_loc:
            x_conv4 = self.mamba_blocks['xcout'](x_conv4)

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
