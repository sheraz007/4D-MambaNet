from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from .utils import misc
import random
# from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .block import Block
# from .build import MODELS


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3) # 512, 32, 3
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        xyz = xyz.contiguous()
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None): # b n_query channel
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

class Config:
    def __init__(self, NAME, trans_dim, depth, cls_dim, num_heads, group_size, 
                 num_group, encoder_dims, rms_norm, drop_path, drop_out):
        self.NAME = NAME
        self.trans_dim = trans_dim
        self.depth = depth
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.rms_norm = rms_norm
        self.drop_path = drop_path
        self.drop_out = drop_out

class PointMamba_ori(nn.Module):
    def __init__(self, **kwargs):
        super(PointMamba, self).__init__()

        config= Config(
            NAME="PointMamba",  # Specify as a string
            trans_dim=384,
            depth=12,
            cls_dim=40,
            num_heads=6,
            group_size=32,
            num_group=64,
            encoder_dims=384,
            rms_norm=False,
            drop_path=0.3,
            drop_out=0.0
        )
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out)
        # self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts[...,:3]) # 8 64 32 3; 8 64 3
        group_input_tokens = self.encoder(neighborhood)  # 8 64(邻域数目) 384
        pos = self.pos_embed(center) # 8 64 384

        # reordering strategy 融合不同排序方式的信息
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
        group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
            1, 1, group_input_tokens.shape[-1])))
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],dim=1) # 8 64*3 384
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1) # 8 64*3 384---8 192 384

        x = group_input_tokens
        # transformer
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = x[:, :].mean(1) # 8 384
        ret = self.cls_head_finetune(concat_f) # 8 40
        return ret

class PointMamba(nn.Module):
    def __init__(self, **kwargs):
        super(PointMamba, self).__init__()

        config= Config(
            NAME="PointMamba",  # Specify as a string
            trans_dim=128, # 384,
            depth=12,
            cls_dim=40,
            num_heads=6,
            group_size=32,
            num_group=64,
            encoder_dims=128, # 384,
            rms_norm=False,
            drop_path=0.3,
            drop_out=0.0
        )
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.drop_out = nn.Dropout(config.drop_out)
        # self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def l2b(self, group_input_tokens_, center_xyz, batch_l):
        batch = int(batch_l[-1]) + 1
        feat_list = []
        center_list = []
        len_list = []
        for i in range(batch):
            mask = (batch_l == i)
            cur_batch_feat = group_input_tokens_[mask]
            cur_batch_center = center_xyz[mask]
            len_list.append(cur_batch_center.shape[0])
            feat_list.append(cur_batch_feat)
            center_list.append(cur_batch_center)
        max_len = max(len_list)
        feat_batch = torch.zeros((batch, max_len, group_input_tokens_.shape[-1])).cuda()
        center_batch = torch.zeros((batch, max_len, center_xyz.shape[-1])).cuda()
        for i in range(batch):
            feat_batch[i, :len_list[i], :] = feat_list[i]
            center_batch[i, :len_list[i], :] = center_list[i]
        return feat_batch, center_batch, len_list
    
    def b2l(self, x, center, len_list):
        feat_list = []
        center_list = []
        for i in range(x.shape[0]):
            cur_batch_feat = x[i, :len_list[i], :]
            cur_center = center[i, :len_list[i], :]
            feat_list.append(cur_batch_feat)
            center_list.append(center)
        feat = torch.cat(feat_list, dim=0)
        center_ = torch.cat(center_list, dim=0)
        return feat, center_

    def forward(self, sp_tensor):
        group_input_tokens_ = sp_tensor.features # bs, n_p, 128
        center_ = sp_tensor.indices[:,[3,2,1]].float() # bs, n_p, 3 xyz
        batch_l = sp_tensor.indices[:, 0]
        batch = int(batch_l[-1]) + 1
        feat_list = []
        indice_list = []
        for cur_batch_id in range(batch):
            # group_input_tokens, center, len_list = self.l2b(group_input_tokens_, center_, batch_l)
            mask_batch = (batch_l == cur_batch_id)
            group_input_tokens = group_input_tokens_[mask_batch].unsqueeze(0)
            center = center_[mask_batch].unsqueeze(0)
            pos = self.pos_embed(center)

            # reordering strategy 融合不同排序方式的信息
            center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
            center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
            center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
            group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
                1, 1, group_input_tokens.shape[-1])))
            pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
            pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
            pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))

            center_x_ = center.gather(dim=1, index=torch.tile(center_x, (1, 1, center.shape[-1])))
            center_y_ = center.gather(dim=1, index=torch.tile(center_y, (1, 1, center.shape[-1])))
            center_z_ = center.gather(dim=1, index=torch.tile(center_z, (1, 1, center.shape[-1])))
            center = torch.cat([center_x_, center_y_, center_z_], dim=1)

            group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],dim=1) # 8 64*3 384
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1) # 8 64*3 384---8 192 384

            # yyt_c
            choice_mode = random.randint(0, 2)
            just_use_x_order = True
            if choice_mode == 0:
                group_input_tokens = group_input_tokens_x
                pos = pos_x
                center = center_x_
            if choice_mode == 1:
                group_input_tokens = group_input_tokens_y
                pos = pos_y
                center = center_y_
            if choice_mode == 2:
                group_input_tokens = group_input_tokens_z
                pos = pos_z
                center = center_z_                                
            indice_list.append(torch.cat((cur_batch_id * torch.ones((center.shape[1],1)).cuda(), center[0,:,[2,1,0]]), dim=-1).int())

            x = group_input_tokens
            # transformer
            x = self.drop_out(x)
            x = self.blocks(x, pos)
            x = self.norm(x)
            # concat_f = x[:, :].mean(1) # 8 384
            # ret = self.cls_head_finetune(concat_f) # 8 40
            feat_list.append(x.squeeze(0))

        # sp_feature, center = self.b2l(x, center, len_list)
        sp_feature = torch.cat(feat_list)
        sp_tensor = sp_tensor.replace_feature(sp_feature)
        sp_tensor.indices = torch.cat(indice_list)
        return sp_tensor
