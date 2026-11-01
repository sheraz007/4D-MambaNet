import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, zeros_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned
from ..dense_heads.target_assigner.hungarian_assigner import HungarianAssigner3D, overlaps
from ...utils import loss_utils, common_utils, box_coder_utils
from ..model_utils import centernet_utils

class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1):
        super(MLP, self).__init__()
        
        self.linear = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.GELU()  # You can also try nn.GELU()
        #self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        #x = self.dropout(x)
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, use_cross_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if self.use_cross_attn:
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, query_pos=None, key=None, key_pos=None, key_padding_mask=None, self_attn_mask=None, cross_attn_mask=None):
        # NxCxP to PxNxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None and key_pos is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        
        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v, attn_mask=self_attn_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        if self.use_cross_attn:
            key = key.permute(2, 0, 1)
            query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                        key=self.with_pos_embed(key, key_pos_embed),
                                        value=self.with_pos_embed(key, key_pos_embed), 
                                        key_padding_mask=key_padding_mask, attn_mask=cross_attn_mask)[0]
            query = query + self.dropout2(query2)
            query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query



class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False, zero_init_keys=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                if zero_init_keys is not None:
                    assert isinstance(zero_init_keys, (list, tuple))
                    if cur_name in zero_init_keys:
                        zeros_(fc[-1].weight.data)
                        zeros_(fc[-1].bias.data)
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class DoubleHeadRefine(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, grid_size, point_cloud_range, voxel_size, **kwargs
    ):
        super(DoubleHeadRefine, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

        self.model_cfg = model_cfg
        # self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        # self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')
        self.code_size = len(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])
        self.encode_method = self.model_cfg.get('ENCODE_METHOD', 'vanilla_all')
        self.use_rank_loss = self.model_cfg.get('USE_RANK_LOSS', False)
        self.use_attn_mask = self.model_cfg.get('USE_ATTN_MASK', False)
        if self.encode_method == 'vanilla_all':
            hidden_channel = self.model_cfg.VX_CHANNELS+7+1+1 # self.model_cfg.HIDDEN_CHANNEL
        elif self.encode_method == 'vanilla_box':
            hidden_channel = 7+1+1
        elif self.encode_method == 'vanilla_feat':
            hidden_channel = self.model_cfg.VX_CHANNELS
        elif self.encode_method == 'subnets':
            self.model_cfg.HEAD_EMBEDDING_METHOD = 'onehot_subnet'
            self.label_subnet = MLP(self.num_classes, 16)
            self.score_subnet = MLP(1, 16)
            self.head_subnet = MLP(2, 16)
            self.box_subnet = MLP(7, 16)
            self.feat_subnet_vx = MLP(self.model_cfg.VX_CHANNELS, 64)
            self.feat_subnet_ac = MLP(self.model_cfg.AC_CHANNELS, 64)
            hidden_channel = 16 * 4 + 64
        else:
            raise NotImplementedError

        self.head_embedding_method = self.model_cfg.get('HEAD_EMBEDDING_METHOD', 'onehot_cat')
        if self.head_embedding_method == 'onehot_cat':
            hidden_channel += 2

        self.pre_encode_box = self.model_cfg.get('PRE_ENCODE_BOX', True)
        if self.pre_encode_box and self.encode_method != 'subnets':
            hidden_channel += self.num_classes-1

        # self.num_proposals = self.model_cfg.NUM_PROPOSALS
        # self.bn_momentum = self.model_cfg.BN_MOMENTUM
        # self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        # loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        # self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        # if not self.use_sigmoid_cls:
        #     self.num_classes += 1
        # self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        # self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        # self.loss_bbox = loss_utils.L1Loss()
        # self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        # self.loss_heatmap = loss_utils.GaussianFocalLoss()
        # self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        # # a shared convolution
        # self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        # layers = []
        # layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        # layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        # self.heatmap_head = nn.Sequential(*layers)
        # self.heatmap_head[-1].bias.data.fill_(-2.19)
        # self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        # heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias, zero_init_keys=('reg',))

        # self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.HUNGARIAN_ASSIGNER)

        self.box_coder = getattr(box_coder_utils, self.model_cfg.BOX_CODER)(
            **self.model_cfg.get('BOX_CODER_CONFIG', {})
        )

        self.head_embedding = nn.Embedding(2, hidden_channel)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.align_ac_to_vx = nn.Sequential(
            nn.Conv1d(self.model_cfg.AC_CHANNELS, self.model_cfg.VX_CHANNELS, 1, bias=False),
            nn.BatchNorm1d(self.model_cfg.VX_CHANNELS, eps=1e-3, momentum=0.01),
            nn.GELU(),
            nn.Dropout1d(0.1)
        )

        # # Position Embedding for Cross-Attention, which is re-used during training
        # x_size = self.grid_size[0] // self.feature_map_stride
        # y_size = self.grid_size[1] // self.feature_map_stride
        # self.bev_pos = self.create_2D_grid(x_size, y_size)

        # self.forward_ret_dict = {}

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, batch_dict):
        # batch_size = inputs.shape[0]
        # lidar_feat = self.shared_conv(inputs)

        # lidar_feat_flatten = lidar_feat.view(
        #     batch_size, lidar_feat.shape[1], -1
        # )
        # bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # # query initialization
        # dense_heatmap = self.heatmap_head(lidar_feat)
        # heatmap = dense_heatmap.detach().sigmoid()
        # padding = self.nms_kernel_size // 2
        # local_max = torch.zeros_like(heatmap)
        # local_max_inner = F.max_pool2d(
        #     heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        # )
        # local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # # for Pedestrian & Traffic_cone in nuScenes
        # if self.dataset_name == "nuScenes":
        #     local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
        #     local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # # for Pedestrian & Cyclist in Waymo
        # elif self.dataset_name == "Waymo":
        #     local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
        #     local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        # heatmap = heatmap * (heatmap == local_max)
        # heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
 
        # # top num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        # top_proposals_class = top_proposals // heatmap.shape[-1]
        # top_proposals_index = top_proposals % heatmap.shape[-1]
        # query_feat = lidar_feat_flatten.gather(
        #     index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
        #     dim=-1,
        # )

        pred_boxes_ac = batch_dict['pred_boxes_ac'].detach()
        pred_scores_ac = batch_dict['pred_scores_ac'].detach()
        pred_labels_ac = batch_dict['pred_labels_ac'].detach()
        pred_cls_ac = batch_dict['pred_cls_ac'].detach()
        pred_feats_ac = batch_dict['pred_feats_ac'].detach()
        pred_scores_logic_ac = inverse_sigmoid(pred_scores_ac)
        
        pred_boxes_vx = batch_dict['pred_boxes_vx'].detach()
        pred_scores_vx = batch_dict['pred_scores_vx'].detach()
        pred_labels_vx = batch_dict['pred_labels_vx'].detach()
        pred_cls_vx = batch_dict['pred_cls_vx'].detach()
        pred_feats_vx = batch_dict['pred_feats_vx'].detach()
        pred_scores_logic_vx = inverse_sigmoid(pred_scores_vx)

        # box特征编码
        if self.pre_encode_box:
            pred_boxes_vx_encoded = self.encode_bbox(pred_boxes_vx)
            pred_boxes_ac_encoded = self.encode_bbox(pred_boxes_ac)
            pred_scores_vx_encoded = pred_scores_vx.unsqueeze(-1)
            pred_scores_ac_encoded = pred_scores_ac.unsqueeze(-1)
            pred_labels_vx_encoded = F.one_hot(pred_labels_vx.long(), self.num_classes).float()
            pred_labels_ac_encoded = F.one_hot(pred_labels_ac.long(), self.num_classes).float()
        else:
            pred_boxes_vx_encoded = pred_boxes_vx
            pred_boxes_ac_encoded = pred_boxes_ac
            pred_scores_vx_encoded = pred_scores_logic_vx.unsqueeze(-1)
            pred_scores_ac_encoded = pred_scores_logic_ac.unsqueeze(-1)
            pred_labels_vx_encoded = pred_labels_vx.unsqueeze(-1)
            pred_labels_ac_encoded = pred_labels_ac.unsqueeze(-1)

        if self.use_attn_mask:
            boxes_two_head = torch.cat([pred_boxes_vx, pred_boxes_ac], dim=1)
            attn_mask = []
            for bs in range(batch_dict['batch_size']):
                attn_mask.append(overlaps(boxes_two_head[bs], boxes_two_head[bs]))
            attn_mask = torch.stack(attn_mask)
            attn_mask = (attn_mask < 0.01).expand(attn_mask.shape[0]*self.model_cfg.NUM_HEADS, -1, -1)
        else:
            attn_mask = None
            

        # (N, C, L)
        # TODO: 怎么encoder这些特征？
        if self.encode_method == 'vanilla_all':
            pred_feats_ac = self.align_ac_to_vx(pred_feats_ac.permute(0, 2, 1)).permute(0, 2, 1)
            query_feat_vx = torch.cat([pred_boxes_vx_encoded, pred_scores_vx_encoded, pred_labels_vx_encoded, pred_feats_vx], dim=-1).permute(0, 2, 1)
            query_feat_ac = torch.cat([pred_boxes_ac_encoded, pred_scores_ac_encoded, pred_labels_ac_encoded, pred_feats_ac], dim=-1).permute(0, 2, 1)
        elif self.encode_method == 'vanilla_box':
            pred_feats_ac = self.align_ac_to_vx(pred_feats_ac.permute(0, 2, 1)).permute(0, 2, 1)
            query_feat_vx = torch.cat([pred_boxes_vx_encoded, pred_scores_vx_encoded, pred_labels_vx_encoded], dim=-1).permute(0, 2, 1)
            query_feat_ac = torch.cat([pred_boxes_ac_encoded, pred_scores_ac_encoded, pred_labels_ac_encoded], dim=-1).permute(0, 2, 1)
        elif self.encode_method == 'vanilla_feat':
            pred_feats_ac = self.align_ac_to_vx(pred_feats_ac.permute(0, 2, 1)).permute(0, 2, 1)
            query_feat_vx = pred_feats_vx.permute(0, 2, 1)
            query_feat_ac = pred_feats_ac.permute(0, 2, 1)
        elif self.encode_method == 'subnets':
            query_feat_vx = torch.cat([self.box_subnet(pred_boxes_vx_encoded.permute(0, 2, 1)), self.score_subnet(pred_scores_vx_encoded.permute(0, 2, 1)),
                                       self.label_subnet(pred_labels_vx_encoded.permute(0, 2, 1)), self.feat_subnet_vx(pred_feats_vx.permute(0, 2, 1))], dim=1)
            query_feat_ac = torch.cat([self.box_subnet(pred_boxes_ac_encoded.permute(0, 2, 1)), self.score_subnet(pred_scores_ac_encoded.permute(0, 2, 1)),
                                       self.label_subnet(pred_labels_ac_encoded.permute(0, 2, 1)), self.feat_subnet_ac(pred_feats_ac.permute(0, 2, 1))], dim=1)
        else:
            raise NotImplementedError
        #query_feat_vx = query_feat_vx.detach()
        #query_feat_ac = query_feat_ac.detach()
        # TODO: 怎么让网络感知到他们来自不同的head?
        if self.head_embedding_method == 'learnable_add':
            query_feat_vx += self.head_embedding(torch.tensor(0).long().cuda())[None, :, None]
            query_feat_ac += self.head_embedding(torch.tensor(1).long().cuda())[None, :, None]
        elif self.head_embedding_method == 'onehot_cat' or self.head_embedding_method == 'onehot_subnet':
            onehot_vx = query_feat_vx.new_zeros((query_feat_vx.shape[0], 2, query_feat_vx.shape[-1]))
            onehot_vx[:, 0, :] = 1.0
            onehot_ac = query_feat_ac.new_zeros((query_feat_ac.shape[0], 2, query_feat_ac.shape[-1]))
            onehot_ac[:, 1, :] = 1.0
            if self.head_embedding_method == 'onehot_cat':
                query_feat_vx = torch.cat([query_feat_vx, onehot_vx], dim=1)
                query_feat_ac = torch.cat([query_feat_ac, onehot_ac], dim=1)
            else:
                query_feat_vx = torch.cat([query_feat_vx, self.head_subnet(onehot_vx)], dim=1)
                query_feat_ac = torch.cat([query_feat_ac, self.head_subnet(onehot_ac)], dim=1)
        else:
            raise NotImplementedError    
        query_feat = torch.cat([query_feat_vx, query_feat_ac], dim=-1)

        
        # (B, C, N)
        query_feat = self.decoder(
            query_feat, self_attn_mask=attn_mask
        )

        pred_dict_combine = {
            'pred_boxes': torch.cat([pred_boxes_vx, pred_boxes_ac], dim=1),
            'pred_scores': torch.cat([pred_scores_vx, pred_scores_ac], dim=1),
            'pred_labels': torch.cat([pred_labels_vx, pred_labels_ac], dim=1),
            'pred_cls': torch.cat([pred_cls_vx, pred_cls_ac], dim=1)
        }
        batch_dict['pred_dict_combine'] = pred_dict_combine
        
        # self.query_labels = top_proposals_class

        # # add category embedding
        # one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        # query_cat_encoding = self.class_encoding(one_hot.float())
        # query_feat += query_cat_encoding

        # query_pos = bev_pos.gather(
        #     index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
        #     dim=1,
        # )
        # # convert to xy
        # query_pos = query_pos.flip(dims=[-1])
        # bev_pos = bev_pos.flip(dims=[-1])

        # query_feat = self.decoder(
        #     query_feat, lidar_feat_flatten, query_pos, bev_pos
        # )
        res_layer = self.prediction_head(query_feat)

        #self.cls_mask = torch.cat([(pred_labels_vx==0), (pred_labels_ac==0)], dim=-1)
        self.cls_mask = torch.ones((pred_labels_vx.shape[0], pred_labels_vx.shape[1]+pred_labels_ac.shape[1])).bool().cuda()
        res_layer['cls'] = res_layer['cls'] * self.cls_mask.unsqueeze(1)

        res_layer['rcnn_reg'] = res_layer['reg'].permute(0, 2, 1).contiguous()
        res_layer['rcnn_cls'] = res_layer['cls'].permute(0, 2, 1).contiguous()# + torch.cat([pred_scores_logic_vx, pred_scores_logic_ac], dim=1).unsqueeze(-1)
        
        # res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

        # res_layer["query_heatmap_score"] = heatmap.gather(
        #     index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
        #     dim=-1,
        # )
        # res_layer["dense_heatmap"] = dense_heatmap

        replace_gt = False
        if replace_gt:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            target_dict = self.get_targets(gt_bboxes_3d, gt_labels_3d, batch_dict['pred_dict_combine'])
            
            code_size = self.box_coder.code_size
            gt_boxes3d_ct = target_dict['gt_of_rois'][..., 0:code_size]
            
            roi_boxes3d = target_dict['rois']
            reg_valid_mask = target_dict['reg_valid_mask']
            rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )
            reg_targets = reg_targets[reg_valid_mask.view(-1)]

            res_layer['rcnn_reg'] = res_layer['rcnn_reg'].view(-1, 7)
            #res['rcnn_reg'][reg_valid_mask.view(-1)] = reg_targets + 0.001*torch.randn(reg_targets.shape, dtype=reg_targets.dtype, device=reg_targets.device)

            #res['rcnn_cls'] = inverse_sigmoid(target_dict['rcnn_cls_labels']).unsqueeze(-1)
            res_layer['rcnn_cls'][self.cls_mask] = (inverse_sigmoid(target_dict['rcnn_cls_labels']).unsqueeze(-1))[self.cls_mask]

            rank_loss_mask = target_dict['rank_loss_mask']
            if rank_loss_mask is not None:
                rank_loss_target = target_dict['rank_targets']
                #res['rcnn_cls'][rank_loss_mask] = (inverse_sigmoid(rank_loss_target[rank_loss_mask])-5.0).unsqueeze(-1)
                res_layer['rcnn_cls'][rank_loss_mask&self.cls_mask] = (inverse_sigmoid(rank_loss_target[rank_loss_mask&self.cls_mask])-5.0).unsqueeze(-1)


        return res_layer

    def forward(self, batch_dict):
        res = self.predict(batch_dict)
        if not self.training:
            rois = batch_dict['pred_dict_combine']['pred_boxes']
            # batch_dict['batch_cls_preds'] = inverse_sigmoid(batch_dict['pred_dict_combine']['pred_scores']).unsqueeze(-1)
            # batch_dict['batch_box_preds'] = rois


            # sanity check whether targets are correct
            replace_gt = False
            if replace_gt:
                gt_boxes = batch_dict['gt_boxes']
                gt_bboxes_3d = gt_boxes[...,:-1]
                gt_labels_3d =  gt_boxes[...,-1].long() - 1
                target_dict = self.get_targets(gt_bboxes_3d, gt_labels_3d, batch_dict['pred_dict_combine'])
                
                code_size = self.box_coder.code_size
                gt_boxes3d_ct = target_dict['gt_of_rois'][..., 0:code_size]
                
                roi_boxes3d = target_dict['rois']
                reg_valid_mask = target_dict['reg_valid_mask']
                rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                reg_targets = self.box_coder.encode_torch(
                    gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
                )
                reg_targets = reg_targets[reg_valid_mask.view(-1)]

                res['rcnn_reg'] = res['rcnn_reg'].view(-1, 7)
                #res['rcnn_reg'][reg_valid_mask.view(-1)] = reg_targets + 0.001*torch.randn(reg_targets.shape, dtype=reg_targets.dtype, device=reg_targets.device)

                #res['rcnn_cls'] = inverse_sigmoid(target_dict['rcnn_cls_labels']).unsqueeze(-1)
                res['rcnn_cls'][self.cls_mask] = (inverse_sigmoid(target_dict['rcnn_cls_labels']).unsqueeze(-1))[self.cls_mask]

                rank_loss_mask = target_dict['rank_loss_mask']
                if rank_loss_mask is not None:
                    rank_loss_target = target_dict['rank_targets']
                    #res['rcnn_cls'][rank_loss_mask] = (inverse_sigmoid(rank_loss_target[rank_loss_mask])-5.0).unsqueeze(-1)
                    res['rcnn_cls'][rank_loss_mask&self.cls_mask] = (inverse_sigmoid(rank_loss_target[rank_loss_mask&self.cls_mask])-5.0).unsqueeze(-1)

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=rois, cls_preds=res['rcnn_cls'], box_preds=res['rcnn_reg']
            )

            batch_dict['has_class_labels'] = True
            batch_dict['roi_labels'] = batch_dict['pred_dict_combine']['pred_labels']+1
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            #bboxes = self.get_bboxes(res)
            #batch_dict['final_box_dicts'] = bboxes

            # for debug 看看validation loss是否下降
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            target_dict = self.get_targets(gt_bboxes_3d, gt_labels_3d, batch_dict['pred_dict_combine'])
            self.forward_ret_dict = {**target_dict, **res}
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            target_dict = self.get_targets(gt_bboxes_3d, gt_labels_3d, batch_dict['pred_dict_combine'])
            self.forward_ret_dict = {**target_dict, **res}
            #loss, tb_dict = self.get_loss(forward_ret_dict)
            #loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res, batch_dict['pred_dict_combine'])
            #batch_dict['loss'] = loss
            #batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        targets_dict = {
            'rois': torch.stack(res_tuple[0]),
            'gt_of_rois': torch.stack(res_tuple[1]),
            'gt_iou_of_rois': torch.stack(res_tuple[2]),
            'roi_scores': torch.stack(res_tuple[3]),
            'roi_labels': torch.stack(res_tuple[4]),
            'reg_valid_mask': torch.stack(res_tuple[5]),
            'rcnn_cls_labels': torch.stack(res_tuple[6]),
            'rank_loss_mask': None if res_tuple[7][0] is None else torch.stack(res_tuple[7]),
            'rank_targets': None if res_tuple[8][0] is None else torch.stack(res_tuple[8])
        }
        # labels = torch.cat(res_tuple[0], dim=0)
        # label_weights = torch.cat(res_tuple[1], dim=0)
        # bbox_targets = torch.cat(res_tuple[2], dim=0)
        # bbox_weights = torch.cat(res_tuple[3], dim=0)
        # num_pos = np.sum(res_tuple[4])
        # matched_ious = np.mean(res_tuple[5])
        # heatmap = torch.cat(res_tuple[6], dim=0)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        batch_size = rois.shape[0]
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict
        return targets_dict#labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap

    def ious_to_scores(self, ious, gt_labels_3d=None):
        iou_bg_thresh = torch.tensor(self.model_cfg.CLS_BG_THRESH).to(ious.device)
        iou_bg_thresh = iou_bg_thresh[gt_labels_3d]
        iou_fg_thresh = torch.tensor(self.model_cfg.CLS_FG_THRESH).to(ious.device)
        iou_fg_thresh = iou_fg_thresh[gt_labels_3d]

        fg_mask = ious > iou_fg_thresh
        bg_mask = ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        cls_labels = (fg_mask > 0).float()
        cls_labels[interval_mask] = (ious[interval_mask] - iou_bg_thresh[interval_mask]) / \
                                (iou_fg_thresh[interval_mask] - iou_bg_thresh[interval_mask])
        return cls_labels

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        num_proposals = preds_dict['pred_boxes'].shape[1]
        # (not used)这个score形状有点奇怪，我们和transfusion head保持一致，下面的assign会处理好这个形状
        bboxes_tensor = preds_dict['pred_boxes'][0].detach()
        labels_tensor = preds_dict['pred_labels'][0].detach()
        gt_bboxes_tensor = gt_bboxes_3d.to(bboxes_tensor.device)

        if gt_bboxes_tensor.shape[0] == 0:
            target_dict = {
                'rois': bboxes_tensor,
                'gt_of_rois': torch.zeros((bboxes_tensor.shape[0], self.code_size), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device),
                'gt_iou_of_rois': torch.zeros((bboxes_tensor.shape[0],), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device),
                'roi_scores': preds_dict['pred_scores'][0],
                'roi_labels': preds_dict['pred_labels'][0],
                'reg_valid_mask': torch.zeros((bboxes_tensor.shape[0],), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device).bool(),
                'score_targets': torch.zeros((bboxes_tensor.shape[0],), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device),
                'rank_loss_mask': None if not self.use_rank_loss else torch.zeros((bboxes_tensor.shape[0],), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device).bool(),
                'rank_targets': None if not self.use_rank_loss else torch.zeros((bboxes_tensor.shape[0],), dtype=bboxes_tensor.dtype, device=bboxes_tensor.device)
                #'rcnn_cls_labels': 
            }
            return tuple(target_dict.values())

        ious = overlaps(bboxes_tensor, gt_bboxes_tensor)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        max_ious, matched_gt_inds = ious.max(-1)

        reg_fg_th = (torch.tensor(self.model_cfg.REG_FG_THRESH).to(ious.device))[gt_labels_3d[matched_gt_inds]]
        reg_valid_mask = (max_ious > reg_fg_th)
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(bboxes_tensor.device)
        bbox_targets[reg_valid_mask] = gt_bboxes_tensor[matched_gt_inds[reg_valid_mask]]
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(bboxes_tensor.device)
        bbox_weights[reg_valid_mask] = 1.0

        # 思路一
        if not self.use_rank_loss:
            score_targets = self.ious_to_scores(max_ious, gt_labels_3d[matched_gt_inds])
            score_targets[gt_labels_3d[matched_gt_inds]!=labels_tensor] = 0.0
            rank_loss_mask = None
            rank_targets = None
        # 思路二
        else:
            score_targets = self.ious_to_scores(max_ious, gt_labels_3d[matched_gt_inds])
            rank_loss_mask = torch.zeros_like(score_targets, dtype=bool)
            rank_targets = torch.zeros_like(score_targets)
            max_ious_gt, matched_box_inds = ious.max(0)
            corr_gt_max_ious = max_ious_gt[matched_gt_inds]
            candidate_mask = max_ious > 0.01
            peer_ious = overlaps(bboxes_tensor[candidate_mask], bboxes_tensor[matched_box_inds])
            peer_ious = peer_ious[torch.arange(0, peer_ious.shape[0]).long(), matched_gt_inds[candidate_mask]]
            rank_loss_mask[torch.where(candidate_mask)[0][peer_ious >= 0.01]] = True
            rank_loss_mask[matched_box_inds] = False
            mismatched_label_mask = (gt_labels_3d[matched_gt_inds]!=labels_tensor)
            rank_loss_mask[mismatched_label_mask] = False
            rank_targets[rank_loss_mask] = score_targets[matched_box_inds[matched_gt_inds[rank_loss_mask]]]
            score_targets[mismatched_label_mask] = 0.0

        # assigned_gt_inds, ious = self.bbox_assigner.assign(
        #     bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
        #     score, self.point_cloud_range,
        # )
        # pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        # neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        # pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        # if gt_bboxes_3d.numel() == 0:
        #     assert pos_inds.numel() == 0
        #     pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        # else:
        #     pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # # create target for loss computation
        # bbox_targets = torch.zeros([num_proposals, self.code_size]).to(bboxes_tensor.device)
        # bbox_weights = torch.zeros([num_proposals, self.code_size]).to(bboxes_tensor.device)
        # ious = torch.clamp(ious, min=0.0, max=1.0)
        # labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        # label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        # if gt_labels_3d is not None:  # default label is -1
        #     labels += self.num_classes

        # # both pos and neg have classification loss, only pos has regression and iou loss
        # if len(pos_inds) > 0:
        #     #pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
        #     pos_bbox_targets = pos_gt_bboxes
        #     bbox_targets[pos_inds, :] = pos_bbox_targets
        #     bbox_weights[pos_inds, :] = 1.0

        #     if gt_labels_3d is None:
        #         labels[pos_inds] = 1
        #     else:
        #         labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
        #     label_weights[pos_inds] = 1.0

        # if len(neg_inds) > 0:
        #     label_weights[neg_inds] = 1.0

        # mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)

        target_dict = {
            'rois': bboxes_tensor,
            'gt_of_rois': bbox_targets,
            'gt_iou_of_rois': max_ious,
            'roi_scores': preds_dict['pred_scores'][0],
            'roi_labels': preds_dict['pred_labels'][0],
            'reg_valid_mask': reg_valid_mask,
            'score_targets': score_targets,
            'rank_loss_mask': rank_loss_mask,
            'rank_targets': rank_targets
            #'rcnn_cls_labels': 
        }
        return tuple(target_dict.values())
        #return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou))


    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0) & (self.cls_mask.view(-1))
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.box_coder.encode_torch(
            gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
        )

        rcnn_loss_reg = self.reg_loss_func(
            rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
            reg_targets.unsqueeze(dim=0),
        )  # [B, M, 7]
        rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
        rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

        if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
            gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
            # TODO: NEED to BE CHECK
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
            batch_anchors = fg_roi_boxes3d.clone().detach()
            roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
            roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
            batch_anchors[:, :, 0:3] = 0
            rcnn_boxes3d = self.box_coder.decode_torch(
                fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
            ).view(-1, code_size)

            rcnn_boxes3d = common_utils.rotate_points_along_z(
                rcnn_boxes3d.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)
            rcnn_boxes3d[:, 0:3] += roi_xyz

            loss_corner = loss_utils.get_corner_loss_lidar(
                rcnn_boxes3d[:, 0:7],
                gt_of_rois_src[fg_mask][:, 0:7]
            )
            loss_corner = loss_corner.mean()
            loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

            rcnn_loss_reg += loss_corner
            tb_dict['rcnn_loss_corner'] = loss_corner.item()

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0)
            rank_loss_mask = forward_ret_dict['rank_loss_mask']
            if rank_loss_mask is not None:
                cls_valid_mask = cls_valid_mask & (~rank_loss_mask.view(-1))
            cls_valid_mask = cls_valid_mask.float() * self.cls_mask.view(-1)
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0)
            rank_loss_mask = forward_ret_dict['rank_loss_mask'].view(-1)
            if rank_loss_mask is not None:
                cls_valid_mask = cls_valid_mask & (~rank_loss_mask)
            cls_valid_mask = cls_valid_mask.float() * self.cls_mask.view(-1)
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict
    
    def get_rank_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls'].view(-1)
        rank_targets = forward_ret_dict['rank_targets'].view(-1)
        rank_loss_mask = forward_ret_dict['rank_loss_mask'].view(-1)
        rank_loss_mask &= self.cls_mask.view(-1)
        rank_loss = torch.clamp_min(torch.sigmoid(rcnn_cls[rank_loss_mask])-rank_targets[rank_loss_mask]+0.02, 0)
        rank_loss = rank_loss.sum()/max(rank_loss_mask.sum(), 1) * loss_cfgs.LOSS_WEIGHTS['rank_loss_weight']
        return rank_loss, {'rcnn_rank_loss': rank_loss.item()}

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        
        if self.forward_ret_dict['rank_targets'] is not None:
            rcnn_rank_loss, rank_tb_dict = self.get_rank_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_rank_loss
            tb_dict.update(rank_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds

    def loss(self, gt_bboxes_3d, gt_labels_3d, res, pred_dicts, **kwargs):
        
        self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = self.code_size
        targets = torch.zeros([*bboxes.shape[:-1], code_size]).to(bboxes.device)
        targets[..., 0] = (bboxes[..., 0] - self.point_cloud_range[0]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
        targets[..., 1] = (bboxes[..., 1] - self.point_cloud_range[1]) / (self.point_cloud_range[4] - self.point_cloud_range[1])
        targets[..., 2] = (bboxes[..., 2] - self.point_cloud_range[2]) / (self.point_cloud_range[5] - self.point_cloud_range[2])
        targets[..., 3:6] = bboxes[..., 3:6].log()
        targets[..., 6] = bboxes[..., 6] / (2*torch.pi)
        return targets

    # def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     score_thresh = post_process_cfg.SCORE_THRESH
    #     post_center_range = post_process_cfg.POST_CENTER_RANGE
    #     post_center_range = torch.tensor(post_center_range).cuda().float()
    #     # class label
    #     final_preds = heatmap.max(1, keepdims=False).indices
    #     final_scores = heatmap.max(1, keepdims=False).values

    #     center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
    #     center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
    #     dim = dim.exp()
    #     rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
    #     rot = torch.atan2(rots, rotc)

    #     if vel is None:
    #         final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
    #     else:
    #         final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

    #     predictions_dicts = []
    #     for i in range(heatmap.shape[0]):
    #         boxes3d = final_box_preds[i]
    #         scores = final_scores[i]
    #         labels = final_preds[i]
    #         predictions_dict = {
    #             'pred_boxes': boxes3d,
    #             'pred_scores': scores,
    #             'pred_labels': labels
    #         }
    #         predictions_dicts.append(predictions_dict)

    #     if filter is False:
    #         return predictions_dicts

    #     thresh_mask = final_scores > score_thresh        
    #     mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
    #     mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

    #     predictions_dicts = []
    #     for i in range(heatmap.shape[0]):
    #         cmask = mask[i, :]
    #         cmask &= thresh_mask[i]

    #         boxes3d = final_box_preds[i, cmask]
    #         scores = final_scores[i, cmask]
    #         labels = final_preds[i, cmask]
    #         predictions_dict = {
    #             'pred_boxes': boxes3d,
    #             'pred_scores': scores,
    #             'pred_labels': labels,
    #         }

    #         predictions_dicts.append(predictions_dict)

    #     return predictions_dicts

    # def get_bboxes(self, preds_dicts):

    #     batch_size = preds_dicts["heatmap"].shape[0]
    #     batch_score = preds_dicts["heatmap"].sigmoid()
    #     one_hot = F.one_hot(
    #         self.query_labels, num_classes=self.num_classes
    #     ).permute(0, 2, 1)
    #     batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
    #     batch_center = preds_dicts["center"]
    #     batch_height = preds_dicts["height"]
    #     batch_dim = preds_dicts["dim"]
    #     batch_rot = preds_dicts["rot"]
    #     batch_vel = None
    #     if "vel" in preds_dicts:
    #         batch_vel = preds_dicts["vel"]

    #     ret_dict = self.decode_bbox(
    #         batch_score, batch_rot, batch_dim,
    #         batch_center, batch_height, batch_vel,
    #         filter=True,
    #     )
    #     for k in range(batch_size):
    #         ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

    #     return ret_dict 
