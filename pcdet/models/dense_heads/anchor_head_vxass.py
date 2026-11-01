import numpy as np
import torch
import torch.nn as nn

from .anchor_head_single import AnchorHeadSingle
from ...utils.spconv_utils import replace_feature, spconv
from ...utils import loss_utils

class AnchorHeadSingleVxAss(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, **kwargs)
        self.anc_loss_type = model_cfg.get('anc_loss_type', 'anc')
        self.build_losses_hm()

    def build_losses_hm(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    def get_cls_layer_loss(self):
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # torch.Size([bs, n_a])
        batch_size = int(box_cls_labels.shape[0])
        cared = box_cls_labels >= 0
        cls_preds = self.forward_ret_dict['cls_preds'] # bs w h num_anchor_per_loc
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        cls_targets = box_cls_labels[cared].unsqueeze(-1)
        cls_preds = cls_preds[cared]
        cls_preds = self.sigmoid(cls_preds)
        cls_loss = self.hm_loss_func(cls_preds, cls_targets)
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
    