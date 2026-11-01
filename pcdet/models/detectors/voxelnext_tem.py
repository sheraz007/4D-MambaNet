import copy
import torch
from .detector3d_template import Detector3DTemplate
from ..dense_heads.anchor_head_single import AnchorHeadSingle
from ..model_utils import model_nms_utils
from ..backbones_3d import vfe
from .voxelnext_doublehead import VoxelNeXtDoubleHead


class VoxelNeXt_tem(VoxelNeXtDoubleHead):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
    
    def forward(self, batch_dict):
        for i, cur_module in enumerate(self.module_list):
            if i == 0:
                batch_dict = cur_module(batch_dict, self.module_list)
            else:
                batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            # for debug 看看validation loss收不收敛
            if self.model_cfg.get('TRACK_VAL_LOSS', False):
                loss, tb_dict, disp_dict = self.get_training_loss()
                recall_dicts['val_loss'] = {}
                recall_dicts['val_loss'].update(tb_dict)
            return pred_dicts, recall_dicts