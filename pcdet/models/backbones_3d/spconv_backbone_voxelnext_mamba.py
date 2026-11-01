import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import voxel_aggregation_utils
from ...utils.spconv_utils import replace_feature, spconv
from ..sparse_attention.point_transformer_v3 import VoxelAttnPTv3
from .spconv_backbone_voxelnext_ptv3 import VoxelResBackBone8xVoxelNeXtPTv3

class VoxelResBackBone8xVoxelNeXtMamba(VoxelResBackBone8xVoxelNeXtPTv3):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
    

