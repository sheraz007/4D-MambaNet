from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt, VoxelResBackBone8xVoxelNeXt_group, VoxelResBackBone8xVoxelNeXt_separate
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
from .spconv_unet import UNetV2
from .dsvt import DSVT
from .spconv_backbone_focal_voxelnext import VoxelBackBone8xFocalVoxelNeXt
from .spconv_backbone_voxelnext_ptv3 import VoxelResBackBone8xVoxelNeXtPTv3

# Optional backbones that may require extra compiled deps (e.g. pointnet2_ops)
try:
    from .spconv_backbone_voxelnext_attnbase import VoxelResBackBone8xVoxelNeXtattn
except Exception:
    VoxelResBackBone8xVoxelNeXtattn = None

try:
    from .spconv_backbone_voxelnext_mamba import VoxelResBackBone8xVoxelNeXtMamba
except Exception:
    VoxelResBackBone8xVoxelNeXtMamba = None

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt_group': VoxelResBackBone8xVoxelNeXt_group,
    'VoxelResBackBone8xVoxelNeXt_separate': VoxelResBackBone8xVoxelNeXt_separate,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'DSVT': DSVT,
    'VoxelBackBone8xFocalVoxelNeXt': VoxelBackBone8xFocalVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXtPTv3': VoxelResBackBone8xVoxelNeXtPTv3,
    'VoxelResBackBone8xVoxelNeXtMamba': VoxelResBackBone8xVoxelNeXtMamba,
    'VoxelResBackBone8xVoxelNeXtattn': VoxelResBackBone8xVoxelNeXtattn
}
