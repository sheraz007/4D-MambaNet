import torch.nn as nn
from .ptv3.point_transformer_v3 import PointTransformerV3
from .mamba.point_mamba import PointMamba

class VoxelAttn_Layer(nn.Module):
    def __init__(
            self,
            attn_type='ptv3',
            enc_depths=(2,),
            enc_channels=(32,),
            enc_num_head=(8,),
            enc_patch_size=(48,),
            enable_flash=False,
            indice_key='ptv3_1'
        ):
        super().__init__()
        if attn_type == 'ptv3':
            self.attn_layer = PointTransformerV3(
                enc_depths=enc_depths,
                enc_channels=enc_channels,
                enc_num_head=enc_num_head,
                enc_patch_size=enc_patch_size,
                enable_flash=enable_flash,
                indice_key=indice_key  
            )
        if attn_type == 'mamba':
            self.attn_layer = PointMamba()

    def forward(self, sp_tensor):
        sp_tensor = self.attn_layer(sp_tensor)
        return sp_tensor