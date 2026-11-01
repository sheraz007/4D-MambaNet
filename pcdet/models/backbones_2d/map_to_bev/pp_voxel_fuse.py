import torch
import torch.nn as nn
import torch.nn.functional as F

class PPVoxelFuse(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.detach = self.model_cfg.get('DETACH', False)
        self.use_downsample = model_cfg.get('USE_DOWNSAMPLE', True)
        if self.use_downsample:
            self.downsample_method = model_cfg.DOWNSAMPLE_METHOD
            if self.downsample_method == 'conv':
                self.downconv = nn.Sequential(
                    nn.ZeroPad2d(1),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
        else:
            self.upsample_method = model_cfg.UPSAMPLE_METHOD
            if self.upsample_method == 'conv':
                self.upconv = nn.Sequential(
                    nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
                    nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        if self.detach:
            spatial_features = spatial_features.detach()

        spatial_features_pp = batch_dict['spatial_features_pp']
        if self.use_downsample:
            if self.downsample_method == 'interp':
                spatial_features_pp = F.interpolate(spatial_features_pp, spatial_features.shape[-2:], align_corners=True, mode='bilinear')
            elif self.downsample_method == 'conv':
                spatial_features_pp = self.downconv(spatial_features_pp)
            else:
                raise NotImplementedError
        else:
            if self.upsample_method == 'interp':
                spatial_features = F.interpolate(spatial_features, spatial_features_pp.shape[-2:], align_corners=True, mode='bilinear')
            elif self.upsample_method == 'conv':
                spatial_features = self.upconv(spatial_features)
            else:
                raise NotImplementedError
        # TODO: an annoying bug: RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 6 but got size 7 for tensor number 1 in the list.
        try:
            spatial_features = torch.cat([spatial_features, spatial_features_pp], dim=1)
        except:
            print(spatial_features.shape)
            print(spatial_features_pp.shape)
            print(batch_dict['frame_id'])
            raise RuntimeError(f"wtf!, {spatial_features.shape}, {spatial_features_pp.shape}, {batch_dict['frame_id']}")

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_2d'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict   