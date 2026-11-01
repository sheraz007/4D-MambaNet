import torch
import torch.nn as nn

class PesudoHeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.detach = self.model_cfg.get('DETACH', False)

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
        spatial_features = encoded_spconv_tensor.dense() # bs channel bevh bevw
        if self.detach: # false
            spatial_features = spatial_features.detach()
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_2d'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        if self.model_cfg.get('OUTPUT_BEV_NON_EMPTY_MASK', False):
            bev_non_empty_indices = encoded_spconv_tensor.indices.long()
            bev_non_empty_mask = torch.zeros([spatial_features.shape[0], *spatial_features.shape[-2:]], dtype=bool, device=spatial_features.device)
            bev_non_empty_mask[bev_non_empty_indices[:, 0], bev_non_empty_indices[:, 1], bev_non_empty_indices[:, 2]] = True
            batch_dict['bev_non_empty_indices'] = bev_non_empty_indices
            batch_dict['bev_non_empty_mask'] = bev_non_empty_mask
        return batch_dict    

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

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
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
