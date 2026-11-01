import torch
from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        pretrain_path = model_cfg.get('PRETRAIN_PATH', None)
        if pretrain_path is not None:
            self.prefix = 'model_points'
            state_dict = self._load_checkpoint_with_prefix(
                self.prefix, pretrain_path, map_location="cpu")
            if 'final.weight' in state_dict:
                del state_dict['final.weight']
            self.module_list[1].load_state_dict(state_dict,strict=True)
            print(f"load pretrain checkpoint from {pretrain_path}..........")

    def _load_checkpoint_with_prefix(self, prefix, load_path, map_location="cpu"):
        checkpoint = torch.load(load_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        # yyt_C openpcdet
        elif 'model_state' in checkpoint: 
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
        if not prefix.endswith('.'):
            prefix += '.'
        prefix_len = len(prefix)

        state_dict = {
            k[prefix_len:]: v
            for k, v in state_dict.items() if k.startswith(prefix)
        }

        assert state_dict, f'{prefix} is not in the pretrained model'
        return state_dict  
    
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
