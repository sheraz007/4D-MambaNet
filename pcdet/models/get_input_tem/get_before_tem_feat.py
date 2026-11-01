import torch
import torch.nn as nn

class get_before_tem_feat(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

    def get_point_time_list(self, points, data_dict):
        batch_size = data_dict['batch_size']
        tem_num = int(data_dict['temporal_num'][0])
        indices_all_frames = points

        all_time_sp = []
        for i in range(tem_num + 1):
            cur_time_mask =  torch.logical_and((batch_size * i) <= indices_all_frames[:, 0] , indices_all_frames[:, 0] < (batch_size * (i + 1)))
            cur_time_point = points[cur_time_mask]
            cur_time_point[:, 0] = cur_time_point[:, 0] - i * batch_size
            all_time_sp.append(cur_time_point)
        return all_time_sp
     
    def forward(self, batch_dict, module_list):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
        radar_point_list = self.get_point_time_list(points, batch_dict)

        before_feat_list = []
        for i in range(1, len(radar_point_list)):
            with torch.no_grad():
                batch_dict['points'] = radar_point_list[i]
                for i, cur_module in enumerate(module_list[1:3]):
                    batch_dict = cur_module(batch_dict)
                before_feat_list.append(batch_dict['encoded_spconv_tensor'])
        
        batch_dict['before_feat_list'] = before_feat_list
        batch_dict['points'] = radar_point_list[0]
        batch_dict['prev_points'] = radar_point_list[1:]
        return batch_dict