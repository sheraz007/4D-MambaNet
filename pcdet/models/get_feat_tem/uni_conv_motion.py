import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv
from .uni_conv import UniConv
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import copy

class UniConv_motion(UniConv):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        off_laynum = model_cfg.get('off_laynum', 2)
        self.use_vis =  model_cfg.get('use_vis', False)
        self.use_off_gt =  model_cfg.get('use_off_gt', False)
        input_channels = 2 * self.input_channels
        output_channels = self.input_channels
        self.off_conv = nn.ModuleList()
        for i in range(off_laynum):
            if i > 0:
                input_channels = output_channels
            off_conv1 = [
                nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            self.off_conv.extend(off_conv1)
        off_conv2 = [
                nn.Conv2d(input_channels, 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(2, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
        self.off_conv.extend(off_conv2)

    def trans_from_two_boxes(self, box1, box2):
        '''
        Args:
            box1: np.ndarray [7,]
            box2: np.ndarray [7,]
        Ret:
            t_mat: from box1 to box2
        '''
        trans = box2[:3] - box1[:3]
        ang = box2[6] - box1[6]
        rot = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1]
        ])
        t_mat = np.concatenate([rot, trans[:, np.newaxis]], axis=-1)
        t_mat = np.concatenate([t_mat, np.zeros((1, 4))], axis=0)
        t_mat[-1, -1] = 1.0
        return t_mat
    
    def box3d_to_bev(self, gt_boxes): # n, 7
        center_point = gt_boxes[:,:3]
        grid_center = self.point2grid(center_point)
        dim_point = gt_boxes[:,3:6]
        dim_bev = dim_point / (self.voxel_size * self.downsample_times)
        #dim_bev[:,-1] = 100 # bev height max
        gt_boxes[:,:3] = grid_center
        gt_boxes[:,3:6] = dim_bev
        return gt_boxes

    def motion_compensation_bev_onebatch(self, prev_grid, before_gt_boxes, cur_gt_box):
        before_gt_boxes = self.box3d_to_bev(before_gt_boxes)
        cur_gt_box = self.box3d_to_bev(cur_gt_box)
        before_gt_boxes = before_gt_boxes.cpu().numpy()
        cur_gt_box = cur_gt_box.cpu().numpy()
        if isinstance(prev_grid, np.ndarray) == False: 
            prev_grid = prev_grid.cpu().numpy()
        offset_all = np.zeros((prev_grid.shape[0], 2))
        for box_id, box in enumerate(before_gt_boxes):
            target_box = None
            for target_box_id, b in enumerate(cur_gt_box):
                if int(b[-2]) == int(box[-2]):
                    target_box = b # before frame box 对应的 cur box
                    break
            if target_box is None:
                continue
            prev_grid[:, -1] = box[2]
            cur_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                prev_grid, box[np.newaxis, :7])  # (nboxes, npoints)
            inbox_mask = (cur_point_indices.reshape(-1) == 1)
            ori_before_point_in_box = prev_grid[inbox_mask]
            before_point_in_box = copy.deepcopy(ori_before_point_in_box)
            t_box_target = self.trans_from_two_boxes(box, target_box)
            before_point_in_box[:, :3] -= box[None, :3]
            motion_com_point_in_box = (np.concatenate((before_point_in_box[:,0:3], np.ones((before_point_in_box.shape[0], 1))), axis=1) @ t_box_target.T)[:, :3] + box[None, :3]
            offset = ori_before_point_in_box[:,:2] - motion_com_point_in_box[:,:2]
            offset_all[inbox_mask] = offset    
        return torch.tensor(offset_all).cuda()
    
    def get_before_feat(self, radar2_radar1, before_feat, dense_cur_feat, cur_batch_gt, time_idx):
        w, h = before_feat.shape[-1], before_feat.shape[-2]
        grid_cur_frame = self.gen_cur_grid(w, h)
        prev_grid = self.gen_before_grid_from_cur(grid_cur_frame, radar2_radar1) # bs,x,y,z

        # get offset
        use_feat = torch.cat((before_feat, dense_cur_feat), dim=0)
        use_feat = use_feat.unsqueeze(0)
        for i in range(len(self.off_conv)):
            use_feat = self.off_conv[i](use_feat) # torch.Size([1, 2, 128, 128]) bs,c,h,w
        offset_we_pred = use_feat.permute(0, 2, 3, 1)
        prev_grid = prev_grid.view(1, h, w, 2)
        prev_sta_grid = prev_grid.clone().reshape(-1, 2) # xy
        prev_sta_grid = torch.cat((prev_sta_grid, torch.ones((prev_sta_grid.shape[0], 1)).cuda()) , dim = -1)

        if cur_batch_gt.shape[0] > 0:
            offset_target = self.motion_compensation_bev_onebatch(prev_sta_grid, cur_batch_gt[cur_batch_gt[:, -1] == (time_idx + 1)], cur_batch_gt[cur_batch_gt[:, -1] == 0])
            offset_target = offset_target.view(1, h, w, 2) # xy
        else:
            offset_target = torch.zeros((1, h, w, 2)).cuda() # xy
        
        if self.use_off_gt:
            offset = offset_target
        else:
            offset = offset_we_pred

        prev_grid_before_offset = prev_grid.clone()
        prev_grid = prev_grid + offset
        prev_grid[..., 0] = prev_grid[..., 0] * 2 / (w - 1) - 1 # only one batch data input
        prev_grid[..., 1] = prev_grid[..., 1] * 2 / (h - 1) - 1 # only one batch data input
        output = F.grid_sample(before_feat.unsqueeze(0), prev_grid.to(before_feat.dtype), align_corners=True, padding_mode='zeros')
        return output, offset_target, offset_we_pred, prev_grid_before_offset, grid_cur_frame[:,[1,2]]

    def get_motion_loss(self):
        loss_list = []
        h, w = self.offset_list_prev_list[0][0].shape[1], self.offset_list_prev_list[0][0].shape[2]
        for i in range(len(self.offset_list_prev_list)):
            for j in range(len(self.offset_list_prev_list[i])):
                loss_ = torch.abs(self.offset_list_prev_list[i][j] - self.offset_target_list_prev_list[i][j])
                loss_list.append(torch.sum(loss_) / (h * w))
        return sum(loss_list)
    
    def forward(self, data_dict):
        prev_feats = data_dict['before_feat_list']
        sparse_tensor = data_dict['encoded_spconv_tensor']
        
        indices_prev = []
        feat_prev_list = []
        offset_list_prev_list = []
        offset_target_list_prev_list = []
        indices_cur = sparse_tensor.indices
        indices_cur = torch.cat([indices_cur[:, 0:1], torch.zeros((indices_cur.shape[0], 1), dtype=indices_cur.dtype, device=indices_cur.device), indices_cur[:, 1:3]], dim=1)
        
        indices_prev_list = []
        for k, prev_feat in enumerate(prev_feats):
            feat_list = []
            indice_list = []
            offset_list = []
            offset_target_list = []
            before_feat = prev_feat.dense()
            dense_cur_feat = sparse_tensor.dense()
            for bs in range(data_dict['batch_size']):
                frame_id = data_dict['frame_id'][bs]
                cur_batch_gt = data_dict['gt_boxes_list'][bs]
                cur_box_class = data_dict['gt_name_list'][bs]
                fi = 1
                for fi in range(cur_batch_gt.shape[0]):
                    if torch.sum(cur_batch_gt[fi,:]) == 0:
                        break
                cur_batch_gt = cur_batch_gt[:fi, :]
                cur_box_class = cur_box_class[:fi, :]       
                feat_prev, offset_target, offset_we_pred, prev_grid_before_offset, grid_cur_frame = self.get_before_feat(data_dict['radar2_radar1_list'][bs][k], before_feat[bs], dense_cur_feat[bs], cur_batch_gt, k)

                # self.use_vis = True
                if self.use_vis:
                    import matplotlib.pyplot as plt
                    from pcdet.utils.visualize_utils import draw_bev_bboxes
                    import numpy as np  
                    dpi = 600
                    fig = plt.figure(dpi=dpi)
                    ax = fig.add_axes([0, 0, 1, 1])

                    points = data_dict['encoded_spconv_tensor'].indices[:,[2,1]]
                    ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), s=0.1)

                    boxes = cur_batch_gt
                    box_class = cur_box_class
                    time_mask = boxes[:, -1].int()==0
                    boxes = boxes[time_mask][:, :7] # cur time
                    box_class = box_class[time_mask]
                    mask = torch.tensor([(b[3] >0.1 and b[4] >0.1) for b in boxes], dtype=bool).cuda()
                    boxes = boxes[mask]
                    box_class = box_class[mask]
                    boxes[:, :3] = (boxes[:, :3] - self.point_cloud_range[:3]) / (self.voxel_size * 8)
                    boxes[:, 3:6] = boxes[:, 3:6] / (self.voxel_size * 8)
                    boxes = boxes.cpu().numpy()
                    if boxes.size > 0:
                        # cur_box_draw
                        for box_id in range(boxes.shape[0]):
                            lidar_boxes = boxes[box_id, :7][np.newaxis, :]
                            if box_class[box_id] == 1:
                                box_colors = ['green' for n in range(lidar_boxes.shape[0])]
                                colls, polys = draw_bev_bboxes(lidar_boxes, edge_colors=box_colors, axis_align=False)
                            else:
                                box_colors_ = ['red' for n in range(lidar_boxes.shape[0])]
                                colls, polys = draw_bev_bboxes(lidar_boxes, edge_colors=box_colors_, axis_align=False)                            
                            ax.add_collection(colls)

                    vis_prev = False
                    if vis_prev:
                        prev_points = data_dict['prev_points'][k][data_dict['prev_points'][k][:, 0].int()==bs].clone()[:, 1:4]
                        prev_points = (prev_points - self.point_cloud_range[:3]) / (self.voxel_size * 8)
                        ax.scatter(prev_points[:, 0].cpu().numpy(), prev_points[:, 1].cpu().numpy(), s=0.1, c='r')
                        prev_boxes = data_dict['gt_boxes_list'][bs]
                        prev_boxes = prev_boxes[prev_boxes[:, -1].int()==(k+1)][:, :7]
                        mask = torch.tensor([(b[3]>0.1 and b[4]>0.1) for b in prev_boxes], dtype=bool).cuda()
                        prev_boxes = prev_boxes[mask]
                        prev_boxes[:, :3] = (prev_boxes[:, :3] - self.point_cloud_range[:3]) / (self.voxel_size * 8)
                        prev_boxes[:, 3:6] = prev_boxes[:, 3:6] / (self.voxel_size * 8)
                        prev_boxes = prev_boxes.cpu().numpy()
                        if prev_boxes.size > 0:
                            # prev box draw
                            lidar_boxes = prev_boxes[:, :7]
                            box_colors = ['red' for n in range(lidar_boxes.shape[0])]
                            colls, polys = draw_bev_bboxes(lidar_boxes, edge_colors=box_colors, axis_align=False)
                            ax.add_collection(colls)

                        prev_grid_before_offset = prev_grid_before_offset.view((-1, 2))
                        prev_grid_after_offset = prev_grid_before_offset.view((-1, 2)) + offset_target.view((-1, 2))
                        grid_cur_frame = grid_cur_frame.view((-1, 2))

                        moving_mask = offset_target.view((-1, 2)).sum(-1)!=0

                        grid_cur_frame = grid_cur_frame[moving_mask]
                        ax.scatter(grid_cur_frame[:, 0].cpu().numpy(), grid_cur_frame[:, 1].cpu().numpy(), s=0.1, c='cyan')
                        
                        prev_grid_after_offset = prev_grid_after_offset[moving_mask]
                        prev_grid_after_offset = prev_grid_after_offset.view((-1, 2))
                        ax.scatter(prev_grid_after_offset[:, 0].cpu().numpy(), prev_grid_after_offset[:, 1].cpu().numpy(), s=0.1, c='yellow')
                    
                    ax.set_xlim((0, 128))
                    ax.set_ylim((0, 128))
                    ax.grid(False)
                    fig.savefig(f'./z_vis/' + frame_id + '.png')

                feat_prev = feat_prev[0]
                non_empty_indice = feat_prev.any(0).nonzero()
                non_empty_feats = feat_prev[:, non_empty_indice[:, 0], non_empty_indice[:, 1]]
                feat_list.append(non_empty_feats.permute(1, 0))
                indice = torch.cat((bs * torch.ones((non_empty_indice.shape[0], 1),device=before_feat.device), non_empty_indice), dim=1)
                indice_list.append(indice)
                offset_list.append(offset_we_pred)
                offset_target_list.append(offset_target)

            feat_prev = torch.cat(feat_list, dim=0)
            indices_prev = torch.cat(indice_list, dim=0)
            feat_prev_list.append(feat_prev)
            indices_prev_list.append(torch.cat([indices_prev[:, 0:1], (k+1)+torch.zeros((indices_prev.shape[0], 1), dtype=indices_prev.dtype, device=indices_prev.device), indices_prev[:, 1:]], dim=1))
            offset_list_prev_list.append(offset_list)
            offset_target_list_prev_list.append(offset_target_list)

        indices_all = torch.cat([indices_cur, *indices_prev_list], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat([sparse_tensor.features, *feat_prev_list], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = spconv.SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[len(prev_feats)+1, *sparse_tensor.spatial_shape],
            batch_size=sparse_tensor.batch_size
        )
        st_tensor = self.st_conv(st_tensor)
        st_tensor = self.bev_out(st_tensor)

        data_dict['encoded_spconv_tensor'] = st_tensor
        self.offset_list_prev_list = offset_list_prev_list
        self.offset_target_list_prev_list = offset_target_list_prev_list
        
        return data_dict
