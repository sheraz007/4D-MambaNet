import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ...utils.spconv_utils import replace_feature, spconv

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.use_sp_input = model_cfg.get('use_sp_input', True)
        self.predict_boxes_when_training = self.predict_boxes_when_training or self.model_cfg.get('PREDICT_BOXES_FOR_DOUBLEHEAD_REFINE', False)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_cls_sp = spconv.SubMConv2d(input_channels, self.num_anchors_per_location * self.num_class, 1, bias=True)
        self.conv_box_sp = spconv.SubMConv2d(input_channels, self.num_anchors_per_location * self.box_coder.code_size, 1, bias=True)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )

            self.conv_dir_cls_sp = spconv.SubMConv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
                bias=True
            )

        else:
            self.conv_dir_cls = None
            self.conv_dir_cls_sp = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv_cls_sp.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box_sp.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        if self.use_sp_input:
            spatial_features_2d = data_dict['encoded_spconv_tensor'] # 8 128 128 128
            cls_preds_sp = self.conv_cls_sp(spatial_features_2d) # 18
            box_preds_sp = self.conv_box_sp(spatial_features_2d) # 42
            cls_preds = cls_preds_sp.dense()
            box_preds = box_preds_sp.dense()
        else:
            spatial_features_2d = data_dict['spatial_features_2d'] # 8 128 128 128
            cls_preds = self.conv_cls(spatial_features_2d)
            box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] 8,128,128,18
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] 42

        self.forward_ret_dict['cls_preds'] = cls_preds # 2
        self.forward_ret_dict['box_preds'] = box_preds # 14

        if self.conv_dir_cls is not None:
            if self.use_sp_input:
                dir_cls_preds = self.conv_dir_cls_sp(spatial_features_2d)
                dir_cls_preds = dir_cls_preds.dense()
            else:
                dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds # 12
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )

            if 'bev_non_empty_mask' in data_dict:
                batch_size = data_dict['batch_size']
                m = data_dict['bev_non_empty_mask'].view((batch_size, -1)).repeat_interleave(self.num_class*2, dim=1)
                targets_dict['box_cls_labels'][~m] = -1 # torch.Size([8, 98304])
            self.forward_ret_dict.update(targets_dict)

            vis = False
            if vis:
                import matplotlib.pyplot as plt
                from pcdet.utils.visualize_utils import draw_bev_bboxes
                import numpy as np
                point_cloud_range = torch.tensor([0, -25.6, -3], dtype=torch.float32).cuda()
                voxel_size = torch.tensor([0.05, 0.05, 0.125], dtype=torch.float32).cuda()
                batch_idx = 0
                batch_size = data_dict['batch_size']
                for batch_idx in range(batch_size):
                    dpi = 600
                    fig = plt.figure(dpi=dpi)
                    ax = fig.add_axes([0, 0, 1, 1])
                    batch_mask = spatial_features_2d.indices[:, 0]==batch_idx
                    points = spatial_features_2d.indices[batch_mask][:, [2, 1]]
                    ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), s=0.5, c='k')

                    m = data_dict['bev_non_empty_mask'].view((batch_size, -1)).repeat_interleave(self.num_class*2, dim=1)
                    box_cls_labels = targets_dict['box_cls_labels'][batch_idx][m[batch_idx]].clone()
                    # 一个位置两个anchor
                    box_cls_labels1 = box_cls_labels[::2]
                    box_cls_labels2 = box_cls_labels[1::2]
                    # box_cls_labels = box_cls_labels1+box_cls_labels2
                    box_cls_labels = box_cls_labels1
                    pos_mask = box_cls_labels == 1
                    points_pos = points[pos_mask]
                    ax.scatter(points_pos[:, 0].cpu().numpy(), points_pos[:, 1].cpu().numpy(), s=0.5, c='r')

                    gt_boxes_ = data_dict['gt_boxes'][batch_idx].clone()
                    gt_boxes_ = gt_boxes_[gt_boxes_.any(-1)]
                    gt_boxes_[:, :3] = (gt_boxes_[:, :3] - point_cloud_range[:3]) / (voxel_size * 8)
                    gt_boxes_[:, 3:6] = gt_boxes_[:, 3:6] / (voxel_size * 8)
                    gt_boxes_np = gt_boxes_.cpu().numpy()
                    # cur_box_draw
                    for box_id in range(gt_boxes_.shape[0]):
                        lidar_boxes = gt_boxes_np[box_id, :][np.newaxis, :-1]
                        cur_class_id = int(gt_boxes_np[box_id, :][-1])
                        if cur_class_id == 1:
                            box_colors = ['green' for n in range(lidar_boxes.shape[0])]
                            colls, polys = draw_bev_bboxes(lidar_boxes, edge_colors=box_colors, axis_align=False)
                        else:
                            box_colors_ = ['red' for n in range(lidar_boxes.shape[0])]
                            colls, polys = draw_bev_bboxes(lidar_boxes, edge_colors=box_colors_, axis_align=False)                            
                        ax.add_collection(colls)
                    
                    ax.set_xlim((0, 128))
                    ax.set_ylim((0, 128))
                    ax.grid(False)
                    plt.axis('equal')
                    fig.savefig(f'./z_vis/' + f'vx_{batch_idx}' + '.png')
                    plt.close()

        if not self.training or self.predict_boxes_when_training:
            if self.model_cfg.get('PREDICT_BOXES_FOR_DOUBLEHEAD_REFINE', False):
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=data_dict['batch_size'],
                    cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
                )
                batch_size = batch_cls_preds.shape[0]
                batch_score_logic_preds = batch_cls_preds.max(-1)[0]
                batch_cls_preds = batch_cls_preds.sigmoid()
                batch_score_preds, batch_label_preds = batch_cls_preds.max(-1)
                num_rois = self.model_cfg.NUM_ROIS
                batch_score_preds, topk_indices = torch.topk(batch_score_preds, num_rois, dim=-1)
                batch_label_preds = torch.gather(batch_label_preds, 1, topk_indices)
                batch_score_logic_preds = torch.gather(batch_score_logic_preds, 1, topk_indices)
                topk_indices_box = topk_indices.unsqueeze(-1).expand(-1, -1, batch_box_preds.size(-1))
                batch_box_preds = torch.gather(batch_box_preds, 1, topk_indices_box)
                topk_indices_featmap = topk_indices // (len(self.anchors) * self.anchors[0].shape[-2])
                feat_flatten = spatial_features_2d.view(*spatial_features_2d.shape[:2], -1).permute(0, 2, 1)
                topk_indices_featmap = topk_indices_featmap.unsqueeze(-1).expand(-1, -1, feat_flatten.size(-1))
                batch_box_feats = torch.gather(feat_flatten, 1, topk_indices_featmap)
                topk_indices_cls = topk_indices.unsqueeze(-1).expand(-1, -1, batch_cls_preds.size(-1))
                batch_cls_preds = torch.gather(batch_cls_preds, 1, topk_indices_cls)
                data_dict['pred_scores_ac'] = batch_score_preds
                data_dict['pred_boxes_ac'] = batch_box_preds
                data_dict['pred_feats_ac'] = batch_box_feats
                data_dict['pred_labels_ac'] = batch_label_preds
                data_dict['pred_cls_ac'] = batch_cls_preds
                data_dict['pred_scores_logic_ac'] = batch_score_logic_preds
            else:
                if 'bev_non_empty_mask' in data_dict:
                    m = data_dict['bev_non_empty_mask']
                    cls_preds[~m] = -10000
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=data_dict['batch_size'],
                    cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
                )
                data_dict['batch_cls_preds'] = batch_cls_preds
                data_dict['batch_box_preds'] = batch_box_preds
                data_dict['cls_preds_normalized'] = False

        return data_dict
