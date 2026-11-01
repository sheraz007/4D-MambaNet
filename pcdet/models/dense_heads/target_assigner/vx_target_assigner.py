import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils
from .axis_aligned_target_assigner import AxisAlignedTargetAssigner
from ...model_utils import centernet_utils

class vxTargetAssigner(AxisAlignedTargetAssigner):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__(model_cfg, class_names, box_coder, match_height=False)

        self.point_cloud_range = torch.Tensor(model_cfg.TARGET_ASSIGNER_CONFIG.get('point_cloud_range', None)).cuda()
        self.use_gaussian = model_cfg.get('use_gaussian', False)

        self.voxel_size = torch.Tensor(model_cfg.TARGET_ASSIGNER_CONFIG.get('voxel_size', None)).cuda()
        self.feature_map_stride = model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.gaussian_ratio = 1
        
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances
    
    def draw_gaussian_to_heatmap_voxels(heatmap, distances, radius, k=1):
        diameter = 2 * radius + 1
        sigma = diameter / 6
        masked_gaussian = torch.exp(- distances / (2 * sigma * sigma))

        torch.max(heatmap, masked_gaussian, out=heatmap)

        return heatmap
    
    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        spatial_shape = [128, 128]
        gaussian_overlap = 0.1
        min_radius = 2
        num_max_objs = 500
        num_classes = 1 # 上一步已经取出了其中一个类别
        num_anchors = anchors.shape[0] # 上一步已经取出了当前anchor
        anchor_num = 2
        self.max_sp_num_per_gt = anchor_num
        num_voxels = num_anchors

        point_xy = anchors[:, :2]
        point_coord_x = (point_xy[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        point_coord_y = (point_xy[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
        spatial_indices = torch.cat((point_coord_x[:, None], point_coord_y[:, None]), dim=-1)

        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)
        # ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        # ret_mul_boxes = gt_boxes.new_zeros((num_max_objs//self.max_sp_num_per_gt, self.max_sp_num_per_gt, gt_boxes.shape[-1]))

        inds = gt_boxes.new_zeros(num_max_objs).long()
        inds_mul = gt_boxes.new_zeros(num_max_objs//self.max_sp_num_per_gt, self.max_sp_num_per_gt).long()

        mask = gt_boxes.new_zeros(num_max_objs).long()
        mask_mul = gt_boxes.new_zeros(num_max_objs//self.max_sp_num_per_gt, self.max_sp_num_per_gt).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / self.feature_map_stride
        dy = dy / self.voxel_size[1] / self.feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            distance_min = torch.min(distance)
            inds_min = torch.where((distance == distance_min))[0].cuda()

            for inds_min_id in range(min(len(inds_min), anchor_num)):
                inds_mul[k][inds_min_id] = inds_min[inds_min_id]
                mask_mul[k][inds_min_id] = 1

            centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[0], distance, radius[k].item() * self.gaussian_ratio)

            centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[0], self.distance(spatial_indices, spatial_indices[inds_mul[k][0]]), radius[k].item() * self.gaussian_ratio)

        heatmap = heatmap.reshape(-1)

        labels = heatmap
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        for inds_min_id in range(min(len(inds_min), anchor_num)):
            inds = inds_mul[:, inds_min_id]
            fg_inds = inds[inds > 0]
            assert fg_inds.shape[0] == gt_boxes.shape[0]
            if len(gt_boxes) > 0 and anchors.shape[0] > 0:
                fg_gt_boxes = gt_boxes[:, :7]
                fg_anchors = anchors[fg_inds, :]
                bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        reg_weights[labels == 1] = 1.0

        if self.use_gaussian == False:
            mask_labels = labels == 1
            labels[~mask_labels] = 0
        
        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
