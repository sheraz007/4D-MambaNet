from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pcdet.utils.object3d_kitti import get_objects_from_label
from pcdet.utils import calibration_kitti
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def draw_bev_bboxes(bboxes,
                    edge_colors='r',
                    line_styles='-',
                    line_widths=1,
                    face_colors='none',
                    alpha=1):
    # scale the bev bboxes for better visualization
    bev_bboxes = bboxes[:, [0, 1, 3, 4, 6]]
    ctr, w, h, theta = np.split(bev_bboxes, [2, 3, 4], axis=-1)
    cos_value, sin_value = np.cos(theta), np.sin(theta)
    vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
    vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    poly = np.stack([pt1, pt2, pt3, pt4], axis=-2)
    # move the object along x-axis
    poly[:, :, [0, 1]] = poly[:, :, [1, 0]]
    poly[:, :, 0] = - poly[:, :, 0]
    poly = [p for p in poly]
    return draw_polygons(
        poly,
        alpha=alpha,
        edge_colors=edge_colors,
        line_styles=line_styles,
        line_widths=line_widths,
        face_colors=face_colors), poly

def draw_polygons(
    polygons,
    edge_colors='g',
    line_styles='-',
    line_widths=2,
    face_colors='none',
    alpha=0.8,
):
    from matplotlib.collections import PolyCollection
    if isinstance(line_widths, (int, float)):
        line_widths = [line_widths] * len(polygons)
    line_widths = [
        min(max(linewidth, 1), 4 / 4)
        for linewidth in line_widths
    ]
    polygon_collection = PolyCollection(
        polygons,
        alpha=alpha,
        facecolor=face_colors,
        linestyles=line_styles,
        edgecolors=edge_colors,
        linewidths=line_widths)
    return polygon_collection

VOD_CLASSES = {
    'Car',
    'Pedestrian',
    'Cyclist'
}
COLOR_MAP = {
    'Car': 'blue',
    'Pedestrian': 'red',
    'Cyclist': 'green',
    'human_depiction': 'black',
    'ride_other': 'black',
    'truck': 'black',
    'bicycle_rack': 'black',
    'ride_uncertain': 'black',
    'rider': 'black',
    'motor': 'black',
    'moped_scooter': 'black',
    'DontCare': 'black',
    'vehicle_other': 'black',
    'bicycle': 'black'
}

def trans_boxes(bboxes, trans_mat):
    '''
    Modified from mmdetection3d/mmdet3d/structures/bbox_3d/lidar_box3d.py:rotate
    Args:
        bboxes: np.ndarray [N, 7]        
        trans_mat: np.ndarray [4, 4]
    '''
    rot_mat_T = trans_mat[:3, :3].T
    rot_sin = rot_mat_T[0, 1]
    rot_cos = rot_mat_T[0, 0]
    angle = np.arctan2(rot_sin, rot_cos)
    bboxes[:, 0:3] = bboxes[:, 0:3] @ rot_mat_T + trans_mat[:3, -1]
    bboxes[:, 6] += angle
    return bboxes

def trans_from_two_boxes(box1, box2):
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

def wrap_points(points_all_frames, gt_boxes_all_frames, pose_all_frames, source_frame_idx):
    '''
    Wrap all points to source frame.
    Args:
        points_all_frames: List[np.ndarray[N, C]]
        gt_boxes_all_frames: List[np.ndarray[Nb, 7+Cb]]
        pose_all_frames: List[np.dnarray[4, 4]] 每帧中点坐标到世界坐标的变换矩阵
        source_frame_idx: 需要变换到的帧索引
    Rets:
        wrapped_points_all_frames: List[np.ndarray[N, C]] 所有帧的点都变换到了目标帧下
        wrapped_boxes_all_frames: List[np.ndarray[Nb, 7+Cb]]
    '''
    wrapped_points_list = []
    wrapped_boxes_list = []
    for target_frame_idx in range(len(points_all_frames)):
        # 当前帧为source帧 不需要变换
        if target_frame_idx == source_frame_idx:
            wrapped_points_list.append(points_all_frames[source_frame_idx].copy())
            wrapped_boxes_list.append(gt_boxes_all_frames[source_frame_idx].copy())
            continue
        # 得到target到source的变换矩阵
        t_world_target = pose_all_frames[target_frame_idx]
        t_world_source = pose_all_frames[source_frame_idx]
        t_source_target = np.dot(np.linalg.inv(t_world_source), t_world_target) 

        # 变换点
        wrapped_points = points_all_frames[target_frame_idx].copy()
        wrapped_points_hom = np.concatenate([wrapped_points[:, :3], np.ones((wrapped_points.shape[0], 1))], axis=-1)
        wrapped_points[:, :3] = (wrapped_points_hom @ t_source_target.T)[:, :3]

        # 变换框
        wrapped_boxes = gt_boxes_all_frames[target_frame_idx].copy()
        wrapped_boxes[:, :7] = trans_boxes(wrapped_boxes[:, :7], t_source_target)
        wrapped_points_list.append(wrapped_points)
        wrapped_boxes_list.append(wrapped_boxes)
    return wrapped_points_list, wrapped_boxes_list

def wrap_boxes(wrapped_points_all_frames, wrapped_boxes_all_frames, source_frame_idx, keep_background=False):
    """
    根据跟踪标签补偿目标运动

    Args:
        wrapped_points_all_frames: List[np.ndarray[N, C]] 已经补偿了自车运动的点集
        wrapped_boxes_all_frames: List[np.ndarrray[Nb, 8(loc, dim, rot, track_id)]] 已经补偿了自车运动的检测框
        source_frame_idx: 需要变换到的帧索引
        keep_background: 是否保留其他帧的背景点 若保留则返回的aggregated_points中所有帧都包含背景点 否则只有source帧包含背景
    Returns:
        aggregated_points: 聚合好的点, 包含目标帧的所有点, 以及其他帧变换过来的点, 不包含其他帧的背景点
    """
    
    gt_boxes_source_frame = wrapped_boxes_all_frames[source_frame_idx]
    aggregated_boxes_all_frames = []
    aggregated_points_all_frames = []
    keep_mask_all_frames = []
    for k in range(len(wrapped_points_all_frames)):
        # 当前帧为source帧 不需要变换
        if k == source_frame_idx:
            aggregated_boxes_all_frames.append(gt_boxes_source_frame)
            aggregated_points_all_frames.append(wrapped_points_all_frames[k])
            keep_mask_all_frames.append(np.ones((gt_boxes_source_frame.shape[0],), dtype=bool))
            continue
        wrapped_boxes = wrapped_boxes_all_frames[k]
        wrapped_points = wrapped_points_all_frames[k]
        agg_points_all_boxes = []
        agg_boxes = []
        points_in_boxes_mask = []
        boxes_keep_mask = np.ones((wrapped_boxes.shape[0],), dtype=bool)
        for j, box in enumerate(wrapped_boxes):
            # 找到目标帧中有相同track id的物体
            source_box = None
            for b in gt_boxes_source_frame:
                if int(b[-1]) == int(box[-1]):
                    source_box = b
                    break
            # 如果找不到则不管这个框
            if source_box is None:
                boxes_keep_mask[j] = False
                continue
            # 得到在框内的点
            points_in_box = roiaware_pool3d_utils.points_in_boxes_cpu(wrapped_points[:, :3], box[np.newaxis, :7])
            points_in_box_mask = points_in_box[0] > 0
            points_in_box = wrapped_points[points_in_box_mask]

            # 得到变换矩阵
            t_source_target = trans_from_two_boxes(box, source_box)
            
            # 变换框
            agg_box = box.copy()
            agg_box[:3] += t_source_target[:3, -1]
            agg_box[6] = source_box[6]

            # 变换点 先挪到框坐标系下 执行变换 再挪回去
            agg_points_in_box = points_in_box.copy()
            agg_points_in_box[:, :3] -= box[:3]
            agg_points_in_box[:, :3] = (np.concatenate([agg_points_in_box[:, :3], np.ones((agg_points_in_box.shape[0], 1))], axis=-1) @ t_source_target.T)[:, :3]
            agg_points_in_box[:, :3] += box[:3]

            agg_boxes.append(agg_box)
            agg_points_all_boxes.append(agg_points_in_box)
            points_in_boxes_mask.append(points_in_box_mask)
        
        keep_mask_all_frames.append(boxes_keep_mask)
        agg_points_all_boxes = np.concatenate(agg_points_all_boxes, axis=0)
        agg_boxes = np.concatenate(agg_boxes, axis=0)

        if keep_background:
            # 先把原来的点去掉
            agg_points = wrapped_points.copy()
            mask = np.zeros((agg_points.shape[0],), dtype=bool)
            for m in points_in_boxes_mask:
                mask = np.logical_or(mask, m)
            agg_points = agg_points[np.logical_not(mask)]
            agg_points = np.concatenate([agg_points, agg_points_all_boxes], axis=0)
        else:
            agg_points = agg_points_all_boxes

        aggregated_points_all_frames.append(agg_points)
        aggregated_boxes_all_frames.append(agg_boxes)
    return aggregated_points_all_frames, aggregated_boxes_all_frames, keep_mask_all_frames

def get_subsquence_ids(cur_id, seq_len, num_frames, direction=0, padding=False):
    '''
    获得子序列的indices, 从前往后的顺序
    Args:
        cur_id: 参考帧的索引
        num_frames: 需要取多少帧, 注意: 当direction=0时, 双向取, 实际取的帧数是2*num_frames+1, 当direction=-1时, 只取比cur_id小的, 实际取的帧数是num_frames+1
        direction: 取值为0/-1, 取0时双向取, 取-1时只取左边
        padding: 当没办法取够足够帧时, 是否拿当前帧去pad, 保证最后输出的结果长度一致
    '''
    i = cur_id
    if (i-num_frames) < 0:
        if direction == 0:
            n_prev = i
            n_after = (2*num_frames-n_prev)
        elif direction == -1:
            n_prev = i
            n_after = 0
    elif (i+num_frames) >= seq_len:
        if direction == 0:
            n_after = seq_len-i-1
            n_prev = 2*num_frames-n_after
        elif direction == -1:
            n_prev = num_frames
            n_after = 0
    else:
        if direction == 0:
            n_prev = num_frames
            n_after = num_frames
        elif direction == -1:
            n_prev = num_frames
            n_after = 0
    if direction == 0:
        return list(range(i-n_prev, i+n_after+1)), n_prev
    elif direction == -1:
        if not padding:
            return list(range(i-n_prev, i+1)), n_prev
        else:
            return list(range(i-n_prev, i+1)) + [i for _ in range(num_frames-n_prev)], num_frames
    else:
        return None