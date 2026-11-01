import json
import numpy as np
from ...utils import box_utils
import cv2

def vod_get_pose(pose_file):
    jsons = []
    for line in open(pose_file, 'r'):
        jsons.append(json.loads(line))

    t_odom_camera = np.array(jsons[0]["odomToCamera"], dtype=np.float32).reshape(4, 4)
    t_map_camera = None
    t_utm_camera = None
    if len(jsons) > 1:
        t_map_camera = np.array(jsons[1]["mapToCamera"], dtype=np.float32).reshape(4, 4)
        if len(jsons) > 2:
            t_utm_camera = np.array(jsons[2]["UTMToCamera"], dtype=np.float32).reshape(4, 4)
    pose = {
        't_odom_camera': t_odom_camera,
        't_map_camera': t_map_camera,
        't_utm_camera': t_utm_camera
    }
    return pose

def cart_to_hom(pts):
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def vod_projectionV2(radar_point, calib, img_w, img_h):
    pts_img, pts_depth = calib.lidar_to_img(radar_point)

    pts_depth_pad = np.expand_dims(pts_depth, axis=1)
    radar_point_in_imgcoor = np.concatenate((pts_img,pts_depth_pad), axis=1)

    points_2d = radar_point_in_imgcoor[:,:2]
    points_2d = np.floor(points_2d)
    points_x, points_y = points_2d[:, 0], points_2d[:, 1]   

    depths = pts_depth
    valid_mask = (points_x > 0) & (points_x < img_w) & (points_y >0) & (points_y < img_h) & (radar_point_in_imgcoor[:,2] > 0)

    projected_points = np.zeros((radar_point.shape[0], 4))

    valid_projected_points = projected_points[valid_mask]
    valid_projected_points[:, :2] = points_2d[valid_mask]
    valid_projected_points[:, 2] = depths[valid_mask]
    valid_projected_points[:,3] = np.ones((valid_projected_points.shape[0]))
    projected_points[valid_mask] = valid_projected_points 
    return projected_points, valid_mask, depths

def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw

def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask:
        pcd[mask] = 0
    return pcd

def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2