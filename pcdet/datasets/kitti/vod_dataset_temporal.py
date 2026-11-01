import copy
import pickle
import os
import numpy as np
from pathlib import Path
from skimage import io
from PIL import Image

import json
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from .vod_dataset import VODDataset

from .vod_utils import vod_projectionV2
from pathlib import Path
import torch

from collections import defaultdict

class vodDataset_Temporal(VODDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.in_vis = self.dataset_cfg.get('in_vis', False)
        self.use_tem_feat = self.dataset_cfg.get('use_tem_feat', True)

        self.sta_com = self.dataset_cfg.get('sta_com', False)
        self.motion_com = self.dataset_cfg.get('motion_com', False)
        self.in_distill = self.dataset_cfg.get('in_distill', False)
        # Ensure types are consistent even if yaml provides strings
        self.lidar_path = Path(self.dataset_cfg.get('lidar_path', 'data/vod/lidar'))
        self.clip_path = str(self.dataset_cfg.get('clip_path', 'data/vod/radar_5frames/clips'))

        self.class_wise_num_tf = self.dataset_cfg.get('class_wise_num_tf', [3, 6, 6])
        self.class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        self.num_temporal_frame = self.dataset_cfg.get('num_temporal_frame', 2)

        self.modal = str(self.root_path).split('/')[-1]
        self.num_feature = 7 if 'radar' in self.modal else 4

        # 传递参数 训练集是train 验证集是val
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # root_path的路径是../data/vod/
        # 如果是训练集，指向train，否则指向tesing
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.lidar_split_path = self.lidar_path /  ('training' if self.split != 'test' else 'testing')
        # /data/vod/ImageSets/ 下面一共有三个文件： test.txt train.txt val.txt
        # 根据split选择一个 训练集是train 验证集是val
        
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 得到选取的.txt文件下的序列号，组成sample_id_list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        # 创建用于存放vod信息的空列表
        self.kitti_infos = []
        # 调用函数，加载vod数据，mode的值为train或者val
        self.include_kitti_data(self.mode)
        self._lidar_idx_to_info_index = None

    def _ensure_lidar_idx_map(self):
        if self._lidar_idx_to_info_index is None:
            self._lidar_idx_to_info_index = {}
            for idx, info in enumerate(self.kitti_infos):
                try:
                    lidar_idx = info['point_cloud']['lidar_idx']
                except Exception:
                    continue
                self._lidar_idx_to_info_index[str(lidar_idx)] = idx

    def _get_info_by_lidar_idx(self, lidar_idx: str):
        self._ensure_lidar_idx_map()
        idx = self._lidar_idx_to_info_index.get(str(lidar_idx), None)
        if idx is None:
            return None
        return self.kitti_infos[idx]

    # 读取时序的帧
    def get_temporal_radar(self, idx, info): # create_tem_info
        cur_lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        radar_list = []
        temporal_list = []
        cur_radar = np.fromfile(str(cur_lidar_file), dtype=np.float32).reshape(-1, self.num_feature)
        cur_radar_con = np.concatenate((cur_radar, np.zeros((cur_radar.shape[0],1))), axis=1)
        temporal_list.append(0)
        radar_list.append(cur_radar_con)

        # Some info PKLs may miss 'clip_index'. When missing, fall back to "no temporal"
        # behavior (reuse current frame) instead of crashing.
        cur_lidar_clip = None
        try:
            if isinstance(info, dict):
                cur_lidar_clip = info.get('clip_index', None)
        except Exception:
            cur_lidar_clip = None

        cur_data_list = None
        if cur_lidar_clip is not None:
            try:
                with open(os.path.join(self.clip_path, cur_lidar_clip + '.txt')) as f:
                    cur_data_list = f.readlines()
                    cur_data_list = [line.strip() for line in cur_data_list]
            except Exception:
                cur_data_list = None

        for i in range(1, 1 + self.num_temporal_frame):
            before_lidar_idx = "{:05d}".format(int(idx) - i)
            
            if (cur_data_list is not None) and (before_lidar_idx in cur_data_list):
                temporal_list.append(i)
            else:
                temporal_list.append(0)
                before_lidar_idx = idx
            before_lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % before_lidar_idx)
            before_radar = np.fromfile(str(before_lidar_file), dtype=np.float32).reshape(-1, self.num_feature)
            before_radar_con = np.concatenate((before_radar, i * np.ones((before_radar.shape[0],1))), axis=1)
            radar_list.append(before_radar_con)
 
        return radar_list, temporal_list
    
    def get_temporal_calib(self, idx, temporal_list):
        calib_list = []
        for i in temporal_list:
            cur_lidar_idx = "{:05d}".format(int(idx) - i)
            before_calib_file = self.root_split_path / 'calib' / ('%s.txt' % cur_lidar_idx)
            before_calib = calibration_kitti.Calibration(before_calib_file)
            calib_list.append(before_calib)
        return calib_list
        
    def get_temporal_pose(self, idx, temporal_list):
        pose_list = []
        for i in temporal_list:
            cur_lidar_idx = "{:05d}".format(int(idx) - i)
            jsons = []
            pose_file = self.root_split_path / 'pose' / ('%s.json' % cur_lidar_idx)
            try:
                with open(pose_file, 'r') as f:
                    for line in f:
                        jsons.append(json.loads(line))
            except Exception:
                jsons = []

            # Robust fallback: if pose file is missing or malformed, use identity transforms
            if len(jsons) == 0:
                t_odom_camera = np.eye(4, dtype=np.float32)
                t_map_camera = np.eye(4, dtype=np.float32)
                t_utm_camera = np.eye(4, dtype=np.float32)
            else:
                t_odom_camera = np.array(jsons[0].get("odomToCamera", np.eye(4)), dtype=np.float32).reshape(4, 4)
                t_map_camera = None
                t_utm_camera = None
                if len(jsons) > 1 and "mapToCamera" in jsons[1]:
                    t_map_camera = np.array(jsons[1]["mapToCamera"], dtype=np.float32).reshape(4, 4)
                if len(jsons) > 2 and "UTMToCamera" in jsons[2]:
                    t_utm_camera = np.array(jsons[2]["UTMToCamera"], dtype=np.float32).reshape(4, 4)
            pose = {
                't_odom_camera': t_odom_camera,
                't_map_camera': t_map_camera,
                't_utm_camera': t_utm_camera
            }
            pose_list.append(pose)
        return pose_list
    
    def get_pose(self, idx):
        jsons = []
        pose_file = self.root_split_path / 'pose' / ('%s.json' % idx)
        try:
            with open(pose_file, 'r') as f:
                for line in f:
                    jsons.append(json.loads(line))
        except Exception:
            jsons = []

        if len(jsons) == 0:
            t_odom_camera = np.eye(4, dtype=np.float32)
        else:
            t_odom_camera = np.array(jsons[0].get("odomToCamera", np.eye(4)), dtype=np.float32).reshape(4, 4)
        # Keep map/utm as identity if missing
        t_map_camera = np.eye(4, dtype=np.float32)
        t_utm_camera = np.eye(4, dtype=np.float32)
        pose = {
            't_odom_camera': t_odom_camera,
            't_map_camera': t_map_camera,
            't_utm_camera': t_utm_camera
        }
        return pose

    def get_calib(self, idx):
        # /data/vod/training/calib/xxxxxx.txt
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        # 不存在直接报错
        assert calib_file.exists()

        return calibration_kitti.Calibration(calib_file)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:
            true if the point in teh fov

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        # 判断投影点是否在图像范围内
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        # 深度大于0， 才可以判断fov视角
        # pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个
        # 用于判断该点云能否有效 （是否用于训练）
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=1, has_label=True, count_inside_pts=True, sample_id_list=None
    ):
        import concurrent.futures as futures
        # 线程函数

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 初始化一个空字典
            info = {}
            # 定义点云的信息，点云的特征维度和索引

            # clip_info
            clip_path = self.clip_path
            clip_name_list = os.listdir(clip_path)
            clip_name_list.sort()
            clip_name_list.pop(-1)
            for i in range(len(clip_name_list)):
                with open(os.path.join(clip_path, clip_name_list[i])) as f:
                    cur_data_list = f.readlines()
                    cur_data_list = [line.strip() for line in cur_data_list]
                if sample_idx in cur_data_list:
                    info['clip_index'] = clip_name_list[i][:-4]
                    break
                else:
                    continue
            pc_info = {'num_features': self.num_feature, 'lidar_idx': sample_idx}
            # 添加点云的信息到 info字典中
            info['point_cloud'] = pc_info

            # 图像信息 : 样本索引和图像的高宽
            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # # 添加图像信息到info字典中
            info['image'] = image_info

            # 根据索引获得Calibration对象
            calib = self.get_calib(sample_idx)
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            # 标定信息 : P2、R0_rect和T_V_C
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            # 添加标定信息
            info['calib'] = calib_info

            if has_label:
                # 根据索引读取label，构造object对象
                obj_list = self.get_label(sample_idx)
                annotations = {}
                # 根据属性将所有的obj_list的属性添加进annotations
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list]) #截断的
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # 计算有效物体的个数，排除掉DonCare类型
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                # 得到总物体个数
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)  # -1 代表 DonCare
                # 由此可以得到 index=[0,1,2,3,4,5,-1,-1,-1,-1]
                annotations['index'] = np.array(index, dtype=np.int32)

                # 假设有效物体的个数是N
                # 去除的有效物体的loc是（N， 3）、dim是（N， 3）， rot_y是（N， 1）信息
                # 因为在vod中类别DonCare一定是放在每个标签的最后，所以可以直接索引
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 通过计算得到在lidar坐标系下的坐标 : loc_lidar （N， 3）
                loc_lidar = calib.rect_to_lidar(loc)
                # 分别取 dims中的第一列、第二列、第三列：l,h,w（N,1）
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # 将物体坐标的原点由物体的底部移动到目标的中心
                loc_lidar[:, 2] += h[:, 0] / 2
                # （N， 7）   [x, y, z, dx, dy, dz, heading]
                # np.newaxis 在列上添加一维，因为rots是（N，）
                # -(np.pi / 2 + rots[..., np.newaxis]) 因为在vod中，
                # camera的坐标系下定义物体的朝向与camera的x轴夹角的顺时针方向为正，逆时针方向为负
                # 在pcdet中，lidar坐标系下定义的物体朝向与lidar的x轴夹角的逆时针方向为正，顺时针方向为负，所以二者正负相反
                # camera:         lidar:        camera： x=right y=down z=forward velodyne: x=forward y=left z=up
                # Y                    X
                # |                    |
                # |____X         Y_____|
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                # 添加注释信息到info字典中
                info['annos'] = annotations
                if count_inside_pts:
                    points = self.get_radar(sample_idx)
                    # 根据索引获取Calibration对象
                    calib = self.get_calib(sample_idx)
                    # 将lidar的坐标系的点变换到rect坐标系
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # 返回true or false list 判断点云是否在fov下，判断该点云能否有效（是否用于训练）
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # 提取有效点
                    pts_fov = points[fov_flag]

                    pts_fov = points # 取所有点为有效点
                    
                    # gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                    # 返回值corners_lidar为（N,8,3）
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    # num_gt是这一帧图像里物体的总个数，假设为10，
                    # 则num_points_in_gt=array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    # num_objects是有效物体的个数，为N，假设N=6
                    for k in range(num_objects):
                        # in_hull函数判断点云是否在bbox中，（是否偶在2d的检测框中）
                        # 如果是，返回flag
                        # 运用到了“三角剖分”的概念和方法
                        # 输入是当前帧FOV视角的点云和第K个box的信息
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        # 计算框内包含的点云
                        num_points_in_gt[k] = flag.sum()
                    # 添加点云数量信息
                    annotations['num_points_in_gt'] = num_points_in_gt
            # infos是一个列表，每一个元素代表了一帧的信息（字典格式）
            return info
        #是.txt文件下的序列号，组成列表sample_id_list，上面的函数的是一个帧的信息
        #下面几行是将该sample_id_list列表上的都执行一下，每个返回的信息info都存放在infos里面
        #最后执行完成后，infos是一个列表，每一个元素代表了一帧的信息
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # 　创建线程池，多线程异步处理数据
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            '''map_fun：你传入的要执行的map函数
                   itr_arg：一个可迭代的参数，可以是列表字典等可迭代的对象
                   基本上和python的map函数一样
                   注意result并不是你map_fun返回的结果，而是一个生成器，如果要从中去结果，你可以使用列表生成式或者其他你想使用的方法
            '''
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        from pathlib import Path
        # 如果是“train”，创建的路径是  /data/vod/gt_database
        root_path = os.path.dirname(info_path)
        database_save_path = Path(root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # 在/data/vod/下创建保存vod_dbinfos_train的文件
        db_info_save_path = Path(root_path) / ('vod_dbinfos_%s.pkl' % split)

        # parents=True，可以同时创建多级目录
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # 传入的参数info_path是一个.pkl文件，ROOT_DIR/data/vod/vod_infos_train.pkl
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 读取infos里的每个info信息，一个info是一帧的数据
        for k in range(len(infos)):
            # 输出的是第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            # 取当前帧的信息 info
            info = infos[k]
            # 取里面的样本序列，其实就是data/vod/ImageSets/train.txt里面的数字序列，
            # 如000000，000003,000007....
            sample_idx = info['point_cloud']['lidar_idx']
            # 读取该bin文件类型，并将点云数据以numpy的格式输出
            # points是一个数组（M,4）
            points = self.get_radar(sample_idx)
            num_points = points.shape[0]
            # 读取注释信息
            annos = info['annos']
            # name的数据是['car','car','pedestrian'...'dontcare'...]表示当前帧里面的所有物体objects
            names = annos['name']
            # difficulty：[0,1,2,-1,0,0,-1,1,...,]里面具体物体的难度，长度为总物体的个数
            difficulty = annos['difficulty']
            # box是一个数组，表示2D物体边框的个数
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]
            # 返回每个box中点云索引[0 0 0 1 0 1 1...]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)
            # num_fg = 0
            # for i in range(point_indices.shape[0]):
            #     m = np.sum(point_indices[i,:]==1)
            #     num_fg = num_fg + m
            # fg_rate = num_fg / num_points
            # print(fg_rate)
            for i in range(num_obj):
                # 创建文件名，并设置保存路径，最后文件如：000007_Cyclist_3.bin
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                # e.g. /data/vod/gt_database/000007_Cyclist_3.bin
                filepath = database_save_path / filename
                # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                # 再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                # 将第i个box内点转化为局部坐标点
                gt_points[:, :3] -= gt_boxes[i, :3]
                # 把gt_points的信息写入文件里
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # 获取文件相对路径
                    db_path = str(filepath.relative_to(root_path))  # gt_database/xxxxx.bin
                    # 根据当前物体的信息组成info
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    # 把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        # 如果存在该类别则追加
                        all_db_infos[names[i]].append(db_info)
                    else:
                        # 如果不存在该类别则新增
                        all_db_infos[names[i]] = [db_info]

        # 输出数据集中不同类别物体的个数
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        # 把所有的all_db_infos写入到文件里面
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                为字典包含batch的calib和image_shape等信息，通过frame_id索引
            pred_dicts: list of pred_dicts 预测列表包含：
                pred_boxes: (N, 7), Tensor 预测的框，包含七个信息
                pred_scores: (N), Tensor   预测得分
                pred_labels: (N), Tensor   预测的类比
            class_names:
            output_path:
        Returns:

        """

        # 获取预测后的模板字典pred_dict，全部定义为全零的向量
        # vod格式增加了score和boxes_lidar信息
        # 参数num_samples表示这一帧里面的物体个数
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            """
            接收模型预测的在统一坐标系下表示的3D检测框，并转回自己所需格式,生成一帧的预测字典
            Args:
                batch_index:batch的索引id
                box_dict:预测的结果，字典包含pred_scores、pred_boxes、pred_labels等信息
            """
            # pred_scores: (N), Tensor      预测得分，N是这一帧预测物体的个数
            # pred_boxes: (N, 7), Tensor    预测的框，包含七个信息
            # pred_labels: (N), Tensor      预测的标签
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            # 定义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # 如果没有物体，则返回空字典
            if pred_scores.shape[0] == 0:
                return pred_dict

            # 获取该帧的标定和图像尺寸信息
            calib = batch_dict['calib'][batch_index]

            image_shape = np.array([1216, 1936])
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # 将预测的box3D转化从lidar系转化到camera系
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # 将camera系下的box3D信息转化为box2D信息
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # 获取id帧号
            frame_id = batch_dict['frame_id'][index]
            # 获取单帧的预测结果
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                # 定义输出结果的文件路径比如： data/vod/output/000007.txt
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    # 从单帧预测dict中，提取bbox、loc和dims等信息
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    # 将预测信息输出至终端同时写入文件
                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        # 返回所有预测信息
        return annos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        # 等于返回训练帧的总个数，等于图片的总个数，帧的总个数
        return len(self.kitti_infos)
    
    def static_compensation(self, points_all_frames, pose_all_frames, calib_all_frames, gt_boxes_all_frames, return_radar2_radar1=False):
        cur_frame_idx = 0
        wrapped_points_all_frames = []
        wrapped_boxes_all_frames = []
        radar2_radar1_list = []

        wrapped_points_all_frames.append(points_all_frames[0])
        wrapped_boxes_all_frames.append(gt_boxes_all_frames[0])

        for i in range(1, len(points_all_frames)):
            t_odom_cam_1 = pose_all_frames[i]['t_odom_camera']
            t_odom_cam_2 = pose_all_frames[cur_frame_idx]['t_odom_camera']
            t_cam_radar_1 = calib_all_frames[i].V2C
            t_cam_radar_2 = calib_all_frames[cur_frame_idx].V2C
            t_cam_radar_1 = np.concatenate( (t_cam_radar_1, np.zeros((1, t_cam_radar_1.shape[1]))), axis=0)
            t_cam_radar_1[-1, -1] = 1
            t_cam_radar_2 = np.concatenate( (t_cam_radar_2, np.zeros((1, t_cam_radar_2.shape[1]))), axis=0)
            t_cam_radar_2[-1, -1] = 1
            odom_radar_1 = np.dot(t_odom_cam_1, t_cam_radar_1)
            odom_radar_2 = np.dot(t_odom_cam_2,t_cam_radar_2)
            radar2_radar1 = np.dot(np.linalg.inv(odom_radar_2), odom_radar_1)
            radar2_radar1_list.append(radar2_radar1)
            # temporal first
            wrapped_points = np.concatenate((points_all_frames[i][:,0:3], np.ones((points_all_frames[i].shape[0], 1))), axis=1) @ radar2_radar1.T
            wrapped_points_xyz = wrapped_points[:,:-1]
            ori_point = points_all_frames[i]
            ori_point[:,0:3] = wrapped_points_xyz
            wrapped_boxes = self.trans_boxes(copy.deepcopy(gt_boxes_all_frames[i]), radar2_radar1)
            wrapped_points_all_frames.append(ori_point)
            wrapped_boxes_all_frames.append(wrapped_boxes)

        if return_radar2_radar1:
            return radar2_radar1_list, wrapped_boxes_all_frames
        return wrapped_points_all_frames, wrapped_boxes_all_frames, radar2_radar1_list

    def trans_boxes(self, bboxes, trans_mat):
        '''
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
        
    def motion_compensation(self, wrapped_points_all_frames, wrapped_boxes_list, gt_name_list):
        motioncom_points_all_frames = [] 
        all_points_all_frames = []
        motioncom_points_all_frames.append(wrapped_points_all_frames[0]) # 当前全取
        all_points_all_frames.append(wrapped_points_all_frames[0])
        
        for frame_id in range(1, len(wrapped_points_all_frames)):
            cur_points = wrapped_points_all_frames[frame_id]
            cur_gt_boxes = wrapped_boxes_list[frame_id]
            cur_gt_names = gt_name_list[frame_id]
            gt_boxes_use_frame = wrapped_boxes_list[0]
            foreground_points_in_boxes = []
            
            for box_id, box in enumerate(cur_gt_boxes):
                target_box = None
                for target_box_id, b in enumerate(gt_boxes_use_frame):
                    if int(b[-1]) == int(box[-1]):
                        target_box = b # before frame box 对应的 cur box
                        break
                if target_box is None:
                    continue
                
                cur_box_label = gt_name_list[0][target_box_id]
                assert cur_box_label == cur_gt_names[box_id]
                class_id = self.class_map[cur_box_label]
                if frame_id > self.class_wise_num_tf[class_id]:
                    continue
                cur_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    cur_points[:, 0:3], box[np.newaxis, :7])  # (nboxes, npoints)
                inbox_mask = (cur_point_indices.reshape(-1) == 1)
                cur_point_in_box = cur_points[inbox_mask].copy()
                t_box_target = self.trans_from_two_boxes(box, target_box)
                cur_point_in_box[:, :3] -= box[:3]
                cur_point_in_box[:, :3] = (np.concatenate((cur_point_in_box[:,0:3], np.ones((cur_point_in_box.shape[0], 1))), axis=1) @ t_box_target.T)[:, :3]
                cur_point_in_box[:, :3] += box[:3]
                cur_gt_aug_point_xyz = cur_point_in_box[:,:3]
                # 覆盖静态点
                cur_points[inbox_mask, 0:3] = cur_gt_aug_point_xyz
                foreground_points_in_boxes.append(cur_points[inbox_mask])

            if len(foreground_points_in_boxes) > 0: # 后续取前景
                motion_com_points = np.concatenate(foreground_points_in_boxes, axis=0)
            else:
                motion_com_points = np.zeros((0, 8), dtype=np.float32)

            all_points_all_frames.append(cur_points)
            motioncom_points_all_frames.append(motion_com_points)

        if self.use_before_background:
            return all_points_all_frames
        return motioncom_points_all_frames

    # 将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的self.prepare_data()
    def __getitem__(self, index):
        """
       从pkl文件中获取相应index的info，然后根据info['point_cloud']['lidar_idx']确定帧号，进行数据读取和其他info字段的读取
       初步读取的data_dict,要传入prepare_data（dataset.py父类中定义）进行统一处理，然后即可返回
       """ 
        
        if self._merge_all_iters_to_one_epoch: # false
            index = index % len(self.kitti_infos)

        if self.in_vis:
            all_frame_ids = [item['point_cloud']['lidar_idx'] for item in self.kitti_infos]
            index = all_frame_ids.index('02080')

        # 拷贝第index帧中的新到到info中
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        # 获取该序列号相应的 图像宽高
        img_shape = info['image']['image_shape']

        # 获取item列表
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        # 定义输入数据的字典包含帧id和标定信息
        input_dict = {
            'frame_id': sample_idx,
        }
        
        # tem input
        radar_list, temporal_list = self.get_temporal_radar(sample_idx, info)

        pose = self.get_temporal_pose(sample_idx, temporal_list)
        input_dict['pose'] = pose

        calib = self.get_temporal_calib(sample_idx, temporal_list)
        input_dict['calib'] = calib
        cur_calib = calib[0]

        # just for teacher
        gt_name_list= []
        gt_boxes_list = []

        # tem input
        for i, tem_idx in enumerate(temporal_list):
            cur_lidar_idx = "{:05d}".format(int(sample_idx) - tem_idx)
            # Prefer mapping by lidar_idx; fall back to index-based if missing
            cur_info_raw = self._get_info_by_lidar_idx(cur_lidar_idx)
            if cur_info_raw is None:
                cur_info_raw = self.kitti_infos[max(index - tem_idx, 0)]
            cur_info = copy.deepcopy(cur_info_raw)

            annos = cur_info['annos']
            cur_calib = calib[i]
            annos = common_utils.drop_info_with_use_name(annos, name=['Car', 'Pedestrian', 'Cyclist'])
            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            trk_id = annos['truncated'].astype(np.float32)
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, cur_calib)
            # 加上track id
            gt_boxes_lidar = np.concatenate([gt_boxes_lidar, trk_id[..., np.newaxis]], axis=-1)
            gt_name_list.append(gt_names)
            gt_boxes_list.append(gt_boxes_lidar)

        input_dict.update({
            'gt_names': gt_name_list,
            'gt_boxes': gt_boxes_list
        })

        if self.sta_com:
            wrapped_points_all_frames, wrapped_boxes_list, radar2_radar1_list = self.static_compensation(radar_list, pose, calib, gt_boxes_list)
        else:
            wrapped_points_all_frames = radar_list
            radar2_radar1_list, wrapped_boxes_list = self.static_compensation(radar_list, pose, calib, gt_boxes_list, return_radar2_radar1=True)
        input_dict['radar2_radar1_list'] = radar2_radar1_list

        box_list = []
        for i in range(len(wrapped_boxes_list)):
            # cur_time_box = gt_boxes_list[i]
            cur_time_box = wrapped_boxes_list[i]
            cur_time_box = np.concatenate((cur_time_box, i * np.ones((cur_time_box.shape[0], 1))), axis=1)
            box_list.append(cur_time_box)
        box_con = np.concatenate(box_list, axis=0)
        input_dict['gt_boxes_list'] = box_con
        input_dict['gt_name_list'] = np.concatenate(gt_name_list, axis=0)

        # 利用gt修正动态点
        if self.motion_com == True:
            motioncom_points_all_frames = self.motion_compensation(wrapped_points_all_frames, wrapped_boxes_list, gt_name_list)
            tem_point = np.concatenate(motioncom_points_all_frames)
        else:
            tem_point = np.concatenate(wrapped_points_all_frames)
            
        if self.use_tem_feat:
            input_dict['temporal_num'] = self.num_temporal_frame # for collete batch
        
        vis = False
        if vis:
            lidar_point = self.get_lidar(sample_idx)
            lidar_point_xyz = lidar_point[:,:3]
            points_xyz = tem_point[:,:3]
            points_xyz_color = np.zeros((points_xyz.shape[0], 3))
            points_xyz_color[:, 0] = 255
            box = gt_boxes_list[0]
            cur_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points_xyz[:, 0:3]), torch.from_numpy(box[:,:7])).numpy()
            mask = np.sum(cur_point_indices, axis=0) > 0
            points_xyz_color[mask, :] = np.array([255, 255, 255])
            points_with_color = np.concatenate((points_xyz, points_xyz_color), axis=1)
            points_with_color[mask].astype(np.float32).tofile('z_tem.bin')

        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = cur_calib.lidar_to_rect(tem_point[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, cur_calib)
            tem_point = tem_point[fov_flag]
            #temporal_point_idx = temporal_point_idx[fov_flag]
        
        input_dict['points'] = tem_point

        # 只需要当前帧的标注就行了
        #input_dict['pose'] = input_dict['pose'][0]
        input_dict['pose_list'] = input_dict['pose']
        input_dict['calib_list'] = input_dict['calib']

        input_dict.pop('pose')
        input_dict['calib'] = input_dict['calib'][0]
        input_dict['gt_boxes'] = input_dict['gt_boxes'][0][:, :7]
        input_dict['gt_names'] = input_dict['gt_names'][0]

        # 加入图片
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        # 加入深度图
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # 加入标定信息
        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = vod_utils.calib_to_matricies(calib)
        
        # only add
        # student_point
        if self.in_distill:
            radar_point = self.get_radar(sample_idx)
            radar_point_pad = np.concatenate((radar_point, -1 * np.ones((radar_point.shape[0], 1))), axis = 1)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = cur_calib.lidar_to_rect(radar_point_pad[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, cur_calib)
                radar_point_pad = radar_point_pad[fov_flag]
                # input_dict['student_points'] = radar_point_pad
                input_dict['points'] = np.concatenate((tem_point ,radar_point_pad), axis = 0)
                input_dict['distill'] = True # for voxelize
        
        input_dict['aug_mat'] = np.eye(4, dtype='float')
        # input_dict['points_before_aug'] = copy.deepcopy(input_dict['points'])
        data_dict = self.prepare_data(input_dict)
        for i in range(len(data_dict['radar2_radar1_list'])):
            data_dict['radar2_radar1_list'][i] = data_dict['aug_mat'] @ data_dict['radar2_radar1_list'][i] @ np.linalg.inv(data_dict['aug_mat'])
        data_dict['image_shape'] = img_shape
        
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 
                            'teacher_voxels', 'teacher_voxel_num_points', 
                            'student_voxels', 'student_voxel_num_points',
                           ]:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)

                elif key in ['points', 'voxel_coords', 
                             'teacher_voxel_coords', 
                             'student_voxel_coords']:
                    coors = []

                    if 'temporal_num' in data_dict:
                        # temporal
                        tempor_val = [[] for i in range(data_dict['temporal_num'][0] + 1)]
                        for cur_val in val:
                            for j in range(data_dict['temporal_num'][0] + 1):
                                cur_time_point_mask = (cur_val[:, -1] == j)
                                cur_time_point = cur_val[cur_time_point_mask]
                                tempor_val[j].append(cur_time_point)
                        tempor_val_list = []
                        for k in range(data_dict['temporal_num'][0] + 1):
                            tempor_val_list += tempor_val[k]
                        for i, coor in enumerate(tempor_val_list):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)
                    else:
                        coors = []
                        if isinstance(val[0], list):
                            val =  [i for item in val for i in item]
                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)

                elif key in ['gt_boxes', 'gt_boxes_list', 'gt_name_list']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['camera_imgs']:
                    ret[key] = torch.stack([torch.stack(imgs,dim=0) for imgs in val],dim=0)
                elif key in ['sp_info', 'calib_list', 'pose_list', 'radar2_radar1_list']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret

def create_vod_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    """
        生成.pkl文件（包括train、test、val），提前读取点云格式、image格式、calib矩阵和label
        可以看到这个函数干了两个事情：
        1、准备了train和val的infos，也就dataset.get_infos这个函数。具体的可以看一下这个函数内部，这里就简单说一下，这个函数内部把train和val对应的所有的label都拿出来，然后去掉不相关的（例如 DonotCare类别），最重要的就是把在相机坐标系下的3D bouding boxes的lable转到了LiDAR坐标系下。然而这个也是只转了中心点的坐标xyz，长宽高与坐标系无关，所以不用转，ry也没转，这个操作我不太理解，为什么不统一直接转过去好了。
        2、准备gt_database，将train下的要预测的box都拿出来，并且把其中的点都抠出来，这一步在gt_aug中会用到
    """
    dataset = vodDataset_Temporal(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    # 定义文件的路径和名称
    train_filename = save_path / ('vod_infos_%s_temporal.pkl' % train_split)
    val_filename = save_path / ('vod_infos_%s_temporal.pkl' % val_split)
    trainval_filename = save_path / 'vod_infos_trainval.pkl'
    test_filename = save_path / 'vod_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    # # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    vod_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(train_filename, 'wb') as f:
       pickle.dump(vod_infos_train, f)
    print('vod info train file is saved to %s' % train_filename)

    # 对验证集的数据进行信息统计并保存
    dataset.set_split(val_split)
    vod_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(vod_infos_val, f)
    print('vod info val file is saved to %s' % val_filename)

    # # 对验证集的数据进行信息统计并保存
    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(vod_infos_train + vod_infos_val, f)
    # print('vod info trainval file is saved to %s' % trainval_filename)

    # # 对验证集的数据进行信息统计并保存
    # dataset.set_split('test')
    # vod_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(vod_infos_test, f)
    # print('vod info test file is saved to %s' % test_filename)

    #print('---------------Start create groundtruth database for data augmentation---------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围点的信息，用于数据增强
    
    #dataset.set_split(train_split)
    #dataset.create_groundtruth_database(info_path=train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_vod_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_vod_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'view_of_delft_PUBLIC' / 'radar_5frames',
            save_path=ROOT_DIR / 'data'/ 'info',
        )
