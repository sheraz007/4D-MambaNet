import numpy as np
import torch
import torch.nn as nn

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

class MultiScaleDynamicVFE_refine(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        ms_voxel_sizes = model_cfg.MS_VOXEL_SIZES
        grid_size = [(point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size) for voxel_size in ms_voxel_sizes]
        grid_size = [np.round(gs).astype(np.int64) for gs in grid_size]
        voxel_size = ms_voxel_sizes
        assert isinstance(voxel_size, (list, tuple)) and isinstance(grid_size, (list, tuple))
        self.use_simple_mean_vfe = self.model_cfg.USE_SIMPLE_MEAN_VFE
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        voxel_size_list = voxel_size
        grid_size_list = grid_size
        self.grid_size = torch.tensor(np.array(grid_size_list)).cuda()
        self.voxel_size = torch.tensor(np.array(voxel_size_list)).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = [voxel_size[0] for voxel_size in voxel_size_list]
        self.voxel_y = [voxel_size[1] for voxel_size in voxel_size_list]
        self.voxel_z = [voxel_size[2] for voxel_size in voxel_size_list]
        self.x_offset = [voxel_x / 2 + point_cloud_range[0] for voxel_x in self.voxel_x]
        self.y_offset = [voxel_y / 2 + point_cloud_range[1] for voxel_y in self.voxel_y]
        self.z_offset = [voxel_z / 2 + point_cloud_range[2] for voxel_z in self.voxel_z]

        self.scale_xyz = [grid_size[0] * grid_size[1] * grid_size[2] for grid_size in grid_size_list]
        self.scale_yz = [grid_size[1] * grid_size[2] for grid_size in grid_size_list]
        self.scale_z = [grid_size[2] for grid_size in grid_size_list]

        # A workaround, as OpenPCDet only optimize leaf module
        self.scale_embeddings = nn.Embedding(2, out_filters, _weight=torch.zeros(2, out_filters))
        self.use_scale_embed = self.model_cfg.get('USE_SCALE_EMBED', True)

    def get_output_feature_dim(self):
        return self.num_point_features

    def filter_base_coords(self, target_coords, source_coords):
        """
        从target_coords中挑出里面source_coords对应位置, 假设source_coords是target_coords子集
        """
        
        # 将三元组转换为单个整数表示，以便使用排序和搜索 b(2位)z(2位)y(4位)x(4位)
        def hash_triplets(triplets):
            # b x y z
            return triplets[:, 0] * self.scale_xyz[0] + triplets[:, 3] * self.scale_yz[0] + triplets[:, 2] * self.scale_z[0] + triplets[:, 1]

        # 哈希化三元组
        hashed_tensor1 = hash_triplets(source_coords)
        hashed_tensor2 = hash_triplets(target_coords)

        # 对tensor2进行排序
        sorted_hashed_tensor2, sorted_indices = torch.sort(hashed_tensor2)

        # 使用torch.searchsorted在排序后的tensor2中找到tensor1的位置
        positions_in_sorted = torch.searchsorted(sorted_hashed_tensor2, hashed_tensor1)

        # 将排序后的索引映射回原始索引
        positions = sorted_indices[positions_in_sorted]
        return positions

    def get_multi_scale_pointcoords(self, points, cur_scale_layer):
        cur_voxel_size = self.voxel_size[cur_scale_layer]
        base_voxel_size = self.voxel_size[0]
        ratio_xyz = cur_voxel_size / base_voxel_size
        shape = cur_voxel_size.shape[0]
        ratio = int(ratio_xyz[0])
        assert ratio_xyz[0] == ratio_xyz[1] == ratio_xyz[2]
        repeat_time = ratio ** shape
        gird = ratio_xyz[0].item()

        # 1. 计算中心
        delta_voxel_center = (cur_voxel_size / 2 - base_voxel_size / 2)
        delta_voxel_center_repeat = delta_voxel_center.repeat(repeat_time).reshape(-1, 3)

        # 2. 偏移中心
        # 事先计算好
        for ix in range(ratio):
            for iy in range(ratio):
                for iz in range(ratio):
                    k =  int(ix * (gird**2) + iy * (gird**1) + iz) # core
                    delta_voxel_center_repeat[k , 0] -= ix* base_voxel_size[0]
                    delta_voxel_center_repeat[k , 1] -= iy* base_voxel_size[1]
                    delta_voxel_center_repeat[k , 2] -= iz* base_voxel_size[2] # 8, 3

        points_repeat = points.repeat(repeat_time, 1).reshape(repeat_time, points.shape[0], points.shape[-1]).permute(1,0,2) # n_p, 8, 3
        point_cloud_range_repeat = self.point_cloud_range.repeat(repeat_time, 1).reshape(repeat_time, self.point_cloud_range.shape[-1])
        cur_voxel_size_repeat = cur_voxel_size.repeat(repeat_time, 1).reshape(repeat_time, cur_voxel_size.shape[-1])
        point_coords_repeat = torch.floor((points_repeat[:,:, 1:4] + delta_voxel_center_repeat - point_cloud_range_repeat[:, 0:3]) / cur_voxel_size_repeat).int()

        for ix in range(ratio):
            for iy in range(ratio):
                for iz in range(ratio):
                    k =  int(ix * (gird**2) + iy * (gird**1) + iz)
                    ixyz = torch.tensor([ix, iy, iz]).cuda()
                    point_coords_repeat[:, k, :] = point_coords_repeat[:, k, :] * ratio + ixyz

        point_coords_repeat = point_coords_repeat.permute(1,0,2) # 8, n_p, n_c
        point_coords_repeat = point_coords_repeat.reshape(-1, point_coords_repeat.shape[-1])
        return point_coords_repeat

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        base_voxel_center = self.voxel_size[0] / 2.0
        voxel_coords_lvl = []
        voxel_features_lvl = []
        for i in range(len(self.voxel_size)):
            points: torch.Tensor = batch_dict['points'] # (batch_idx, x, y, z, i, e)
            if i > 0:
                ratio_xyz = self.voxel_size[i] / self.voxel_size[0]
                point_coords = self.get_multi_scale_pointcoords(points, i)
                points = points.repeat([int(torch.prod(ratio_xyz)), 1])
            else:
                point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()

            mask = ((point_coords >= 0) & (point_coords < self.grid_size[0])).all(dim=1)
            points = points[mask]
            point_coords = point_coords[mask]
            merge_coords = points[:, 0].int() * self.scale_xyz[0] + \
                            point_coords[:, 0] * self.scale_yz[0] + \
                            point_coords[:, 1] * self.scale_z[0] + \
                            point_coords[:, 2]
            
            if self.model_cfg.get('ONLY_KEEP_BASE_COORDS', True) and i > 0:
                base_coords_mask = torch.isin(merge_coords, base_unq_coords)
                points = points[base_coords_mask]
                point_coords = point_coords[base_coords_mask]
                merge_coords = merge_coords[base_coords_mask]
            
            points_data = points[:, 1:].contiguous()
            
            unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, sorted=True)
            unq_coords = unq_coords.int()
            if i == 0:
                base_unq_coords = unq_coords
            voxel_coords = torch.stack((unq_coords // self.scale_xyz[0],
                                        (unq_coords % self.scale_xyz[0]) // self.scale_yz[0],
                                        (unq_coords % self.scale_yz[0]) // self.scale_z[0],
                                        unq_coords % self.scale_z[0]), dim=1)
            voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
            
            points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)

            if self.use_simple_mean_vfe or (i == 0):
                features = points_mean
            else:
                points_mean_xyz = points_mean[:, :3]
                points_xyz = points[:, [1, 2, 3]].contiguous()
                f_cluster = points_xyz - points_mean_xyz[unq_inv, :]
                f_center = torch.zeros_like(points_xyz)
                f_center[:, 0] = points_xyz[:, 0] - (point_coords[:, 0].to(points_xyz.dtype) * self.voxel_x[0] + self.x_offset[0])
                f_center[:, 1] = points_xyz[:, 1] - (point_coords[:, 1].to(points_xyz.dtype) * self.voxel_y[0] + self.y_offset[0])
                # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
                f_center[:, 2] = points_xyz[:, 2] - (point_coords[:, 2].to(points_xyz.dtype) * self.voxel_z[0] + self.z_offset[0])

                if self.use_absolute_xyz:
                    features = [points[:, 1:], f_cluster, f_center]
                else:
                    features = [points[:, 4:], f_cluster, f_center]

                if self.with_distance:
                    points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
                    features.append(points_dist)
                features = torch.cat(features, dim=-1)

                for pfn in self.pfn_layers:
                    features = pfn(features, unq_inv)

            if i > 0 and self.use_scale_embed:
                features += self.scale_embeddings(torch.tensor(i-1).long().cuda())
            voxel_coords_lvl.append(voxel_coords.contiguous())
            voxel_features_lvl.append(features.contiguous())
            # batch_dict['voxel_features'] = points_mean.contiguous()
            # batch_dict['voxel_coords'] = voxel_coords.contiguous()

        if self.model_cfg.get('ONLY_KEEP_BASE_COORDS', True):
            batch_dict['voxel_coords'] = voxel_coords_lvl[0]
            batch_dict['voxel_features'] = torch.cat(voxel_features_lvl, dim=-1)
        return batch_dict
