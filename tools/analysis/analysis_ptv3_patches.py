#!/usr/bin/env python3
"""Measure PTv3 patch geometry and attention entropy on real VoD frames.

Part 1. Load the M3-Net Repro strict-sweep checkpoint and run the val loader
(batch size 1). A forward hook on backbone_3d.voxel_attn[2] captures the
SparseConvTensor that module actually receives. Patch groups are the ones that
module built: Point.serialization (its order list and shuffle_orders flag) and
SerializedAttention's patch_size rule plus its own padding. Padding copies are
dropped. For every remaining patch: max 3D distance between voxel centers,
BEV area of the occupied grid cells, and the number of distinct GT boxes plus
background those centers fall into. A hook on the same module's attention
softmax records normalized entropy, entropy / log(patch_size), per query.

Part 2. The same val frames' VoD LiDAR is transformed LiDAR -> rect camera ->
radar using the two calib files, then cropped with the radar point-cloud range,
voxelized at the xcout voxel size, and passed through the same
Point.serialization and padding. Current-frame radar points (tem_idx == 0) go
through that same voxelization. The hooked xcout tensor is reported separately,
including the fraction of tokens that share a coordinate with another token.
No attention entropy for the pure-geometry branches.

Part 3. Points per non-empty voxel for current-frame radar only (tem_idx == 0)
and for the transformed LiDAR, at the input voxel size and at the xcout voxel size.

Every number written to disk is computed in this process. The script does not
read paper tables or fill in constants for the reported statistics.

Run from myopenpcdet/:
  python tools/rev1/analysis_ptv3_patches.py
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.model_utils.pointcept_utils.misc import offset2bincount
from pcdet.models.structures.pointcept_point import Point
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from pcdet.utils import common_utils


CFG_REL = 'tools/cfgs/vod_models/last_model/uni_ptv3_xcout.yaml'
CKPT_REL = 'output/uni_ptv3_xcout_seed_sweep/seed_0/ckpt/checkpoint_epoch_78.pth'
N_FRAMES = 300
BATCH_SIZE = 1
FRAME_SELECTION = '300 frames from numpy.linspace(0, len(dataset)-1, 300).round().astype(int)'

OUT_DIR = ROOT / 'docs' / 'rev1'
FIG_DIR = OUT_DIR / 'figs'
CSV_PATH = OUT_DIR / 'fig4_new_stats.csv'
MD_PATH = OUT_DIR / 'fig4_new_summary.md'
FRAMES_PATH = OUT_DIR / 'fig4_frames.txt'


def xcout_voxel_size_xyz(point_cloud_range, input_voxel_xyz, xcout_spatial_zyx):
    """xcout voxel size in (x, y, z): range extent / hooked spatial_shape.

    spatial_shape is (z, y, x). x and y must equal input_voxel * an integer
    stride greater than 1. z is an effective cell size and may be non-integer,
    because the sparse convolutions downsample z with padding.
    """
    extent = (
        np.asarray(point_cloud_range[3:6], dtype=np.float64)
        - np.asarray(point_cloud_range[0:3], dtype=np.float64)
    )
    spatial_zyx = np.asarray(xcout_spatial_zyx, dtype=np.float64).reshape(3)
    if np.any(spatial_zyx <= 0):
        raise RuntimeError(f'xcout spatial_shape must be positive, got {spatial_zyx.tolist()}')
    voxel = extent / spatial_zyx[::-1]
    base = np.asarray(input_voxel_xyz, dtype=np.float64).reshape(3)
    xy_stride = []
    for axis, name in ((0, 'x'), (1, 'y')):
        ratio = voxel[axis] / base[axis]
        stride = int(np.rint(ratio))
        if stride <= 1 or not np.isclose(ratio, stride, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f'{name} xcout voxel {voxel[axis]} is not input voxel {base[axis]} times an integer stride > 1 '
                f'(ratio {ratio})'
            )
        if not np.isclose(voxel[axis], base[axis] * stride, rtol=0.0, atol=1e-8):
            raise RuntimeError(
                f'{name} xcout voxel {voxel[axis]} != {base[axis]} * {stride}'
            )
        xy_stride.append(stride)
    if not np.isfinite(voxel[2]) or voxel[2] <= 0:
        raise RuntimeError(f'z xcout voxel size must be positive, got {voxel[2]}')
    voxel = np.array([base[0] * xy_stride[0], base[1] * xy_stride[1], voxel[2]], dtype=np.float64)
    print(
        f'xcout voxel size (x, y, z) = ({voxel[0]:.6g}, {voxel[1]:.6g}, {voxel[2]:.6g}) '
        f'xy stride = {tuple(xy_stride)}',
        flush=True,
    )
    return voxel, tuple(xy_stride)


class XcoutRecorder:
    """Keep only the last voxel_attn[2] call in a forward.

    input_tem runs the backbone on earlier scans before the current-frame
    pass. The last call is the current frame, which is the tensor whose
    indices are scored against the current-frame GT boxes.
    """

    def __init__(self):
        self.n_calls = 0
        self.current = None

    def reset_batch(self):
        self.n_calls = 0
        self.current = None

    def on_pre(self, module, inputs):
        sp = inputs[0]
        self.n_calls += 1
        self.current = {
            'indices': sp.indices.detach(),
            'spatial_shape': tuple(int(v) for v in sp.spatial_shape),
            'batch_size': int(sp.batch_size),
            'blocks': [],
            # Filled by the softmax hook, which runs inside SerializedAttention
            # before that module's own forward hook appends the block record.
            'entropy': {},
        }

    def on_forward(self, module, inputs, output):
        sp = inputs[0]
        self.current['indices'] = sp.indices.detach()
        self.current['spatial_shape'] = tuple(int(v) for v in sp.spatial_shape)

    def on_attn(self, block_index, module, inputs, output):
        point = inputs[0]
        order_index = int(module.order_index)
        if block_index not in self.current['entropy']:
            raise RuntimeError(f'softmax hook did not fire for block {block_index}')
        entropy, softmax_k = self.current['entropy'][block_index]
        self.current['blocks'].append({
            'block_index': block_index,
            'order_index': order_index,
            'patch_size': int(module.patch_size),
            'patch_size_max': int(module.patch_size_max),
            'pad': point['pad'].detach(),
            'serialized_order': point.serialized_order[order_index].detach(),
            'n_orders': int(point.serialized_order.shape[0]),
            'entropy': entropy,
            'softmax_k': softmax_k,
        })

    def on_softmax(self, block_index, module, inputs, output):
        attn = output.detach()
        # SerializedAttention softmax over keys: (N', heads, queries, keys).
        k = int(attn.shape[-1])
        if k <= 1:
            ent = torch.zeros(attn.shape[:-1], device=attn.device, dtype=torch.float32)
        else:
            prob = attn.float().clamp_min(torch.finfo(torch.float32).tiny)
            ent = -(prob * prob.log()).sum(dim=-1) / math.log(k)
        # (N', heads, queries). Move off the GPU before the next block.
        self.current['entropy'][block_index] = (ent.float().cpu(), k)


def padding_duplicate_mask(ordered_point_index):
    """True where get_padding_and_inverse repeated an earlier point index.

    Padding writes a copy of a previous patch slot into the tail of the last
    patch. The first time a point index appears in the padded order is the
    real voxel; a later copy is padding. Real count must equal the number of
    distinct point indices.
    """
    first = {}
    mask = torch.zeros(ordered_point_index.numel(), dtype=torch.bool, device=ordered_point_index.device)
    for pos, point_index in enumerate(ordered_point_index.tolist()):
        if point_index in first:
            mask[pos] = True
        else:
            first[point_index] = pos
    n_real = int((~mask).sum().item())
    if n_real != len(first):
        raise RuntimeError('padding mask does not keep each point index once')
    return mask


def patch_rows(ordered_point_index, patch_size):
    n_pad = int(ordered_point_index.numel())
    if patch_size <= 0 or n_pad % patch_size != 0:
        raise RuntimeError(f'padded length {n_pad} is not a multiple of patch_size {patch_size}')
    dup = padding_duplicate_mask(ordered_point_index)
    n_patches = n_pad // patch_size
    ordered = ordered_point_index.view(n_patches, patch_size)
    real = (~dup).view(n_patches, patch_size)
    return ordered, real


def xyz_index(indices_bzyx, point_index=None):
    """(batch, z, y, x) -> (x, y, z) index. Last axis is the coordinate axis."""
    selected = indices_bzyx if point_index is None else indices_bzyx[point_index]
    zyx = selected[..., 1:4].float()
    return zyx.flip(-1)


def voxel_centers_xyz(indices_bzyx, point_index, voxel_size_xyz, point_cloud_range):
    """Center of each voxel: (idx_xyz + 0.5) * stage voxel size + range origin."""
    xyz_idx = xyz_index(indices_bzyx, point_index)
    voxel_size = torch.as_tensor(voxel_size_xyz, device=xyz_idx.device, dtype=xyz_idx.dtype)
    origin = torch.as_tensor(point_cloud_range[:3], device=xyz_idx.device, dtype=xyz_idx.dtype)
    return (xyz_idx + 0.5) * voxel_size + origin


def bev_cell_area(indices_bzyx, point_index, real_mask, voxel_size_xyz):
    """Area of the axis-aligned BEV cells a patch occupies, in square metres."""
    xyz_idx = xyz_index(indices_bzyx, point_index)
    x_idx = xyz_idx[..., 0]
    y_idx = xyz_idx[..., 1]
    x_idx = x_idx.masked_fill(~real_mask, float('nan'))
    y_idx = y_idx.masked_fill(~real_mask, float('nan'))
    x_span = torch.nan_to_num(x_idx, nan=-1e9).amax(dim=1) - torch.nan_to_num(x_idx, nan=1e9).amin(dim=1) + 1
    y_span = torch.nan_to_num(y_idx, nan=-1e9).amax(dim=1) - torch.nan_to_num(y_idx, nan=1e9).amin(dim=1) + 1
    return x_span * float(voxel_size_xyz[0]) * y_span * float(voxel_size_xyz[1])


def max_pairwise_distance(centers, real_mask):
    dist = torch.cdist(centers, centers)
    pair = real_mask.unsqueeze(2) & real_mask.unsqueeze(1)
    dist = dist.masked_fill(~pair, 0)
    return dist.amax(dim=(1, 2))


def patch_label_breakdown(centers_xyz, real_mask, gt_boxes_xyz):
    """Per patch: whether real voxels use ≥2 labels, and the foreground-voxel fraction.

    A label is one GT box, or background. A voxel in any box is foreground.
    Foreground fraction is nan when the patch touches no GT box.
    """
    centers_cpu = centers_xyz.detach().float().cpu()
    real_cpu = real_mask.detach().cpu()
    n_patches, k, _ = centers_cpu.shape
    flat = centers_cpu.reshape(-1, 3)
    n_real = real_cpu.sum(dim=1)
    if gt_boxes_xyz.shape[0] == 0:
        multi = np.zeros(n_patches, dtype=np.bool_)
        fg_frac = np.full(n_patches, np.nan, dtype=np.float64)
        return multi, fg_frac
    inside = points_in_boxes_cpu(flat, torch.as_tensor(gt_boxes_xyz, dtype=torch.float32))
    inside = inside.view(gt_boxes_xyz.shape[0], n_patches, k).bool() & real_cpu.unsqueeze(0)
    boxes_hit = inside.any(dim=-1).sum(dim=0)
    in_any = inside.any(dim=0)
    has_background = (real_cpu & ~in_any).any(dim=-1)
    n_labels = boxes_hit + has_background.long()
    multi = (n_labels >= 2).numpy()
    fg = (real_cpu & in_any).sum(dim=1).float()
    denom = n_real.clamp_min(1).float()
    fg_frac = (fg / denom).numpy().astype(np.float64)
    fg_frac[boxes_hit.numpy() == 0] = np.nan
    return multi, fg_frac


def distinct_box_counts(centers_xyz, real_mask, gt_boxes_xyz):
    """Count of GT boxes that contain at least one real voxel, plus 1 if any voxel is outside every box."""
    centers_cpu = centers_xyz.detach().float().cpu()
    real_cpu = real_mask.detach().cpu()
    n_patches, k, _ = centers_cpu.shape
    flat = centers_cpu.reshape(-1, 3)
    if gt_boxes_xyz.shape[0] == 0:
        inside = torch.zeros((0, flat.shape[0]), dtype=torch.int32)
    else:
        inside = points_in_boxes_cpu(flat, torch.as_tensor(gt_boxes_xyz, dtype=torch.float32))
        inside = inside.view(gt_boxes_xyz.shape[0], n_patches, k)
    real_flat = real_cpu
    if gt_boxes_xyz.shape[0] == 0:
        box_count = torch.zeros(n_patches, dtype=torch.int64)
        in_any = torch.zeros(n_patches, k, dtype=torch.bool)
    else:
        hit = (inside.bool() & real_flat.unsqueeze(0)).any(dim=-1)
        box_count = hit.sum(dim=0)
        in_any = inside.bool().any(dim=0)
    background = (real_flat & ~in_any).any(dim=-1)
    return (box_count + background.long()).numpy()


def geometry_for_block(indices, block, voxel_size_xyz, point_cloud_range, gt_boxes):
    ordered_flat = block['serialized_order'][block['pad']]
    ordered, real = patch_rows(ordered_flat, block['patch_size'])
    if int(real.sum().item()) != int(indices.shape[0]):
        raise RuntimeError(
            f"real voxels {int(real.sum().item())} != sparse voxels {int(indices.shape[0])}"
        )
    centers = voxel_centers_xyz(indices, ordered, voxel_size_xyz, point_cloud_range)
    max_dist = max_pairwise_distance(centers, real).detach().cpu().numpy()
    area = bev_cell_area(indices, ordered, real, voxel_size_xyz).detach().cpu().numpy()
    n_labels = distinct_box_counts(centers, real, gt_boxes)
    multi_label, fg_frac = patch_label_breakdown(centers, real, gt_boxes)
    n_real = real.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    query_is_pad = (~real).detach().cpu().numpy()
    return {
        'max_dist_m': max_dist,
        'bev_area_m2': area,
        'n_labels': n_labels.astype(np.int64),
        'multi_label': multi_label,
        'fg_frac': fg_frac,
        'n_real_voxels': n_real,
        'query_is_pad': query_is_pad,
        'patch_size': int(block['patch_size']),
        'order_index': int(block['order_index']),
        'centers': centers.detach().float().cpu().numpy(),
        'real_mask': real.detach().cpu().numpy(),
    }


def range_violation_m(centers, point_cloud_range):
    """Largest distance any coordinate sits outside [min, max]. Negative means inside."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return 0.0
    lo = np.asarray(point_cloud_range[:3], dtype=np.float64)
    hi = np.asarray(point_cloud_range[3:6], dtype=np.float64)
    outside = np.maximum(lo - pts, pts - hi)
    return float(outside.max())


def nearest_point_violation_m(centers, points_xyz, voxel_size_xyz):
    """Max (nearest-point distance - voxel diagonal). Negative means within the diagonal."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    src = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    diagonal = float(np.linalg.norm(np.asarray(voxel_size_xyz, dtype=np.float64)))
    if pts.shape[0] == 0:
        return 0.0, diagonal
    if src.shape[0] == 0:
        raise RuntimeError('nearest-point check has voxel centers but no radar points')
    centers_t = torch.as_tensor(pts, dtype=torch.float32)
    src_t = torch.as_tensor(src, dtype=torch.float32)
    nearest = torch.cdist(centers_t, src_t).amin(dim=1)
    return float((nearest - diagonal).max().item()), diagonal


def serialize_like_ptv3(grid_zyx, ptv3_module):
    """Call the same Point.serialization and the same padding the xcout module uses."""
    device = grid_zyx.device
    n = int(grid_zyx.shape[0])
    if n == 0:
        return []
    point = Point(dict(
        grid_coord=grid_zyx.int().contiguous(),
        batch=torch.zeros(n, dtype=torch.long, device=device),
        feat=torch.zeros(n, 1, device=device),
    ))
    point.serialization(order=list(ptv3_module.order), shuffle_orders=bool(ptv3_module.shuffle_orders))
    attn = ptv3_module.enc._modules['block0'].attn
    patch_size_max = int(attn.patch_size_max)
    n_min = int(offset2bincount(point.offset).min().item())
    patch_size = max(1, min(n_min, patch_size_max))
    saved = attn.patch_size
    attn.patch_size = patch_size
    try:
        pad, _, _ = attn.get_padding_and_inverse(point)
    finally:
        attn.patch_size = saved
    blocks = []
    for name, block in ptv3_module.enc._modules.items():
        order_index = int(block.attn.order_index)
        blocks.append({
            'order_index': order_index,
            'patch_size': patch_size,
            'patch_size_max': patch_size_max,
            'pad': pad.detach(),
            'serialized_order': point.serialized_order[order_index].detach(),
            'block_name': name,
        })
    return blocks


def valid_gt_boxes(gt_boxes):
    if torch.is_tensor(gt_boxes):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    boxes = np.asarray(gt_boxes[0], dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])
    keep = np.abs(boxes[:, 3:6]).sum(axis=1) > 0
    return boxes[keep, :7].astype(np.float32)


def points_per_voxel_counts(xyz, point_cloud_range, voxel_size_xyz):
    """Non-empty voxel occupancies. Same floor rule as MultiScaleDynamicVFE level 0."""
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(point_cloud_range[:3], dtype=np.float64)
    limit = np.asarray(point_cloud_range[3:6], dtype=np.float64)
    voxel = np.asarray(voxel_size_xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    grid = np.floor((xyz - origin) / voxel).astype(np.int64)
    grid_size = np.round((limit - origin) / voxel).astype(np.int64)
    mask = np.all((grid >= 0) & (grid < grid_size), axis=1)
    grid = grid[mask]
    if grid.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, counts = np.unique(grid, axis=0, return_counts=True)
    return counts.astype(np.int64)


def voxelize_indices(xyz, point_cloud_range, voxel_size_xyz):
    """Non-empty voxel indices as (batch=0, z, y, x), matching sparse-conv layout."""
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(point_cloud_range[:3], dtype=np.float64)
    limit = np.asarray(point_cloud_range[3:6], dtype=np.float64)
    voxel = np.asarray(voxel_size_xyz, dtype=np.float64)
    grid = np.floor((xyz - origin) / voxel).astype(np.int64)
    grid_size = np.round((limit - origin) / voxel).astype(np.int64)
    mask = np.all((grid >= 0) & (grid < grid_size), axis=1)
    grid = grid[mask]
    if grid.shape[0] == 0:
        empty = torch.zeros((0, 4), dtype=torch.int32)
        return empty, torch.zeros((0, 3), dtype=torch.int32)
    uniq = np.unique(grid, axis=0)
    x_idx, y_idx, z_idx = uniq[:, 0], uniq[:, 1], uniq[:, 2]
    indices = np.stack([
        np.zeros(uniq.shape[0], dtype=np.int32),
        z_idx.astype(np.int32),
        y_idx.astype(np.int32),
        x_idx.astype(np.int32),
    ], axis=1)
    grid_zyx = np.stack([z_idx, y_idx, x_idx], axis=1).astype(np.int32)
    return torch.from_numpy(indices), torch.from_numpy(grid_zyx)


LIDAR_TO_RADAR_TRANSFORM = (
    'xyz_radar = radar_calib.rect_to_lidar(lidar_calib.lidar_to_rect(xyz_lidar)). '
    'lidar_calib is lidar_split_path/calib/<frame>.txt (Tr_velo_to_cam, R0_rect). '
    'radar_calib is root_split_path/calib/<frame>.txt (Tr_velo_to_cam, R0_rect). '
    'lidar_to_rect applies V2C then R0. rect_to_lidar inverts R0 and the radar V2C. '
    'The radar point-cloud range is applied only after this transform.'
)


def lidar_xyz_in_radar_frame(dataset, frame_id):
    """VoD LiDAR points in the radar frame, then the radar range mask."""
    from pcdet.utils import calibration_kitti

    frame_id = str(frame_id)
    path = Path(dataset.lidar_split_path) / 'velodyne' / f'{frame_id}.bin'
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise RuntimeError(f'{path} has {raw.size} float32 values, not divisible by 4')
    xyz_lidar = raw.reshape(-1, 4)[:, :3]
    lidar_calib = calibration_kitti.Calibration(
        Path(dataset.lidar_split_path) / 'calib' / f'{frame_id}.txt'
    )
    radar_calib = dataset.get_calib(frame_id)
    xyz_rect = lidar_calib.lidar_to_rect(xyz_lidar)
    xyz_radar = radar_calib.rect_to_lidar(xyz_rect)
    mask = common_utils.mask_points_by_range(xyz_radar, dataset.point_cloud_range)
    return xyz_radar[mask]


def current_frame_radar_xyz(points):
    """Dataloader points whose temporal slot is the current frame (tem_idx == 0).

    Collate writes the temporal slot into column 0 when batch size is 1, and
    leaves the original tem_idx in the last column. Both must be 0.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 5:
        raise RuntimeError(f'unexpected radar point shape {points.shape}')
    slot = points[:, 0]
    tem_idx = points[:, -1]
    if not np.array_equal(slot, tem_idx):
        raise RuntimeError('batch-slot column and tem_idx column disagree; batch size is not 1')
    current = points[tem_idx == 0]
    if current.shape[0] == 0:
        raise RuntimeError('no current-frame radar points (tem_idx == 0)')
    return current[:, 1:4]


def duplicate_coord_fraction(indices_bzyx):
    """Fraction of tokens whose (z, y, x) index is shared with another token."""
    coords = indices_bzyx[:, 1:4].detach().cpu().numpy()
    if coords.shape[0] == 0:
        return 0.0, 0, 0
    _, counts = np.unique(coords, axis=0, return_counts=True)
    n_tokens = int(coords.shape[0])
    n_dup_tokens = int(counts[counts > 1].sum())
    return n_dup_tokens / n_tokens, n_dup_tokens, n_tokens


def without_padded_tail(values, query_is_pad):
    """Drop the last patch when it contains padding copies. Leave a full tail."""
    values = np.asarray(values)
    n_patches = int(query_is_pad.shape[0])
    if values.shape[0] != n_patches:
        raise RuntimeError(f'expected one row per patch, got {values.shape[0]} vs {n_patches}')
    if n_patches == 0:
        return values, query_is_pad
    if bool(np.asarray(query_is_pad[-1]).any()):
        return values[:-1], query_is_pad[:-1]
    return values, query_is_pad


def dist_stats(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    keys = ['count', 'mean', 'std', 'min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max']
    if arr.size == 0:
        return {k: float('nan') for k in keys}
    out = {
        'count': float(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        'min': float(arr.min()),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(arr.max()),
    }
    return out


def append_stats(rows, section, sensor, quantity, values, n_frames, note):
    stats = dist_stats(values)
    for stat, value in stats.items():
        rows.append({
            'section': section,
            'sensor': sensor,
            'quantity': quantity,
            'stat': stat,
            'value': value,
            'n_frames': n_frames,
            'n_items': stats['count'],
            'note': note,
        })
    return stats


def append_hist(rows, section, sensor, quantity, values, n_frames, note, bins):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=bins)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        rows.append({
            'section': section,
            'sensor': sensor,
            'quantity': quantity + '_hist',
            'stat': f'{lo:.6g}:{hi:.6g}',
            'value': int(count),
            'n_frames': n_frames,
            'n_items': int(arr.size),
            'note': note,
        })
    return counts, edges


def hist_bins_from_data(a, b, n_bins=40):
    both = np.concatenate([
        np.asarray(a, dtype=np.float64).reshape(-1),
        np.asarray(b, dtype=np.float64).reshape(-1),
    ])
    both = both[np.isfinite(both)]
    lo = float(both.min()) if both.size else 0.0
    hi = float(both.max()) if both.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def integer_hist_bins(values, cap):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    hi = int(min(cap, arr.max() if arr.size else 1))
    hi = max(hi, 1)
    return np.arange(0.5, hi + 1.5, 1.0)


class OnlineList:
    def __init__(self):
        self.chunks = []

    def add(self, arr):
        arr = np.asarray(arr).reshape(-1)
        if arr.size:
            self.chunks.append(arr)

    def array(self, dtype=np.float64):
        if not self.chunks:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(self.chunks).astype(dtype, copy=False)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['section', 'sensor', 'quantity', 'stat', 'value', 'n_frames', 'n_items', 'note']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'nan'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def write_summary(path, meta, rows_by_key):
    lines = []
    lines.append('# PTv3 patch measurements (VoD val)')
    lines.append('')
    lines.append(
        f"Frames: {meta['n_frames']}. Selection: {meta['frame_selection']}. "
        f"Batch size: {meta['batch_size']}."
    )
    lines.append(f"Checkpoint: `{meta['ckpt']}`. Config: `{meta['cfg']}`.")
    lines.append(
        f"xcout calls per frame (temporal passes included): min {meta['calls_min']}, "
        f"max {meta['calls_max']}. Statistics use the last call, the current-frame tensor."
    )
    lines.append(
        f"Input voxel size (m), from the loaded VFE: {meta['input_voxel_size']}."
    )
    lines.append(
        f"xcout voxel size (m), (range max − min) / spatial_shape in (x, y, z): {meta['xcout_voxel_size']}."
    )
    lines.append(
        f"x and y equal the input voxel size times integer stride {meta['index_stride_xy']}. "
        'The z cell is an effective size because the sparse convolutions downsample z with padding, '
        'so the z stride is not an integer.'
    )
    lines.append(
        'BEV extent and area use x and y only.'
    )
    lines.append(
        f"Hooked SparseConvTensor spatial_shape (z, y, x), first frame: {meta['spatial_shape']}."
    )
    lines.append(
        f"PTv3 orders on the module: {meta['orders']}. shuffle_orders={meta['shuffle_orders']}. "
        f"patch_size_max read from SerializedAttention: {meta['patch_size_max']}."
    )
    lines.append(
        f"Radar temporal slots present in the dataloader batch: {meta['radar_temporal_slots']}. "
        'Part 3 and the radar pure-geometry branch use only tem_idx == 0.'
    )
    lines.append(f"LiDAR to radar transform: {meta['lidar_transform']}")
    lines.append(
        'Current-frame radar points already passed the dataset range mask and FOV filter. '
        'LiDAR is transformed into the radar frame before that same range mask. '
        'LiDAR is not passed through the camera FOV filter.'
    )
    lines.append(
        f"Hooked xcout tokens that share a grid coordinate with another token: "
        f"{meta['dup_tokens']} / {meta['hook_tokens']} "
        f"({fmt(meta['dup_fraction'])}). "
        'These duplicates come from concatenating conv4/conv5/conv6 onto one grid.'
    )
    lines.append(
        'Patch geometry drops padding copies. Max distance is the Euclidean distance '
        'between voxel centers in metres. BEV extent and area use x and y only: '
        '(x-index span) × voxel_x × (y-index span) × voxel_y. '
        'The z cell is an effective size because the sparse convolutions downsample z with padding. '
        'A patch label is one GT box, or background if a real voxel center lies in no box. '
        'The multi-label fraction is patches whose real voxels use at least two of those labels. '
        'The foreground fraction is, among patches that touch any GT box, the mean share of '
        'real voxel centers that lie in a box.'
    )
    lines.append(
        'Normalized attention entropy is (-Σ p log p) / log(patch_size) on the hooked '
        'softmax. Each head of each query is one sample. The whole last patch is '
        'dropped when that patch contains padding copies. Remaining patches have no '
        'padded query slots. The key axis is still the full patch the softmax used.'
    )
    lines.append(
        'The LiDAR-versus-radar comparison below is the pure-geometry pair '
        '(same voxelization, serialization, and padding). Hooked xcout statistics '
        'are a separate block and are not that comparison.'
    )
    lines.append('')
    lines.append('## Distributions')
    lines.append('')
    lines.append('| section | sensor | quantity | n | mean | std | p50 | p05 | p95 |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for key in rows_by_key:
        st = rows_by_key[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {fmt(st['count'])} | {fmt(st['mean'])} | "
            f"{fmt(st['std'])} | {fmt(st['p50'])} | {fmt(st['p05'])} | {fmt(st['p95'])} |"
        )
    lines.append('')
    lines.append('## Points per non-empty voxel')
    lines.append('')
    lines.append(
        'Radar in this ratio is current-frame points only (tem_idx == 0). '
        'LiDAR is in the radar frame, after the range mask.'
    )
    lines.append(
        f"Ratio of means (LiDAR / radar) at input voxel size: {fmt(meta['ratio_input'])}."
    )
    lines.append(
        f"Ratio of means (LiDAR / radar) at xcout voxel size: {fmt(meta['ratio_xcout'])}."
    )
    lines.append('')
    lines.append('## Pure geometry, LiDAR vs current-frame radar')
    lines.append('')
    lines.append(
        f"Median max distance (m): LiDAR {fmt(meta['cmp_lidar_dist'])}, "
        f"radar geometry {fmt(meta['cmp_radar_dist'])}."
    )
    lines.append(
        f"Median BEV cell area (m^2): LiDAR {fmt(meta['cmp_lidar_area'])}, "
        f"radar geometry {fmt(meta['cmp_radar_area'])}."
    )
    lines.append(
        f"Fraction of patches with ≥2 labels: LiDAR {fmt(meta['cmp_lidar_multi'])}, "
        f"radar geometry {fmt(meta['cmp_radar_multi'])}, "
        f"hooked xcout {fmt(meta['cmp_hook_multi'])}."
    )
    lines.append(
        f"Mean foreground-voxel fraction among patches touching a GT box: "
        f"LiDAR {fmt(meta['cmp_lidar_fg'])}, radar geometry {fmt(meta['cmp_radar_fg'])}, "
        f"hooked xcout {fmt(meta['cmp_hook_fg'])}."
    )
    lines.append('')
    lines.append('Plots: `docs/rev1/figs/`.')
    path.write_text('\n'.join(lines) + '\n')


def save_overlay_hist(path, series, bins, xlabel, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, values in series:
        ax.hist(values, bins=bins, histtype='step', linewidth=1.6, density=True, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_bar_hist(path, series, xlabel, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, values in series:
        vals = np.asarray(values, dtype=np.int64).reshape(-1)
        if vals.size == 0:
            continue
        hi = int(vals.max())
        counts = np.bincount(vals, minlength=hi + 1)
        ax.step(np.arange(hi + 1), counts / counts.sum(), where='mid', label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('fraction')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=CFG_REL)
    parser.add_argument('--ckpt', default=CKPT_REL)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max_frames', type=int, default=N_FRAMES)
    args = parser.parse_args()
    if BATCH_SIZE != 1:
        raise RuntimeError('this measurement is fixed at batch size 1')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; the checkpoint forward runs on GPU.')

    torch.manual_seed(0)
    np.random.seed(0)

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    os.chdir(ROOT)
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.config_ext import apply_yaml_ext

    apply_yaml_ext()
    cfg = EasyDict()
    cfg.ROOT_DIR = ROOT
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(str(cfg_path), cfg)
    logger = common_utils.create_logger()
    _, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    dataset = loader.dataset
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset,
    )
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    ptv3 = model.backbone_3d.voxel_attn[2]
    attn_blocks = list(ptv3.enc._modules.values())
    if not attn_blocks:
        raise RuntimeError('voxel_attn[2] has no encoder blocks')

    recorder = XcoutRecorder()
    ptv3.register_forward_pre_hook(recorder.on_pre)
    ptv3.register_forward_hook(recorder.on_forward)
    for block_index, block in enumerate(attn_blocks):
        block.attn.register_forward_hook(
            lambda mod, inputs, output, bi=block_index: recorder.on_attn(bi, mod, inputs, output)
        )
        block.attn.softmax.register_forward_hook(
            lambda mod, inputs, output, bi=block_index: recorder.on_softmax(bi, mod, inputs, output)
        )

    input_voxel = np.asarray(model.vfe.voxel_size[0].detach().float().cpu().numpy(), dtype=np.float64).reshape(3)
    pc_range = np.asarray(dataset.point_cloud_range, dtype=np.float64)
    index_stride_xy = None
    xcout_voxel = None
    patch_size_max = int(attn_blocks[0].attn.patch_size_max)

    if int(loader.batch_size) != 1:
        raise RuntimeError(f'dataloader batch size is {loader.batch_size}, expected 1')
    if len(dataset) < args.max_frames:
        raise RuntimeError(f'val set has {len(dataset)} frames, need {args.max_frames}')
    chosen = np.linspace(0, len(dataset) - 1, args.max_frames).round().astype(int)
    if chosen.shape != (args.max_frames,) or len(np.unique(chosen)) != args.max_frames:
        raise RuntimeError(f'linspace selection did not yield {args.max_frames} distinct indices')
    frame_ids = [str(dataset.kitti_infos[int(i)]['point_cloud']['lidar_idx']) for i in chosen]
    FRAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRAMES_PATH.write_text('\n'.join(frame_ids) + '\n')
    subset_loader = DataLoader(
        Subset(dataset, chosen.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=dataset.collate_batch,
    )

    hook_dist = OnlineList()
    hook_area = OnlineList()
    hook_labels = OnlineList()
    hook_multi = OnlineList()
    hook_fg = OnlineList()
    radar_dist = OnlineList()
    radar_area = OnlineList()
    radar_labels = OnlineList()
    radar_multi = OnlineList()
    radar_fg = OnlineList()
    lidar_dist = OnlineList()
    lidar_area = OnlineList()
    lidar_labels = OnlineList()
    lidar_multi = OnlineList()
    lidar_fg = OnlineList()
    ent_kept = OnlineList()
    radar_ppv = {'input': OnlineList(), 'xcout': OnlineList()}
    lidar_ppv = {'input': OnlineList(), 'xcout': OnlineList()}
    temporal_slot_ids = []
    call_counts = []
    dup_tokens = 0
    hook_tokens = 0
    spatial_shape = None
    n_frames = 0

    for expected_id, batch_dict in zip(frame_ids, subset_loader):
        if int(batch_dict['batch_size']) != 1:
            raise RuntimeError(f"batch_size {batch_dict['batch_size']} is not 1")
        frame_id = str(batch_dict['frame_id'][0])
        if frame_id != expected_id:
            raise RuntimeError(f'frame {frame_id} is not the selected id {expected_id}')
        radar_points = np.array(batch_dict['points'], copy=True)
        temporal_slot_ids.append(np.unique(radar_points[:, 0]).astype(np.int64))
        gt_boxes = valid_gt_boxes(batch_dict['gt_boxes'])

        radar_xyz = current_frame_radar_xyz(radar_points)
        lidar_xyz = lidar_xyz_in_radar_frame(dataset, frame_id)

        recorder.reset_batch()
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict)
        if recorder.current is None or not recorder.current['blocks']:
            raise RuntimeError(f'voxel_attn[2] did not run on frame {frame_id}')
        call_counts.append(recorder.n_calls)
        captured = recorder.current
        if spatial_shape is None:
            spatial_shape = captured['spatial_shape']
            xcout_voxel, index_stride_xy = xcout_voxel_size_xyz(pc_range, input_voxel, spatial_shape)
            if np.allclose(input_voxel, xcout_voxel):
                raise RuntimeError(
                    f'input and xcout voxel sizes are the same {input_voxel.tolist()} '
                    f'(xy stride {index_stride_xy})'
                )
        radar_ppv['input'].add(points_per_voxel_counts(radar_xyz, pc_range, input_voxel))
        radar_ppv['xcout'].add(points_per_voxel_counts(radar_xyz, pc_range, xcout_voxel))
        lidar_ppv['input'].add(points_per_voxel_counts(lidar_xyz, pc_range, input_voxel))
        lidar_ppv['xcout'].add(points_per_voxel_counts(lidar_xyz, pc_range, xcout_voxel))
        indices = captured['indices']
        if int(indices[:, 0].max().item()) != 0 or captured['batch_size'] != 1:
            raise RuntimeError('expected batch size 1 for the current-frame xcout tensor')
        frame_dup, n_dup, n_tok = duplicate_coord_fraction(indices)
        dup_tokens += n_dup
        hook_tokens += n_tok
        del frame_dup

        frame_centers = []
        radar_centers = []
        for block in captured['blocks']:
            if 'entropy' not in block:
                raise RuntimeError('softmax hook did not fire inside voxel_attn[2]')
            if int(block['softmax_k']) != int(block['patch_size']):
                raise RuntimeError(
                    f"softmax K {block['softmax_k']} != patch_size {block['patch_size']}"
                )
            geo = geometry_for_block(indices, block, xcout_voxel, pc_range, gt_boxes)
            frame_centers.append(geo['centers'])
            hook_dist.add(geo['max_dist_m'])
            hook_area.add(geo['bev_area_m2'])
            hook_labels.add(geo['n_labels'])
            hook_multi.add(geo['multi_label'].astype(np.float64))
            hook_fg.add(geo['fg_frac'][np.isfinite(geo['fg_frac'])])
            ent = np.asarray(block['entropy'])
            # (N', heads, queries) -> (N', queries, heads).
            ent = np.transpose(ent, (0, 2, 1))
            if ent.shape[:2] != geo['query_is_pad'].shape:
                raise RuntimeError(
                    f"entropy shape {ent.shape} does not match patches {geo['query_is_pad'].shape}"
                )
            ent, _kept_pad = without_padded_tail(ent, geo['query_is_pad'])
            ent_kept.add(ent)

        for xyz, dist_acc, area_acc, label_acc, multi_acc, fg_acc, is_radar in (
            (lidar_xyz, lidar_dist, lidar_area, lidar_labels, lidar_multi, lidar_fg, False),
            (radar_xyz, radar_dist, radar_area, radar_labels, radar_multi, radar_fg, True),
        ):
            vox_indices, grid_zyx = voxelize_indices(xyz, pc_range, xcout_voxel)
            vox_indices = vox_indices.cuda()
            grid_zyx = grid_zyx.cuda()
            for block in serialize_like_ptv3(grid_zyx, ptv3):
                geo = geometry_for_block(vox_indices, block, xcout_voxel, pc_range, gt_boxes)
                frame_centers.append(geo['centers'])
                if is_radar:
                    radar_centers.append(geo['centers'][geo['real_mask']])
                dist_acc.add(geo['max_dist_m'])
                area_acc.add(geo['bev_area_m2'])
                label_acc.add(geo['n_labels'])
                multi_acc.add(geo['multi_label'].astype(np.float64))
                fg_acc.add(geo['fg_frac'][np.isfinite(geo['fg_frac'])])

        all_centers = np.concatenate([c.reshape(-1, 3) for c in frame_centers], axis=0)
        range_viol = range_violation_m(all_centers, pc_range)
        radar_real = np.concatenate(radar_centers, axis=0) if radar_centers else np.zeros((0, 3))
        near_viol, voxel_diag = nearest_point_violation_m(radar_real, radar_xyz, xcout_voxel)
        if n_frames == 0:
            print(
                f'sanity frame {frame_id} max range violation {range_viol:.6f} m; '
                f'radar nearest-point violation {near_viol:.6f} m '
                f'(voxel diagonal {voxel_diag:.6f} m)',
                flush=True,
            )
        if range_viol > 1e-3:
            raise RuntimeError(f'frame {frame_id} voxel center outside point_cloud_range by {range_viol} m')
        if near_viol > 1e-3:
            raise RuntimeError(
                f'frame {frame_id} radar voxel center is {near_viol} m beyond one voxel diagonal of the nearest radar point'
            )

        n_frames += 1
        print(
            f'frames {n_frames}/{args.max_frames}  frame_id={frame_id}  xcout_calls={call_counts[-1]}',
            flush=True,
        )

    if n_frames != args.max_frames:
        raise RuntimeError(f'collected {n_frames} frames, required {args.max_frames}')

    hook_dist_a = hook_dist.array()
    hook_area_a = hook_area.array()
    hook_labels_a = hook_labels.array(np.int64)
    radar_dist_a = radar_dist.array()
    radar_area_a = radar_area.array()
    radar_labels_a = radar_labels.array(np.int64)
    lidar_dist_a = lidar_dist.array()
    lidar_area_a = lidar_area.array()
    lidar_labels_a = lidar_labels.array(np.int64)
    hook_multi_a = hook_multi.array()
    hook_fg_a = hook_fg.array()
    radar_multi_a = radar_multi.array()
    radar_fg_a = radar_fg.array()
    lidar_multi_a = lidar_multi.array()
    lidar_fg_a = lidar_fg.array()
    ent_kept_a = ent_kept.array()
    ppv = {
        ('radar', 'input'): radar_ppv['input'].array(np.int64),
        ('radar', 'xcout'): radar_ppv['xcout'].array(np.int64),
        ('lidar', 'input'): lidar_ppv['input'].array(np.int64),
        ('lidar', 'xcout'): lidar_ppv['xcout'].array(np.int64),
    }

    slots = sorted(set(int(v) for ids in temporal_slot_ids for v in ids))
    rows = []
    summary_stats = {}

    def take(section, sensor, quantity, values, note):
        summary_stats[(section, sensor, quantity)] = append_stats(
            rows, section, sensor, quantity, values, n_frames, note,
        )

    note_hook = (
        'hooked xcout tokens, padding copies excluded; both attention blocks; '
        'coordinates can repeat across the multi-scale concatenation'
    )
    note_geom = 'pure geometry: one token per non-empty voxel; padding copies excluded; both orders'
    take('hooked', 'radar_xcout', 'max_pairwise_distance_m', hook_dist_a, note_hook)
    take('hooked', 'radar_xcout', 'bev_cell_area_m2', hook_area_a, note_hook)
    take('hooked', 'radar_xcout', 'n_gt_boxes_plus_background', hook_labels_a, note_hook)
    take('hooked', 'radar_xcout', 'fraction_patches_ge2_labels', hook_multi_a,
         '1 if a patch uses at least two labels (distinct GT boxes, or a GT box and background)')
    if hook_fg_a.size:
        take('hooked', 'radar_xcout', 'fg_voxel_fraction_if_touches_box', hook_fg_a,
             'one sample = one patch that touches a GT box; value = foreground voxels / real voxels')
    take('geometry', 'radar_current', 'max_pairwise_distance_m', radar_dist_a, note_geom)
    take('geometry', 'lidar_in_radar_frame', 'max_pairwise_distance_m', lidar_dist_a, note_geom)
    take('geometry', 'radar_current', 'bev_cell_area_m2', radar_area_a, note_geom)
    take('geometry', 'lidar_in_radar_frame', 'bev_cell_area_m2', lidar_area_a, note_geom)
    take('geometry', 'radar_current', 'n_gt_boxes_plus_background', radar_labels_a, note_geom)
    take('geometry', 'lidar_in_radar_frame', 'n_gt_boxes_plus_background', lidar_labels_a, note_geom)
    take('geometry', 'radar_current', 'fraction_patches_ge2_labels', radar_multi_a,
         '1 if a patch uses at least two labels (distinct GT boxes, or a GT box and background)')
    take('geometry', 'lidar_in_radar_frame', 'fraction_patches_ge2_labels', lidar_multi_a,
         '1 if a patch uses at least two labels (distinct GT boxes, or a GT box and background)')
    if radar_fg_a.size:
        take('geometry', 'radar_current', 'fg_voxel_fraction_if_touches_box', radar_fg_a,
             'one sample = one patch that touches a GT box; value = foreground voxels / real voxels')
    if lidar_fg_a.size:
        take('geometry', 'lidar_in_radar_frame', 'fg_voxel_fraction_if_touches_box', lidar_fg_a,
             'one sample = one patch that touches a GT box; value = foreground voxels / real voxels')
    take('attention', 'radar_xcout', 'norm_entropy_excluding_padded_patch', ent_kept_a,
         'one sample = one head of one query; the last patch is dropped when it contains padding')

    for (sensor, scale), values in ppv.items():
        take('points_per_voxel', sensor, f'mean_count_at_{scale}_voxel', values,
             'one sample = one non-empty voxel; radar is tem_idx == 0; lidar is in the radar frame')

    def mean_of(sensor, scale):
        return summary_stats[('points_per_voxel', sensor, f'mean_count_at_{scale}_voxel')]['mean']

    voxel_of = {'input': input_voxel, 'xcout': xcout_voxel}
    ppv_report = {}
    for sensor in ('lidar', 'radar'):
        for scale in ('input', 'xcout'):
            counts = ppv[(sensor, scale)]
            n_vox = int(counts.size)
            total_pts = int(counts.sum()) if n_vox else 0
            mean_pts = float(counts.mean()) if n_vox else float('nan')
            ppv_report[(sensor, scale)] = (n_vox, total_pts, mean_pts)
            print(
                f'ppv {sensor} {scale} voxel_xyz='
                f'({voxel_of[scale][0]:.6g}, {voxel_of[scale][1]:.6g}, {voxel_of[scale][2]:.6g}) '
                f'non_empty={n_vox} total_points={total_pts} mean={mean_pts:.6f}',
                flush=True,
            )
        in_counts = ppv[(sensor, 'input')]
        out_counts = ppv[(sensor, 'xcout')]
        if in_counts.shape == out_counts.shape and np.array_equal(in_counts, out_counts):
            raise RuntimeError(f'{sensor} input and xcout voxelizations produced the same occupancy counts')

    ratio_input = mean_of('lidar', 'input') / mean_of('radar', 'input')
    ratio_xcout = mean_of('lidar', 'xcout') / mean_of('radar', 'xcout')
    for name, value, note in (
        ('lidar_over_radar_at_input_voxel', ratio_input, 'ratio of mean points per non-empty voxel'),
        ('lidar_over_radar_at_xcout_voxel', ratio_xcout, 'ratio of mean points per non-empty voxel'),
    ):
        rows.append({
            'section': 'points_per_voxel',
            'sensor': 'lidar_over_radar',
            'quantity': name,
            'stat': 'ratio_of_means',
            'value': value,
            'n_frames': n_frames,
            'n_items': n_frames,
            'note': note,
        })

    dist_bins = hist_bins_from_data(np.concatenate([radar_dist_a, hook_dist_a]), lidar_dist_a)
    area_bins = hist_bins_from_data(np.concatenate([radar_area_a, hook_area_a]), lidar_area_a)
    ent_bins = np.linspace(0.0, 1.0, 41)
    append_hist(rows, 'geometry', 'radar_current', 'max_pairwise_distance_m', radar_dist_a, n_frames, '', dist_bins)
    append_hist(rows, 'geometry', 'lidar_in_radar_frame', 'max_pairwise_distance_m', lidar_dist_a, n_frames, '', dist_bins)
    append_hist(rows, 'hooked', 'radar_xcout', 'max_pairwise_distance_m', hook_dist_a, n_frames, '', dist_bins)
    append_hist(rows, 'geometry', 'radar_current', 'bev_cell_area_m2', radar_area_a, n_frames, '', area_bins)
    append_hist(rows, 'geometry', 'lidar_in_radar_frame', 'bev_cell_area_m2', lidar_area_a, n_frames, '', area_bins)
    append_hist(rows, 'hooked', 'radar_xcout', 'bev_cell_area_m2', hook_area_a, n_frames, '', area_bins)
    append_hist(rows, 'attention', 'radar_xcout', 'norm_entropy_excluding_padded_patch', ent_kept_a, n_frames, '', ent_bins)
    for (sensor, scale), values in ppv.items():
        cap = max(int(np.percentile(values, 99)) if values.size else 1, 1)
        bins = integer_hist_bins(values, cap)
        append_hist(
            rows, 'points_per_voxel', sensor, f'count_at_{scale}_voxel',
            np.minimum(values, cap), n_frames,
            f'counts above {cap} (empirical p99) are clipped into the last bin for the histogram only',
            bins,
        )

    dup_fraction = dup_tokens / hook_tokens if hook_tokens else float('nan')
    rows.append({
        'section': 'hooked', 'sensor': 'radar_xcout', 'quantity': 'duplicate_coord_fraction',
        'stat': 'fraction', 'value': dup_fraction, 'n_frames': n_frames,
        'n_items': hook_tokens,
        'note': 'tokens whose (z, y, x) is shared with another hooked token, over all hooked tokens',
    })
    meta = {
        'n_frames': n_frames,
        'frame_selection': (
            f'{n_frames} frames from numpy.linspace(0, len(dataset)-1, {n_frames}).round().astype(int)'
        ),
        'batch_size': BATCH_SIZE,
        'ckpt': str(ckpt_path),
        'cfg': str(cfg_path),
        'calls_min': int(min(call_counts)),
        'calls_max': int(max(call_counts)),
        'input_voxel_size': [float(v) for v in input_voxel],
        'index_stride_xy': list(index_stride_xy),
        'xcout_voxel_size': [float(v) for v in xcout_voxel],
        'spatial_shape': list(spatial_shape),
        'orders': list(ptv3.order),
        'shuffle_orders': bool(ptv3.shuffle_orders),
        'patch_size_max': patch_size_max,
        'radar_temporal_slots': slots,
        'lidar_transform': LIDAR_TO_RADAR_TRANSFORM,
        'dup_tokens': dup_tokens,
        'hook_tokens': hook_tokens,
        'dup_fraction': float(dup_fraction),
        'ratio_input': float(ratio_input),
        'ratio_xcout': float(ratio_xcout),
        'cmp_lidar_dist': summary_stats[('geometry', 'lidar_in_radar_frame', 'max_pairwise_distance_m')]['p50'],
        'cmp_radar_dist': summary_stats[('geometry', 'radar_current', 'max_pairwise_distance_m')]['p50'],
        'cmp_lidar_area': summary_stats[('geometry', 'lidar_in_radar_frame', 'bev_cell_area_m2')]['p50'],
        'cmp_radar_area': summary_stats[('geometry', 'radar_current', 'bev_cell_area_m2')]['p50'],
        'cmp_lidar_multi': float(lidar_multi_a.mean()) if lidar_multi_a.size else float('nan'),
        'cmp_radar_multi': float(radar_multi_a.mean()) if radar_multi_a.size else float('nan'),
        'cmp_hook_multi': float(hook_multi_a.mean()) if hook_multi_a.size else float('nan'),
        'cmp_lidar_fg': float(lidar_fg_a.mean()) if lidar_fg_a.size else float('nan'),
        'cmp_radar_fg': float(radar_fg_a.mean()) if radar_fg_a.size else float('nan'),
        'cmp_hook_fg': float(hook_fg_a.mean()) if hook_fg_a.size else float('nan'),
    }
    for key, value in (
        ('n_frames', n_frames),
        ('xcout_calls_per_frame_min', meta['calls_min']),
        ('xcout_calls_per_frame_max', meta['calls_max']),
    ):
        rows.append({
            'section': 'meta', 'sensor': '', 'quantity': key, 'stat': 'value',
            'value': value, 'n_frames': n_frames, 'n_items': n_frames, 'note': '',
        })

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_overlay_hist(
        FIG_DIR / 'fig4_patch_maxdist.png',
        [('radar geometry', radar_dist_a), ('lidar in radar frame', lidar_dist_a), ('hooked xcout', hook_dist_a)],
        dist_bins, 'max pairwise distance (m)', f'PTv3 patch extent, {n_frames} frames',
    )
    save_overlay_hist(
        FIG_DIR / 'fig4_patch_bev_area.png',
        [('radar geometry', radar_area_a), ('lidar in radar frame', lidar_area_a), ('hooked xcout', hook_area_a)],
        area_bins, 'BEV cell area (m^2)', f'PTv3 patch BEV area, {n_frames} frames',
    )
    save_bar_hist(
        FIG_DIR / 'fig4_patch_labels.png',
        [('radar geometry', radar_labels_a), ('lidar in radar frame', lidar_labels_a), ('hooked xcout', hook_labels_a)],
        'distinct GT boxes + background', f'Labels per patch, {n_frames} frames',
    )
    save_overlay_hist(
        FIG_DIR / 'fig4_attn_entropy.png',
        [('excluding padded tail patch', ent_kept_a)],
        ent_bins, 'normalized attention entropy', f'xcout softmax, {n_frames} frames',
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=False)
    for ax, scale, title in (
        (axes[0], 'input', 'input voxel'),
        (axes[1], 'xcout', 'xcout voxel'),
    ):
        for sensor in ('radar', 'lidar'):
            vals = ppv[(sensor, scale)]
            cap = max(int(np.percentile(vals, 99)) if vals.size else 1, 1)
            clipped = np.minimum(vals, cap)
            hi = int(clipped.max()) if clipped.size else 1
            counts = np.bincount(clipped.astype(np.int64), minlength=hi + 1)
            ax.step(np.arange(hi + 1), counts / max(counts.sum(), 1), where='mid', label=sensor)
        ax.set_title(f'{title}, {n_frames} frames')
        ax.set_xlabel('points in non-empty voxel')
        ax.set_ylabel('fraction')
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_points_per_voxel.png', dpi=140)
    plt.close(fig)

    write_csv(CSV_PATH, rows)
    write_summary(MD_PATH, meta, summary_stats)

    print()
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote {MD_PATH}')
    print(f'Wrote plots under {FIG_DIR}')
    print(f'Wrote {FRAMES_PATH}')
    print()
    print(
        f'On {n_frames} VoD val frames evenly spaced over the val set (batch size 1), pure-geometry patches at the '
        f'xcout voxel size have median max distance '
        f'{meta["cmp_lidar_dist"]:.3f} m (LiDAR, in the radar frame) and '
        f'{meta["cmp_radar_dist"]:.3f} m (current-frame radar). '
        f'Median BEV cell area is {meta["cmp_lidar_area"]:.3f} m^2 versus {meta["cmp_radar_area"]:.3f} m^2. '
        f'Fraction of patches with ≥2 labels is {meta["cmp_lidar_multi"]:.3f} (LiDAR), '
        f'{meta["cmp_radar_multi"]:.3f} (radar geometry), {meta["cmp_hook_multi"]:.3f} (hooked tokens). '
        f'Among patches touching a GT box, the mean foreground-voxel fraction is '
        f'{meta["cmp_lidar_fg"]:.3f} (LiDAR), {meta["cmp_radar_fg"]:.3f} (radar geometry), '
        f'{meta["cmp_hook_fg"]:.3f} (hooked tokens). '
        f'Hooked xcout tokens that share a coordinate: {dup_tokens} / {hook_tokens} ({dup_fraction:.4f}). '
        f'Median normalized entropy, with the padded tail patch removed, is '
        f'{summary_stats[("attention", "radar_xcout", "norm_entropy_excluding_padded_patch")]["p50"]:.3f}. '
        f'Mean points per non-empty voxel, LiDAR divided by current-frame radar, is '
        f'{ratio_input:.3f} at the input voxel and {ratio_xcout:.3f} at the xcout voxel.'
    )


if __name__ == '__main__':
    main()
