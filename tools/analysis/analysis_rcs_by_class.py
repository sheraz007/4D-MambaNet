#!/usr/bin/env python3
"""RCS of radar points inside VoD GT boxes, by class (train split).

Uses the same radar_5frames cloud as training (up to 5 sweeps already merged
in each velodyne bin), FOV crop, and point-cloud range mask. No augmentation
and no GPU. The loaded bin is the current frame (tem_idx == 0). The current
scan is the subset with time == 0; that subset is the primary result. The
full 5-scan cloud is reported second.

For every GT Car / Pedestrian / Cyclist box, collect RCS of radar points
inside the box, the object's range, and the aspect angle (heading minus the
radar line-of-sight angle). Also record per-object median and max RCS.

Outputs (under docs/rev1/):
  rcs_by_class_v2.csv
  figs/rcs_by_class_violin.{pdf,png}          current scan, point RCS
  figs/rcs_vs_range.{pdf,png}
  figs/rcs_vs_aspect.{pdf,png}
  figs/rcs_per_object_max_violin.{pdf,png}
  figs/*_5scan.{pdf,png}                      secondary 5-scan copies

Run from myopenpcdet/:
  CUDA_VISIBLE_DEVICES= nice -n 10 python tools/rev1/analysis_rcs_by_class.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Keep this process off the GPU (training may be running).
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.config_ext import apply_yaml_ext
from pcdet.datasets.kitti.vod_dataset import VODDataset
from pcdet.utils import box_utils, common_utils


CFG_REL = 'tools/cfgs/vod_models/last_model/uni_ptv3_mamba_only.yaml'
CLASS_NAMES = ('Car', 'Pedestrian', 'Cyclist')
CLASS_COLORS = {
    'Car': '#1f77b4',
    'Pedestrian': '#ff7f0e',
    'Cyclist': '#2ca02c',
}

OUT_DIR = ROOT / 'docs' / 'rev1'
FIG_DIR = OUT_DIR / 'figs'
CSV_PATH = OUT_DIR / 'rcs_by_class_v2.csv'
PAIRS = (('Car', 'Pedestrian'), ('Car', 'Cyclist'), ('Pedestrian', 'Cyclist'))

DPI = 300
RANGE_BIN_M = 5.0
ASPECT_BIN_DEG = 15.0
# Object-range strata for AUC. The last edge is inclusive.
RANGE_AUC_EDGES = (0.0, 15.0, 30.0, 51.2)
# In-box point-count strata: (label, lo, hi), inclusive.
COUNT_AUC_BINS = (('1', 1, 1), ('2', 2, 2), ('3-5', 3, 5), ('6-10', 6, 10))


def points_in_boxes_numpy(points_xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """CPU-only points-in-boxes; same geometry as roiaware_pool3d CPU kernel.

    Args:
        points_xyz: (N, 3)
        boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        (M, N) bool
    """
    if boxes.size == 0 or points_xyz.size == 0:
        return np.zeros((boxes.shape[0], points_xyz.shape[0]), dtype=bool)

    pts = np.asarray(points_xyz, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    margin = 1e-2

    # Broadcast: (M, 1) against (1, N).
    cx = boxes[:, 0:1]
    cy = boxes[:, 1:2]
    cz = boxes[:, 2:3]
    dx = boxes[:, 3:4]
    dy = boxes[:, 4:5]
    dz = boxes[:, 5:6]
    rz = boxes[:, 6:7]

    z_ok = np.abs(pts[None, :, 2] - cz) <= (dz * 0.5)
    shift_x = pts[None, :, 0] - cx
    shift_y = pts[None, :, 1] - cy
    cosa = np.cos(-rz)
    sina = np.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    in_xy = (np.abs(local_x) < dx * 0.5 + margin) & (np.abs(local_y) < dy * 0.5 + margin)
    return z_ok & in_xy


def wrap_angle_rad(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def object_range_and_aspect(box7: np.ndarray) -> tuple[float, float]:
    """Range (m) and signed aspect angle (rad) for one lidar-frame box.

    Aspect is the angle between the radar line of sight to the box center
    and the object's heading (yaw). Positive when heading is left of LOS.
    """
    x, y = float(box7[0]), float(box7[1])
    heading = float(box7[6])
    rng = float(np.hypot(x, y))
    los = float(np.arctan2(y, x))
    aspect = float(wrap_angle_rad(heading - los))
    return rng, aspect


def load_train_dataset(cfg_path: Path) -> VODDataset:
    os.chdir(ROOT)
    apply_yaml_ext()
    cfg = EasyDict()
    cfg.ROOT_DIR = ROOT
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(str(cfg_path), cfg)
    logger = common_utils.create_logger(log_file=None, rank=0)
    # training=True so DATA_SPLIT/INFO_PATH use the train key; we never call
    # __getitem__, so augmentations are not applied.
    dataset = VODDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=list(cfg.CLASS_NAMES),
        training=True,
        logger=logger,
    )
    return dataset


def _empty_bucket():
    return {
        'class': [],
        'rcs': [],
        'range': [],
        'aspect': [],
        'obj_class': [],
        'obj_median': [],
        'obj_max': [],
        'obj_range': [],
        'obj_aspect': [],
        'obj_n': [],
        'n_boxes': 0,
        'n_empty': 0,
    }


def _flush_bucket(bucket: dict) -> dict:
    return {
        'class': np.asarray(bucket['class']),
        'rcs': np.asarray(bucket['rcs'], dtype=np.float64),
        'range': np.asarray(bucket['range'], dtype=np.float64),
        'aspect': np.asarray(bucket['aspect'], dtype=np.float64),
        'obj_class': np.asarray(bucket['obj_class']),
        'obj_median': np.asarray(bucket['obj_median'], dtype=np.float64),
        'obj_max': np.asarray(bucket['obj_max'], dtype=np.float64),
        'obj_range': np.asarray(bucket['obj_range'], dtype=np.float64),
        'obj_aspect': np.asarray(bucket['obj_aspect'], dtype=np.float64),
        'obj_n': np.asarray(bucket['obj_n'], dtype=np.int64),
        'n_boxes': bucket['n_boxes'],
        'n_empty': bucket['n_empty'],
    }


def _add_box(bucket: dict, cls: str, rcs_vals: np.ndarray, rng: float, aspect: float) -> None:
    n_in = int(rcs_vals.size)
    if n_in == 0:
        bucket['n_empty'] += 1
        return
    bucket['n_boxes'] += 1
    bucket['class'].extend([cls] * n_in)
    bucket['rcs'].extend(rcs_vals.tolist())
    bucket['range'].extend([rng] * n_in)
    bucket['aspect'].extend([aspect] * n_in)
    bucket['obj_class'].append(cls)
    bucket['obj_median'].append(float(np.median(rcs_vals)))
    bucket['obj_max'].append(float(np.max(rcs_vals)))
    bucket['obj_range'].append(rng)
    bucket['obj_aspect'].append(aspect)
    bucket['obj_n'].append(n_in)


def collect_inbox_rcs(dataset: VODDataset, progress_every: int = 200):
    """Current-scan (time == 0) and full 5-scan in-box RCS on the train split.

    get_lidar loads the current frame only, so every point is tem_idx == 0.
    time == 0 is the latest sweep inside that 5-scan bin.
    """
    pc_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)
    fov_only = bool(dataset.dataset_cfg.FOV_POINTS_ONLY)
    current = _empty_bucket()
    five = _empty_bucket()
    n_frames = len(dataset.kitti_infos)

    for fi, info in enumerate(dataset.kitti_infos):
        sample_idx = info['point_cloud']['lidar_idx']
        points = dataset.get_lidar(sample_idx)
        if points.size == 0:
            continue

        calib = dataset.get_calib(sample_idx)
        img_shape = info['image']['image_shape']

        if fov_only:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = dataset.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        range_mask = common_utils.mask_points_by_range(points, pc_range)
        points = points[range_mask]
        if points.size == 0:
            continue

        if 'annos' not in info:
            continue
        annos = common_utils.drop_info_with_name(info['annos'], name='DontCare')
        keep = np.array([n in CLASS_NAMES for n in annos['name']], dtype=bool)
        if keep.size == 0 or not keep.any():
            continue

        names = annos['name'][keep]
        loc = annos['location'][keep]
        dims = annos['dimensions'][keep]
        rots = annos['rotation_y'][keep]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

        box_mask = box_utils.mask_boxes_outside_range_numpy(
            gt_boxes, pc_range, min_num_corners=1, use_center_to_filter=True
        )
        gt_boxes = gt_boxes[box_mask]
        names = names[box_mask]
        if gt_boxes.shape[0] == 0:
            continue

        inbox = points_in_boxes_numpy(points[:, 0:3], gt_boxes[:, :7])
        time0 = points[:, 6] == 0
        for bi in range(gt_boxes.shape[0]):
            pts_mask = inbox[bi]
            rng, aspect = object_range_and_aspect(gt_boxes[bi])
            cls = str(names[bi])
            rcs_all = points[pts_mask, 3].astype(np.float64)
            rcs_cur = points[pts_mask & time0, 3].astype(np.float64)
            _add_box(five, cls, rcs_all, rng, aspect)
            _add_box(current, cls, rcs_cur, rng, aspect)

        if progress_every > 0 and ((fi + 1) % progress_every == 0 or fi + 1 == n_frames):
            print(
                f'  frames {fi + 1}/{n_frames}  '
                f'current boxes={current["n_boxes"]} pts={len(current["rcs"])}  '
                f'5scan boxes={five["n_boxes"]} pts={len(five["rcs"])}',
                flush=True,
            )

    return _flush_bucket(current), _flush_bucket(five), n_frames


def distribution_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'median': np.nan,
            'iqr': np.nan,
            'q25': np.nan,
            'q75': np.nan,
        }
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'median': float(q50),
        'iqr': float(q75 - q25),
        'q25': float(q25),
        'q75': float(q75),
    }


def class_stats(class_arr: np.ndarray, values: np.ndarray) -> list[dict]:
    rows = []
    for name in CLASS_NAMES:
        row = {'class': name}
        row.update(distribution_stats(values[class_arr == name]))
        rows.append(row)
    return rows


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC with higher score favoring the positive class. Ties get half credit."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    n_pos = int(pos.size)
    n_neg = int(neg.size)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    n = scores.size
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    sum_pos = float(ranks[:n_pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pair_aucs(obj_class: np.ndarray, obj_max: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    if mask is None:
        mask = np.ones(obj_class.shape[0], dtype=bool)
    rows = []
    for pos_name, neg_name in PAIRS:
        pos = obj_max[mask & (obj_class == pos_name)]
        neg = obj_max[mask & (obj_class == neg_name)]
        rows.append({
            'pair': f'{pos_name}-{neg_name}',
            'pos': pos_name,
            'neg': neg_name,
            'auc': roc_auc(pos, neg),
            'n_pos': int(pos.size),
            'n_neg': int(neg.size),
        })
    return rows


def range_stratified_aucs(packed: dict) -> list[dict]:
    """AUC of per-object max RCS inside object-range bins."""
    ranges = packed['obj_range']
    rows = []
    edges = RANGE_AUC_EDGES
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (ranges >= lo) & (ranges < hi)
        else:
            mask = (ranges >= lo) & (ranges <= hi)
        for row in pair_aucs(packed['obj_class'], packed['obj_max'], mask):
            row['bin_lo'] = lo
            row['bin_hi'] = hi
            row['n_out'] = int((ranges > edges[-1]).sum())
            rows.append(row)
    return rows


def count_matched_aucs(packed: dict) -> list[dict]:
    """AUC of per-object max RCS inside in-box point-count bins."""
    npts = packed['obj_n']
    rows = []
    for label, lo, hi in COUNT_AUC_BINS:
        mask = (npts >= lo) & (npts <= hi)
        for row in pair_aucs(packed['obj_class'], packed['obj_max'], mask):
            row['stratum'] = label
            row['bin_lo'] = lo
            row['bin_hi'] = hi
            rows.append(row)
    return rows


CSV_FIELDS = [
    'section', 'variant', 'class', 'pair', 'bin_lo', 'bin_hi',
    'count', 'mean', 'median', 'iqr', 'q25', 'q75', 'auc', 'n_pos', 'n_neg',
]


def _blank_row() -> dict:
    return {k: '' for k in CSV_FIELDS}


def save_v2_csv(sections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sections:
            out = _blank_row()
            out.update(row)
            writer.writerow(out)


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_violin_box(class_arr: np.ndarray, rcs: np.ndarray, stem: Path, title: str, ylabel: str) -> None:
    data = [rcs[class_arr == name] for name in CLASS_NAMES]
    labels = list(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(labels) + 1),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.85,
    )
    for body, name in zip(parts['bodies'], labels):
        body.set_facecolor(CLASS_COLORS[name])
        body.set_alpha(0.35)
        body.set_edgecolor('none')
    bp = ax.boxplot(
        data,
        positions=np.arange(1, len(labels) + 1),
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.4),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
    )
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(CLASS_COLORS[name])
        patch.set_alpha(0.85)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, stem)


def binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(centers.shape, np.nan, dtype=np.float64)
    q25 = np.full(centers.shape, np.nan, dtype=np.float64)
    q75 = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            m = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        vals = y[m]
        counts[i] = vals.size
        if vals.size == 0:
            continue
        q25[i], med[i], q75[i] = np.percentile(vals, [25, 50, 75])
    return centers, med, q25, q75, counts


def plot_binned_by_class(
    class_arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    xlabel: str,
    title: str,
    stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in CLASS_NAMES:
        m = class_arr == name
        centers, med, q25, q75, counts = binned_median(x[m], y[m], edges)
        valid = counts >= 5
        if not np.any(valid):
            continue
        c = CLASS_COLORS[name]
        ax.plot(centers[valid], med[valid], '-o', color=c, markersize=4, label=name)
        ax.fill_between(
            centers[valid],
            q25[valid],
            q75[valid],
            color=c,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('RCS (dBsm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, stem)


def _fmt(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'nan'
    return f'{value:.{digits}f}'


def print_stat_table(title: str, rows: list[dict]) -> None:
    print()
    print(title)
    print(f'{"class":<12} {"count":>8} {"mean":>8} {"median":>8} {"IQR":>8} {"Q25":>8} {"Q75":>8}')
    for row in rows:
        print(
            f'{row["class"]:<12} {row["count"]:8d} '
            f'{_fmt(row["mean"]):>8} {_fmt(row["median"]):>8} {_fmt(row["iqr"]):>8} '
            f'{_fmt(row["q25"]):>8} {_fmt(row["q75"]):>8}'
        )


def bin_table_rows(packed: dict, edges: np.ndarray, x_key: str) -> list[dict]:
    rows = []
    x = packed[x_key]
    y = packed['rcs']
    labels = packed['class']
    for name in CLASS_NAMES:
        m = labels == name
        centers, med, q25, q75, counts = binned_median(x[m], y[m], edges)
        for i, center in enumerate(centers):
            rows.append({
                'class': name,
                'bin_lo': float(edges[i]),
                'bin_hi': float(edges[i + 1]),
                'center': float(center),
                'count': int(counts[i]),
                'median': float(med[i]),
                'q25': float(q25[i]),
                'q75': float(q75[i]),
            })
    return rows


def print_bin_table(title: str, rows: list[dict], unit: str) -> None:
    print()
    print(title)
    header = f'{"bin":<16}'
    for name in CLASS_NAMES:
        header += f' {name[:3]+" n":>8} {name[:3]+" med":>8}'
    print(header)
    # rows are class-major; regroup by bin
    by_bin = {}
    for row in rows:
        key = (row['bin_lo'], row['bin_hi'])
        by_bin.setdefault(key, {})[row['class']] = row
    for key in sorted(by_bin):
        lo, hi = key
        line = f'{lo:>6.0f}–{hi:<6.0f}{unit:<3}'
        for name in CLASS_NAMES:
            cell = by_bin[key].get(name)
            if cell is None or cell['count'] == 0:
                line += f' {0:8d} {"—":>8}'
            else:
                line += f' {cell["count"]:8d} {_fmt(cell["median"]):>8}'
        print(line)


def confirm_training_pipeline(dataset: VODDataset) -> None:
    """Fail if this run is not the radar_5frames training geometry and RCS channel."""
    expected_range = np.array([0, -25.6, -3, 51.2, 25.6, 2], dtype=np.float64)
    got_range = np.asarray(dataset.point_cloud_range, dtype=np.float64)
    if not np.allclose(got_range, expected_range):
        raise RuntimeError(f'POINT_CLOUD_RANGE {got_range.tolist()} != {expected_range.tolist()}')
    if not bool(dataset.dataset_cfg.FOV_POINTS_ONLY):
        raise RuntimeError('FOV_POINTS_ONLY is not True')
    if list(dataset.class_names) != list(CLASS_NAMES):
        raise RuntimeError(f'class_names {dataset.class_names} != {CLASS_NAMES}')
    used = list(dataset.point_feature_encoder.used_feature_list)
    src = list(dataset.point_feature_encoder.src_feature_list)
    if used != src or 'rcs' not in used:
        raise RuntimeError(f'RCS feature lists differ or missing rcs: used={used} src={src}')
    rcs_index = used.index('rcs')
    if rcs_index != 3:
        raise RuntimeError(
            f'rcs is src index {rcs_index}; collated model index would be {rcs_index + 1}, not 4'
        )
    processors = dataset.dataset_cfg.DATA_PROCESSOR
    range_proc = None
    for proc in processors:
        if proc.NAME == 'mask_points_and_boxes_outside_range':
            range_proc = proc
            break
    if range_proc is None or not bool(range_proc.REMOVE_OUTSIDE_BOXES):
        raise RuntimeError('mask_points_and_boxes_outside_range REMOVE_OUTSIDE_BOXES is not set')

    yaml_path = ROOT / 'tools/cfgs/dataset_configs/radar_5frames_tem_dataset.yaml'
    print('Training-pipeline check (radar_5frames_tem_dataset.yaml):')
    print(f'  file: {yaml_path}')
    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith((
            'DATASET:', 'DATA_PATH:', 'POINT_CLOUD_RANGE:', 'FOV_POINTS_ONLY:',
            'encoding_type:', 'used_feature_list:', 'src_feature_list:',
            '- NAME: mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES:',
        )):
            print(f'  | {line}')
    print(f'  loaded POINT_CLOUD_RANGE = {got_range.tolist()}')
    print(f'  loaded FOV_POINTS_ONLY = {bool(dataset.dataset_cfg.FOV_POINTS_ONLY)}')
    print(f'  loaded CLASS_NAMES = {list(dataset.class_names)}')
    print(
        '  RCS: raw bin column 3 (x,y,z,rcs,...) is copied unchanged by '
        'absolute_coordinates_encoding. Collate prepends the batch index, so the '
        'model reads points[:, 4]. get_lidar subtracts 0 and divides by 1, so the '
        'value stays raw dBsm. No RCS clip in the data processor.'
    )
    print(
        '  GT: Car/Pedestrian/Cyclist only, DontCare dropped, boxes whose centers '
        'fall outside POINT_CLOUD_RANGE removed (REMOVE_OUTSIDE_BOXES, center filter).'
    )
    print(
        '  Random flip/rotation/scale are not applied; they would move range and aspect.'
    )


def print_auc_table(title: str, rows: list[dict]) -> None:
    print()
    print(title)
    print('Score is per-object max RCS. AUC > 0.5 means the first class tends to be brighter.')
    print(f'{"pair":<22} {"AUC":>8} {"n_pos":>8} {"n_neg":>8}')
    for row in rows:
        print(f'{row["pair"]:<22} {_fmt(row["auc"], 3):>8} {row["n_pos"]:8d} {row["n_neg"]:8d}')


def print_stratified_auc(title: str, rows: list[dict], key_name: str) -> None:
    print()
    print(title)
    print(f'{"stratum":<16} {"pair":<22} {"AUC":>8} {"n_pos":>8} {"n_neg":>8} {"n":>8}')
    for row in rows:
        print(
            f'{row[key_name]:<16} {row["pair"]:<22} {_fmt(row["auc"], 3):>8} '
            f'{row["n_pos"]:8d} {row["n_neg"]:8d} {row["n_pos"] + row["n_neg"]:8d}'
        )


def variant_csv_rows(variant: str, point_rows, obj_med_rows, obj_max_rows, range_rows, aspect_rows, auc_rows):
    out = []
    for row in point_rows:
        out.append({
            'section': 'point_rcs', 'variant': variant, 'class': row['class'],
            'count': row['count'], 'mean': row['mean'], 'median': row['median'],
            'iqr': row['iqr'], 'q25': row['q25'], 'q75': row['q75'],
        })
    for row in obj_med_rows:
        out.append({
            'section': 'object_median_rcs', 'variant': variant, 'class': row['class'],
            'count': row['count'], 'mean': row['mean'], 'median': row['median'],
            'iqr': row['iqr'], 'q25': row['q25'], 'q75': row['q75'],
        })
    for row in obj_max_rows:
        out.append({
            'section': 'object_max_rcs', 'variant': variant, 'class': row['class'],
            'count': row['count'], 'mean': row['mean'], 'median': row['median'],
            'iqr': row['iqr'], 'q25': row['q25'], 'q75': row['q75'],
        })
    for row in range_rows:
        out.append({
            'section': 'bin_range', 'variant': variant, 'class': row['class'],
            'bin_lo': row['bin_lo'], 'bin_hi': row['bin_hi'],
            'count': row['count'], 'median': row['median'], 'q25': row['q25'], 'q75': row['q75'],
        })
    for row in aspect_rows:
        out.append({
            'section': 'bin_aspect', 'variant': variant, 'class': row['class'],
            'bin_lo': row['bin_lo'], 'bin_hi': row['bin_hi'],
            'count': row['count'], 'median': row['median'], 'q25': row['q25'], 'q75': row['q75'],
        })
    for row in auc_rows:
        out.append({
            'section': 'separability', 'variant': variant, 'pair': row['pair'],
            'auc': row['auc'], 'n_pos': row['n_pos'], 'n_neg': row['n_neg'],
            'count': row['n_pos'] + row['n_neg'],
        })
    return out


def control_csv_rows(variant: str, range_auc: list[dict], count_auc: list[dict]) -> list[dict]:
    out = []
    for row in range_auc:
        out.append({
            'section': 'auc_by_range', 'variant': variant, 'pair': row['pair'],
            'bin_lo': row['bin_lo'], 'bin_hi': row['bin_hi'],
            'auc': row['auc'], 'n_pos': row['n_pos'], 'n_neg': row['n_neg'],
            'count': row['n_pos'] + row['n_neg'],
        })
    for row in count_auc:
        out.append({
            'section': 'auc_by_npoints', 'variant': variant, 'class': row['stratum'],
            'pair': row['pair'], 'bin_lo': row['bin_lo'], 'bin_hi': row['bin_hi'],
            'auc': row['auc'], 'n_pos': row['n_pos'], 'n_neg': row['n_neg'],
            'count': row['n_pos'] + row['n_neg'],
        })
    return out


def emit_variant(packed: dict, variant: str, suffix: str, scan_label: str, range_edges, aspect_edges):
    point_rows = class_stats(packed['class'], packed['rcs'])
    obj_med_rows = class_stats(packed['obj_class'], packed['obj_median'])
    obj_max_rows = class_stats(packed['obj_class'], packed['obj_max'])
    aspect_deg = np.degrees(packed['aspect'])
    range_rows = bin_table_rows(
        {'class': packed['class'], 'rcs': packed['rcs'], 'x': packed['range']},
        range_edges,
        'x',
    )
    aspect_rows = bin_table_rows(
        {'class': packed['class'], 'rcs': packed['rcs'], 'x': aspect_deg},
        aspect_edges,
        'x',
    )
    auc_rows = pair_aucs(packed['obj_class'], packed['obj_max'])

    plot_violin_box(
        packed['class'], packed['rcs'],
        FIG_DIR / f'rcs_by_class_violin{suffix}',
        title=f'Point RCS inside GT boxes (VoD train, {scan_label})',
        ylabel='RCS (dBsm)',
    )
    plot_binned_by_class(
        packed['class'], packed['range'], packed['rcs'], range_edges,
        xlabel='Object range (m)',
        title=f'RCS vs range, {scan_label} (median per {RANGE_BIN_M:.0f} m bin; band = IQR)',
        stem=FIG_DIR / f'rcs_vs_range{suffix}',
    )
    plot_binned_by_class(
        packed['class'], aspect_deg, packed['rcs'], aspect_edges,
        xlabel='Aspect angle (deg; heading minus LOS)',
        title=f'RCS vs aspect, {scan_label} (median per {ASPECT_BIN_DEG:.0f}° bin; band = IQR)',
        stem=FIG_DIR / f'rcs_vs_aspect{suffix}',
    )
    plot_violin_box(
        packed['obj_class'], packed['obj_max'],
        FIG_DIR / f'rcs_per_object_max_violin{suffix}',
        title=f'Per-object max RCS (VoD train, {scan_label})',
        ylabel='Max RCS in box (dBsm)',
    )
    return point_rows, obj_med_rows, obj_max_rows, range_rows, aspect_rows, auc_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', default=CFG_REL, help='model yaml that pulls in the VoD 5-frame data config')
    parser.add_argument('--max_frames', type=int, default=None, help='optional cap for a quick smoke test')
    parser.add_argument('--progress_every', type=int, default=200)
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    print(f'Loading dataset from {cfg_path}', flush=True)
    dataset = load_train_dataset(cfg_path)
    if args.max_frames is not None:
        dataset.kitti_infos = dataset.kitti_infos[: int(args.max_frames)]
        print(f'Using first {len(dataset.kitti_infos)} frames only', flush=True)
    print(
        f'Train frames: {len(dataset.kitti_infos)}; '
        f'DATA_PATH={dataset.root_path}; '
        f'POINT_CLOUD_RANGE={dataset.point_cloud_range.tolist()}',
        flush=True,
    )
    confirm_training_pipeline(dataset)

    current, five, n_frames = collect_inbox_rcs(dataset, progress_every=args.progress_every)
    if current['rcs'].size == 0 and five['rcs'].size == 0:
        raise RuntimeError('No in-box radar points found; check data paths and split.')

    r_max = 0.0
    for packed in (current, five):
        if packed['range'].size:
            r_max = max(r_max, float(np.max(packed['range'])))
    r_max = max(float(np.ceil(r_max / RANGE_BIN_M) * RANGE_BIN_M), RANGE_BIN_M)
    range_edges = np.arange(0.0, r_max + RANGE_BIN_M * 0.5, RANGE_BIN_M)
    aspect_edges = np.arange(-180.0, 180.0 + ASPECT_BIN_DEG * 0.5, ASPECT_BIN_DEG)

    cur = emit_variant(current, 'current', '', 'current scan', range_edges, aspect_edges)
    sec = emit_variant(five, '5scan', '_5scan', '5-scan', range_edges, aspect_edges)
    cur_range_auc = range_stratified_aucs(current)
    cur_count_auc = count_matched_aucs(current)
    five_range_auc = range_stratified_aucs(five)
    five_count_auc = count_matched_aucs(five)
    for row in cur_range_auc:
        row['label'] = f'{row["bin_lo"]:.0f}–{row["bin_hi"]:g} m'
    for row in five_range_auc:
        row['label'] = f'{row["bin_lo"]:.0f}–{row["bin_hi"]:g} m'
    save_v2_csv(
        variant_csv_rows('current', *cur)
        + variant_csv_rows('5scan', *sec)
        + control_csv_rows('current', cur_range_auc, cur_count_auc)
        + control_csv_rows('5scan', five_range_auc, five_count_auc),
        CSV_PATH,
    )

    point_rows, obj_med_rows, obj_max_rows, range_rows, aspect_rows, auc_rows = cur
    five_point, five_med, five_max, five_range, five_aspect, five_auc = sec

    print()
    print(f'=== Primary: current scan (time == 0, current frame) — {n_frames} train frames ===')
    print(
        f'Boxes with >=1 point: {current["n_boxes"]}; empty boxes: {current["n_empty"]}; '
        f'in-box points: {current["rcs"].size}.'
    )
    print_stat_table('Point RCS', point_rows)
    print_stat_table('Per-object median RCS', obj_med_rows)
    print_stat_table('Per-object max RCS', obj_max_rows)
    print_bin_table(
        f'Binned median point RCS vs range ({RANGE_BIN_M:.0f} m bins)',
        range_rows, 'm',
    )
    print_bin_table(
        f'Binned median point RCS vs aspect ({ASPECT_BIN_DEG:.0f} deg bins)',
        aspect_rows, 'deg',
    )
    print_auc_table('Separability of per-object max RCS', auc_rows)
    n_far = int((current['obj_range'] > RANGE_AUC_EDGES[-1]).sum())
    print_stratified_auc(
        'Range-stratified AUC of per-object max RCS '
        f'(bins 0–15, 15–30, 30–51.2 m; {n_far} objects beyond 51.2 m excluded)',
        cur_range_auc, 'label',
    )
    n_hi = int((current['obj_n'] > 10).sum())
    print_stratified_auc(
        f'Point-count-matched AUC of per-object max RCS ({n_hi} objects with >10 points excluded)',
        cur_count_auc, 'stratum',
    )

    print()
    print('=== Secondary: all 5 scans in the current frame ===')
    print(
        f'Boxes with >=1 point: {five["n_boxes"]}; empty boxes: {five["n_empty"]}; '
        f'in-box points: {five["rcs"].size}.'
    )
    print_stat_table('Point RCS', five_point)
    print_stat_table('Per-object median RCS', five_med)
    print_stat_table('Per-object max RCS', five_max)
    print_auc_table('Separability of per-object max RCS', five_auc)
    n_far5 = int((five['obj_range'] > RANGE_AUC_EDGES[-1]).sum())
    print_stratified_auc(
        f'Range-stratified AUC of per-object max RCS ({n_far5} objects beyond 51.2 m excluded)',
        five_range_auc, 'label',
    )
    n_hi5 = int((five['obj_n'] > 10).sum())
    print_stratified_auc(
        f'Point-count-matched AUC of per-object max RCS ({n_hi5} objects with >10 points excluded)',
        five_count_auc, 'stratum',
    )

    print()
    print('=== Summary ===')
    bright = max(obj_max_rows, key=lambda r: r['median'] if r['count'] else -np.inf)
    weak = min(obj_max_rows, key=lambda r: r['median'] if r['count'] else np.inf)
    print(
        f'On the current scan, per-object max RCS is highest for {bright["class"]} '
        f'(median {_fmt(bright["median"])} dBsm) and lowest for {weak["class"]} '
        f'(median {_fmt(weak["median"])} dBsm).'
    )
    best = max(auc_rows, key=lambda r: r['auc'] if np.isfinite(r['auc']) else -1)
    print(
        f'The strongest current-scan separation by per-object max RCS is {best["pair"]} '
        f'(AUC {_fmt(best["auc"], 3)}). An AUC of 0.5 would mean the two classes overlap completely.'
    )
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote figures under {FIG_DIR}')


if __name__ == '__main__':
    main()
