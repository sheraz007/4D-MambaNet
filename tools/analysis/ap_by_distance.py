#!/usr/bin/env python3
"""VoD val 3D AP by range bin for the best epoch of each seed.

PROTOCOL
--------
Predictions are the saved VoD val result.pkl of the best epoch of each seed.
Best epoch is the epoch with the highest entire-area 3D mAP in
docs/rev1/per_epoch_vod.csv (the value printed at training time as the mean of
Car / Pedestrian / Cyclist 3D AP). Seeds and run folders:

  M3-Net Repro   docs model M3-Net      output/uni_ptv3_xcout_seed_sweep/seed_{0,1,2,3}
  MambaNet-B     docs model MambaNet-B  output/mamba_only_s{0,1,2,3}_gpu1
  4D-MambaNet    docs model Full        output/mamba_rcs_s0 and output/mamba_rcs_s{1,2,3}_gpu1

No checkpoint is loaded and no GPU is used. CUDA_VISIBLE_DEVICES is cleared
before numeric libraries are imported.

AP is the official VoD 11-point 3D AP from
pcdet/datasets/kitti/vod_evaluation/kitti_official_evaluate.py: eval_class
with metric=2 (3D), one difficulty level, and the second overlap row
(Car 0.5, Pedestrian 0.25, Cyclist 0.25), reduced by get_m_ap. That is the
number training logs as Car_3d_all / Pedestrian_3d_all / Cyclist_3d_all.
Image and BEV AP are not computed. Frames are paired by frame_id with
vod_infos_val_temporal.pkl (1296 val frames).

Region filter: the driving-corridor path (clean_data with roi_clean=True,
custom_method=3). That function ignores a current-class box when its camera
location fails the region test, after the same class, minimum-height (40 px),
occlusion, neighboring-class (Van, Person_sitting), and DontCare rules.
Camera location is x right, y down, z forward; the corridor test is
x < -4 or x > 4 or z > 25.

The only change is that region test. A box is inside a bin when its forward
distance z = location[:, 2] falls in the bin, and outside otherwise. Lateral
position is not gated. Bins are half-open on the right except the last bin,
which includes the point-cloud forward limit:

  [0, 10), [10, 20), [20, 30), [30, 51.2]

z < 0 and z > 51.2 are outside every bin. Inside and outside are applied by
moving a copy of location to a point the official corridor test accepts
(0, 0, 0) or rejects (x=100), then calling the unmodified clean_data. The
original annos are not modified; 3D IoU still uses the real boxes. An ignored
box is neither a detection false positive nor a ground-truth false negative.

n_gt is the official valid-GT count for that class and bin (clean_data
num_valid_gt summed over val), i.e. the recall denominator. It does not depend
on the detector. ap_std is the sample standard deviation over the 4 seeds
(ddof=1).

Run from myopenpcdet/:
  CUDA_VISIBLE_DEVICES= nice -n 15 python tools/rev1/eval_by_distance.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pickle
import sys
import types
from pathlib import Path

# Training may be using the GPU. Do not touch it.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / 'docs' / 'rev1' / 'ap_by_distance.csv'
PER_EPOCH_CSV = REPO / 'docs' / 'rev1' / 'per_epoch_vod.csv'
GT_INFO = Path(__import__('os').environ.get('VOD_VAL_INFO', str(Path(__file__).resolve().parents[2] / 'data/vod/radar_5frames/vod_infos_val_temporal.pkl')))
VOD_EVAL_DIR = REPO / 'pcdet' / 'datasets' / 'kitti' / 'vod_evaluation'

# Half-open except the last edge, which is the forward limit of POINT_CLOUD_RANGE.
BINS = (
    ('0-10', 0.0, 10.0, False),
    ('10-20', 10.0, 20.0, False),
    ('20-30', 20.0, 30.0, False),
    ('30-51.2', 30.0, 51.2, True),
)
CLASSES = ('Car', 'Pedestrian', 'Cyclist')

# (display name, per_epoch model name, seed -> output folder relative to output/)
MODELS = (
    ('M3-Net Repro', 'M3-Net', {
        0: 'uni_ptv3_xcout_seed_sweep/seed_0',
        1: 'uni_ptv3_xcout_seed_sweep/seed_1',
        2: 'uni_ptv3_xcout_seed_sweep/seed_2',
        3: 'uni_ptv3_xcout_seed_sweep/seed_3',
    }),
    ('MambaNet-B', 'MambaNet-B', {
        0: 'mamba_only_s0_gpu1',
        1: 'mamba_only_s1_gpu1',
        2: 'mamba_only_s2_gpu1',
        3: 'mamba_only_s3_gpu1',
    }),
    ('4D-MambaNet', 'Full', {
        0: 'mamba_rcs_s0',
        1: 'mamba_rcs_s1_gpu1',
        2: 'mamba_rcs_s2_gpu1',
        3: 'mamba_rcs_s3_gpu1',
    }),
)

PROTOCOL = (
    'VoD val 3D AP by camera-forward range of the box center (location z, meters). '
    'Best epoch = argmax entire-area 3D mAP in docs/rev1/per_epoch_vod.csv. '
    'AP = official VoD 11-point 3D AP (kitti_official_evaluate.get_m_ap, metric=2, '
    'overlap row 1: Car 0.5, Pedestrian 0.25, Cyclist 0.25). '
    'Region filter = unmodified clean_data(roi_clean=True) driving-corridor path; '
    'the only change is the region: inside iff z is in the bin '
    '[0,10), [10,20), [20,30), [30,51.2] (z<0 and z>51.2 outside all bins). '
    'Inside/outside is encoded by relocating a copy of location to a point the '
    'corridor test accepts or rejects; IoU uses the original boxes. '
    'n_gt = sum of clean_data num_valid_gt (recall denominator; class, height>=40px, '
    'occlusion, and bin). ap_std = sample std over 4 seeds (ddof=1). CPU only.'
)


def load_vod_eval():
    """Load the devkit module without importing pcdet (that pulls in CUDA ops)."""
    pkg_name = 'vod_eval_pkg_rev1'
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(VOD_EVAL_DIR)]
    sys.modules[pkg_name] = pkg
    for name in ('rotate_iou_cpu', 'kitti_official_evaluate'):
        path = VOD_EVAL_DIR / f'{name}.py'
        spec = importlib.util.spec_from_file_location(f'{pkg_name}.{name}', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f'{pkg_name}.kitti_official_evaluate']


def best_epochs():
    """seed -> epoch and the logged corridor APs, keyed by per_epoch model name."""
    best = {}
    corridor = {}
    with PER_EPOCH_CSV.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['model'], int(row['seed']))
            score = float(row['entire_mAP'])
            epoch = int(row['epoch'])
            if key not in best or score > best[key][0]:
                best[key] = (score, epoch)
                corridor[key] = (
                    float(row['corr_car']),
                    float(row['corr_ped']),
                    float(row['corr_cyc']),
                )
    return best, corridor


def result_path(folder: str, epoch: int) -> Path:
    return (
        REPO / 'output' / folder / 'eval' / 'eval_with_train'
        / f'epoch_{epoch}' / 'val' / 'result.pkl'
    )


def resolve_runs():
    best, corridor = best_epochs()
    runs = []
    missing = []
    for display, csv_name, folders in MODELS:
        for seed in range(4):
            score, epoch = best[(csv_name, seed)]
            path = result_path(folders[seed], epoch)
            rec = {
                'model': display,
                'csv_name': csv_name,
                'seed': seed,
                'epoch': epoch,
                'entire_map': score,
                'folder': folders[seed],
                'pkl': path,
                'corridor_logged': corridor[(csv_name, seed)],
            }
            runs.append(rec)
            if not path.is_file():
                missing.append(rec)
    return runs, missing


def frame_id(info) -> str:
    return str(info['image']['image_idx'])


def load_gt():
    with GT_INFO.open('rb') as handle:
        infos = pickle.load(handle)
    by_id = {}
    for info in infos:
        by_id[frame_id(info)] = info['annos']
    return by_id


def load_det(path: Path, gt_by_id: dict):
    with path.open('rb') as handle:
        dets = pickle.load(handle)
    gt_annos, dt_annos = [], []
    for det in dets:
        fid = str(det['frame_id'])
        if fid not in gt_by_id:
            raise KeyError(f'{path} frame_id {fid} is not in the val infos')
        gt_annos.append(gt_by_id[fid])
        dt_annos.append(det)
    if len(dt_annos) != len(gt_by_id):
        raise RuntimeError(
            f'{path} has {len(dt_annos)} frames, val infos have {len(gt_by_id)}'
        )
    return gt_annos, dt_annos


def min_overlaps_vod():
    """Same overlap tensor as get_official_eval_result, classes Car/Ped/Cyc."""
    overlap_0_7 = np.array([
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    ])
    overlap_0_5 = np.array([
        [0.7, 0.50, 0.50, 0.7, 0.50, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    ])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    return min_overlaps[:, :, [0, 1, 2]]


def in_bin(z, lo, hi, hi_inclusive):
    z = np.asarray(z, dtype=np.float64)
    if hi_inclusive:
        return (z >= lo) & (z <= hi)
    return (z >= lo) & (z < hi)


def relocate_location(anno, inside):
    """Copy location so the official corridor test passes iff `inside` is true."""
    loc = np.asarray(anno['location'], dtype=np.float64)
    if loc.size == 0:
        new_loc = np.zeros((0, 3), dtype=np.float64)
    else:
        new_loc = np.array(loc, copy=True, dtype=np.float64)
        keep = np.asarray(inside, dtype=bool)
        new_loc[keep, 0] = 0.0
        new_loc[keep, 2] = 0.0
        new_loc[~keep, 0] = 100.0
    out = dict(anno)
    out['location'] = new_loc
    return out


def make_range_clean(original, lo, hi, hi_inclusive):
    def clean_data(gt_anno, dt_anno, current_class, difficulty, roi_clean=False):
        if not roi_clean:
            return original(gt_anno, dt_anno, current_class, difficulty, roi_clean=False)
        gt_mod = relocate_location(gt_anno, in_bin(np.asarray(gt_anno['location'])[:, 2] if len(gt_anno['location']) else [], lo, hi, hi_inclusive))
        if len(dt_anno['location']) == 0:
            dt_inside = []
        else:
            dt_inside = in_bin(np.asarray(dt_anno['location'])[:, 2], lo, hi, hi_inclusive)
        dt_mod = relocate_location(dt_anno, dt_inside)
        return original(gt_mod, dt_mod, current_class, difficulty, roi_clean=True)

    return clean_data


def ap_3d(vod_eval, gt_annos, dt_annos, overlaps):
    """11-point 3D AP per class. Uses whatever clean_data is installed."""
    ret = vod_eval.eval_class(
        gt_annos, dt_annos,
        current_classes=[0, 1, 2],
        difficulties=[0],
        metric=2,
        min_overlaps=overlaps,
        compute_aos=False,
        num_parts=50,
        custom_method=3,
    )
    map_3d = vod_eval.get_m_ap(ret['precision'])
    return [float(map_3d[j, 0, 1]) for j in range(3)]


def count_gt(vod_eval, gt_annos, clean_fn):
    counts = []
    empty_dt = {
        'name': np.array([]),
        'bbox': np.zeros((0, 4)),
        'location': np.zeros((0, 3)),
    }
    for cls_idx in range(3):
        total = 0
        for gt in gt_annos:
            num_valid, _, _, _ = clean_fn(gt, empty_dt, cls_idx, 0, roi_clean=True)
            total += int(num_valid)
        counts.append(total)
    return counts


def check_corridor(vod_eval, run, gt_annos, dt_annos, overlaps):
    """Official corridor path must reproduce the logged corridor AP."""
    vod_eval.clean_data = vod_eval.clean_data  # original
    got = ap_3d(vod_eval, gt_annos, dt_annos, overlaps)
    logged = run['corridor_logged']
    deltas = [abs(a - b) for a, b in zip(got, logged)]
    print(
        f"  corridor check {run['model']} seed {run['seed']} ep {run['epoch']}: "
        f"got {got[0]:.4f}/{got[1]:.4f}/{got[2]:.4f} "
        f"log {logged[0]:.4f}/{logged[1]:.4f}/{logged[2]:.4f} "
        f"max|d|={max(deltas):.6f}",
        flush=True,
    )
    if max(deltas) > 1e-3:
        raise RuntimeError(
            'Corridor AP does not match the training log. '
            'The range-bin filter is not running the official devkit path.'
        )


def main():
    runs, missing = resolve_runs()
    print('Best-epoch result.pkl files:', flush=True)
    for run in runs:
        flag = 'MISSING' if run in missing else 'OK'
        print(
            f"  {flag:7} {run['model']} seed {run['seed']} epoch {run['epoch']} "
            f"entire_mAP {run['entire_map']:.4f}  {run['pkl']}",
            flush=True,
        )
    if missing:
        print(
            f'STOP: {len(missing)} result.pkl file(s) missing. '
            'Re-evaluation needs the GPU and is not run here.',
            flush=True,
        )
        sys.exit(1)

    print('Loading VoD devkit evaluator (CPU) and val infos...', flush=True)
    vod_eval = load_vod_eval()
    original_clean = vod_eval.clean_data
    gt_by_id = load_gt()
    overlaps = min_overlaps_vod()

    # One official-corridor reproduction before trusting the relocated region.
    first = runs[0]
    gt0, dt0 = load_det(first['pkl'], gt_by_id)
    vod_eval.clean_data = original_clean
    check_corridor(vod_eval, first, gt0, dt0, overlaps)

    per_seed = []  # model, class, bin, seed, ap
    n_gt = {}
    for run in runs:
        print(
            f"{run['model']} seed {run['seed']} epoch {run['epoch']}",
            flush=True,
        )
        gt_annos, dt_annos = load_det(run['pkl'], gt_by_id)
        for label, lo, hi, hi_inclusive in BINS:
            vod_eval.clean_data = make_range_clean(original_clean, lo, hi, hi_inclusive)
            aps = ap_3d(vod_eval, gt_annos, dt_annos, overlaps)
            if run['seed'] == 0 and run['model'] == MODELS[0][0]:
                n_gt[label] = count_gt(vod_eval, gt_annos, vod_eval.clean_data)
            for cls_name, ap in zip(CLASSES, aps):
                per_seed.append({
                    'model': run['model'],
                    'class': cls_name,
                    'bin': label,
                    'seed': run['seed'],
                    'epoch': run['epoch'],
                    'ap': ap,
                })
                print(f'  {label:8} {cls_name:12} {ap:.4f}', flush=True)
        vod_eval.clean_data = original_clean

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as handle:
        handle.write('# ' + PROTOCOL + '\n')
        handle.write(
            '# result.pkl: ' + ' | '.join(
                f"{r['model']} s{r['seed']} ep{r['epoch']} {r['pkl'].relative_to(REPO)}"
                for r in runs
            ) + '\n'
        )
        writer = csv.DictWriter(handle, fieldnames=[
            'model', 'class', 'bin_m', 'ap_mean_pm_std', 'ap_mean', 'ap_std',
            'n_seeds', 'n_gt',
            'ap_seed0', 'ap_seed1', 'ap_seed2', 'ap_seed3',
            'epoch_seed0', 'epoch_seed1', 'epoch_seed2', 'epoch_seed3',
        ])
        writer.writeheader()
        for display, _, _ in MODELS:
            for cls_i, cls_name in enumerate(CLASSES):
                for label, _, _, _ in BINS:
                    rows = [
                        r for r in per_seed
                        if r['model'] == display and r['class'] == cls_name and r['bin'] == label
                    ]
                    rows.sort(key=lambda r: r['seed'])
                    vals = np.array([r['ap'] for r in rows], dtype=np.float64)
                    mean = float(vals.mean())
                    std = float(vals.std(ddof=1))
                    epochs = {r['seed']: r['epoch'] for r in rows}
                    aps = {r['seed']: r['ap'] for r in rows}
                    writer.writerow({
                        'model': display,
                        'class': cls_name,
                        'bin_m': label,
                        'ap_mean_pm_std': f'{mean:.2f} ± {std:.2f}',
                        'ap_mean': f'{mean:.6f}',
                        'ap_std': f'{std:.6f}',
                        'n_seeds': 4,
                        'n_gt': n_gt[label][cls_i],
                        'ap_seed0': f'{aps[0]:.6f}',
                        'ap_seed1': f'{aps[1]:.6f}',
                        'ap_seed2': f'{aps[2]:.6f}',
                        'ap_seed3': f'{aps[3]:.6f}',
                        'epoch_seed0': epochs[0],
                        'epoch_seed1': epochs[1],
                        'epoch_seed2': epochs[2],
                        'epoch_seed3': epochs[3],
                    })
    print(f'Wrote {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
