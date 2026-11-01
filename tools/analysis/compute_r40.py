#!/usr/bin/env python3
"""R11 and R40 VoD 3D AP for the epochs used in the summary.

Same evaluator and overlap branch as the R11-vs-R40 audit: eval_class in
pcdet/datasets/kitti/vod_evaluation/kitti_official_evaluate.py, metric=2 (3D),
difficulty 0, one overlap row taken from the official second stack
(Car 0.5, Pedestrian 0.25, Cyclist 0.25). R11 is get_m_ap, R40 is
get_m_ap_r40. Entire area is custom_method=0. Driving corridor is
custom_method=3 (clean_data roi_clean: camera x in [-4, 4] and z <= 25).

Per run, only the best epoch and epochs 76-80 are evaluated. Best epoch is
the epoch with the highest entire-area 3D mAP in docs/rev1/per_epoch_vod.csv
(the training-log R11 mAP; ties keep the earliest epoch). That is six result
files when the best epoch is outside 76-80, and five when it is inside.

No checkpoint is loaded and torch is not imported. CUDA_VISIBLE_DEVICES is
cleared before numeric libraries load. Launch with nice -n 10; the process
also raises its niceness to 10 if it is lower.

Runs:
  M3-Net repro   output/uni_ptv3_xcout_seed_sweep/seed_{0,1,2,3}
  MambaNet-B     output/mamba_only_s{0,1,2,3}_gpu1
  full           output/mamba_rcs_s0 and output/mamba_rcs_s{1,2,3}_gpu1
  plus every finished VoD run under output/rev1/ (epoch 80 is in the
  per-epoch CSV and the needed result.pkl files exist; TJ4D runs are skipped).

Rows already stored in docs/rev1/per_epoch_vod_r40.csv are not recomputed.
A later invocation picks up rev1 runs that have since finished.

  CUDA_VISIBLE_DEVICES= nice -n 10 python tools/rev1/eval_r40.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pickle
import sys
import types
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    _nice = os.nice(0)
    if _nice < 10:
        os.nice(10 - _nice)
except OSError:
    pass

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / 'docs' / 'rev1' / 'per_epoch_vod_r40.csv'
PER_EPOCH_CSV = REPO / 'docs' / 'rev1' / 'per_epoch_vod.csv'
GT_INFO = Path(__import__('os').environ.get('VOD_VAL_INFO', str(Path(__file__).resolve().parents[2] / 'data/vod/radar_5frames/vod_infos_val_temporal.pkl')))
VOD_EVAL_DIR = REPO / 'pcdet' / 'datasets' / 'kitti' / 'vod_evaluation'
OUTPUT = REPO / 'output'

TAIL_EPOCHS = (76, 77, 78, 79, 80)
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
# Paper comparison. model names match docs/rev1/per_epoch_vod.csv.
SUBMITTED = (
    ('M3-Net', {
        0: 'uni_ptv3_xcout_seed_sweep/seed_0',
        1: 'uni_ptv3_xcout_seed_sweep/seed_1',
        2: 'uni_ptv3_xcout_seed_sweep/seed_2',
        3: 'uni_ptv3_xcout_seed_sweep/seed_3',
    }),
    ('MambaNet-B', {
        0: 'mamba_only_s0_gpu1',
        1: 'mamba_only_s1_gpu1',
        2: 'mamba_only_s2_gpu1',
        3: 'mamba_only_s3_gpu1',
    }),
    ('Full', {
        0: 'mamba_rcs_s0',
        1: 'mamba_rcs_s1_gpu1',
        2: 'mamba_rcs_s2_gpu1',
        3: 'mamba_rcs_s3_gpu1',
    }),
)
PAPER_LABEL = {
    'M3-Net': 'M3-Net repro',
    'MambaNet-B': 'MambaNet-B',
    'Full': 'full',
}
PAIRS = (
    ('B - repro', 'MambaNet-B', 'M3-Net'),
    ('full - repro', 'Full', 'M3-Net'),
    ('full - B', 'Full', 'MambaNet-B'),
)

FIELDS = [
    'run_id', 'model', 'seed', 'epoch',
    'r11_entire_car', 'r11_entire_ped', 'r11_entire_cyc', 'r11_entire_mAP',
    'r11_corr_car', 'r11_corr_ped', 'r11_corr_cyc', 'r11_corr_mAP',
    'r40_entire_car', 'r40_entire_ped', 'r40_entire_cyc', 'r40_entire_mAP',
    'r40_corr_car', 'r40_corr_ped', 'r40_corr_cyc', 'r40_corr_mAP',
]

# Audit reference: mamba_only_s0_gpu1 epoch 76 entire-area R11.
AUDIT_CHECK = {
    'run_id': 'mamba_only_s0_gpu1',
    'epoch': 76,
    'r11_entire_car': 42.83796401,
    'r11_entire_ped': 49.16654746,
    'r11_entire_cyc': 76.27319012,
    'r11_entire_mAP': 56.09256720,
}


def load_vod_eval():
    """Load the devkit without importing pcdet (that pulls in CUDA ops)."""
    pkg_name = 'vod_eval_pkg_r40'
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


def min_overlaps_audit():
    """Second official overlap row only, classes Car / Pedestrian / Cyclist.

    Shape [1, 3, 3] = [overlap, metric, class], matching the audit call
    (image / bev / 3d rows; eval uses metric=2).
    """
    overlap_0_5 = np.array([
        [0.7, 0.50, 0.50, 0.7, 0.50, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    ], dtype=np.float64)
    classes = [0, 1, 2]
    return overlap_0_5[:, classes][None, ...]


def result_path(run_id: str, epoch: int) -> Path:
    return (
        OUTPUT / run_id / 'eval' / 'eval_with_train'
        / f'epoch_{epoch}' / 'val' / 'result.pkl'
    )


def load_logged_curves():
    """(model, seed) -> {epoch: entire mAP} from the training-log CSV."""
    curves = {}
    with PER_EPOCH_CSV.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['model'], int(row['seed']))
            curves.setdefault(key, {})[int(row['epoch'])] = float(row['entire_mAP'])
    return curves


def best_epoch_of(curve):
    """Earliest epoch with the highest logged entire-area mAP."""
    return min(curve, key=lambda epoch: (-curve[epoch], epoch))


def logged_curve(curves, model, seed):
    """Match this run to a per_epoch_vod.csv row.

    Submitted runs use the model name directly. Rev1 rows are stored as
    '{experiment}_s{seed}' (seed column still set).
    """
    for key in ((model, seed), (f'{model}_s{seed}', seed), (f'{model}_s{seed}', 0)):
        if key in curves:
            return key, curves[key]
    return None, None


def epochs_needed(best_epoch):
    return sorted(set((best_epoch,) + TAIL_EPOCHS))


def missing_pkls(run_id, epochs):
    return [epoch for epoch in epochs if not result_path(run_id, epoch).is_file()]


def is_vod_run(run_dir: Path) -> bool:
    blobs = []
    for path in sorted(run_dir.glob('*.yaml')):
        blobs.append(path.read_text(errors='ignore'))
    text = '\n'.join(blobs).lower()
    if 'voddataset' in text or 'radar_5frames' in text or 'vod_models' in text:
        return True
    return False


def add_run(runs, curves, model, seed, run_id, required):
    key, curve = logged_curve(curves, model, seed)
    if curve is None or 80 not in curve:
        message = f'skip {run_id}: no finished curve (through epoch 80) in {PER_EPOCH_CSV.name}'
        if required:
            raise SystemExit(message)
        print(message, flush=True)
        return
    best = best_epoch_of(curve)
    epochs = epochs_needed(best)
    missing = missing_pkls(run_id, epochs)
    if missing:
        message = f'skip {run_id}: missing result.pkl for epochs {missing} (best {best})'
        if required:
            raise SystemExit(message)
        print(message, flush=True)
        return
    runs.append({
        'model': model,
        'seed': seed,
        'run_id': run_id,
        'epochs': epochs,
        'best_epoch': best,
        'logged_model': key[0],
    })


def discover_runs(curves):
    """Submitted VoD runs, then finished VoD runs under output/rev1/."""
    runs = []
    for model, folders in SUBMITTED:
        for seed, folder in folders.items():
            add_run(runs, curves, model, seed, folder, required=True)

    rev_root = OUTPUT / 'rev1'
    if rev_root.is_dir():
        for exp in sorted(p for p in rev_root.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in exp.iterdir() if p.is_dir()):
                if not seed_dir.name.startswith('seed'):
                    continue
                try:
                    seed = int(seed_dir.name.replace('seed', '').lstrip('_'))
                except ValueError:
                    print(f'skip {seed_dir}: seed name not an int', flush=True)
                    continue
                run_id = str(seed_dir.relative_to(OUTPUT))
                if not is_vod_run(seed_dir):
                    print(f'skip non-VoD {run_id}', flush=True)
                    continue
                add_run(runs, curves, exp.name, seed, run_id, required=False)
    return runs


def load_done():
    done = set()
    rows = []
    if not OUT_CSV.is_file():
        return done, rows
    with OUT_CSV.open(newline='') as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            done.add((row['run_id'], int(row['epoch'])))
    return done, rows


def load_gt():
    with GT_INFO.open('rb') as handle:
        infos = pickle.load(handle)
    gt = [info['annos'] for info in infos]
    gt_ids = [str(info['point_cloud']['lidar_idx']) for info in infos]
    return gt, gt_ids


def load_det(path: Path, gt_ids):
    with path.open('rb') as handle:
        det = pickle.load(handle)
    if len(det) != len(gt_ids):
        raise RuntimeError(f'{path} has {len(det)} frames, val infos have {len(gt_ids)}')
    det_ids = [str(d.get('frame_id', '')) for d in det]
    if det_ids != gt_ids:
        if set(det_ids) != set(gt_ids):
            raise RuntimeError(f'frame id set mismatch {path}')
        pos = {fid: i for i, fid in enumerate(det_ids)}
        det = [det[pos[fid]] for fid in gt_ids]
    return det


def ap_from_precision(vod_eval, precision):
    r11 = vod_eval.get_m_ap(precision)[:, 0, 0]
    r40 = vod_eval.get_m_ap_r40(precision)[:, 0, 0]
    return r11, r40


def eval_epoch(vod_eval, gt, det, overlaps):
    """Entire area and driving corridor, R11 and R40, 3D only."""
    out = {}
    for prefix, method in (('entire', 0), ('corr', 3)):
        ret = vod_eval.eval_class(
            gt, det, [0, 1, 2], [0],
            metric=2,
            min_overlaps=overlaps,
            compute_aos=False,
            num_parts=50,
            custom_method=method,
        )
        r11, r40 = ap_from_precision(vod_eval, ret['precision'])
        for kind, values in (('r11', r11), ('r40', r40)):
            for name, value in zip(('car', 'ped', 'cyc'), values):
                out[f'{kind}_{prefix}_{name}'] = float(value)
            out[f'{kind}_{prefix}_mAP'] = float(np.mean(values))
    return out


def check_audit(row):
    if row['run_id'] != AUDIT_CHECK['run_id'] or row['epoch'] != AUDIT_CHECK['epoch']:
        return
    for key in ('r11_entire_car', 'r11_entire_ped', 'r11_entire_cyc', 'r11_entire_mAP'):
        got = row[key]
        ref = AUDIT_CHECK[key]
        if abs(got - ref) > 1e-4:
            raise RuntimeError(
                f'audit mismatch {key}: got {got:.6f} expected {ref:.6f}. '
                'Overlap branch or eval_class settings differ from the R11 audit.'
            )
    print(
        f"audit check OK {row['run_id']} epoch {row['epoch']} "
        f"R11 entire mAP {row['r11_entire_mAP']:.4f}",
        flush=True,
    )


def append_row(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_CSV.is_file()
    with OUT_CSV.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _f(row, key):
    return float(row[key])


def rows_by_run(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['run_id'], []).append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda r: int(r['epoch']))
    return grouped


def series_for(model_rows, mode, key):
    """One value per seed.

    mode 'best': the epoch with the highest R11 entire mAP.
    mode 'ep80': epoch 80.
    mode 'last5': mean over epochs 76-80.
    """
    by_seed = {}
    for row in model_rows:
        by_seed.setdefault(int(row['seed']), []).append(row)
    values = {}
    epochs = {}
    for seed, seed_rows in sorted(by_seed.items()):
        if mode == 'best':
            chosen = max(seed_rows, key=lambda r: (_f(r, 'r11_entire_mAP'), -int(r['epoch'])))
            values[seed] = _f(chosen, key)
            epochs[seed] = int(chosen['epoch'])
        elif mode == 'ep80':
            hit = [r for r in seed_rows if int(r['epoch']) == 80]
            if len(hit) != 1:
                continue
            values[seed] = _f(hit[0], key)
            epochs[seed] = 80
        elif mode == 'last5':
            hit = [r for r in seed_rows if 76 <= int(r['epoch']) <= 80]
            if len(hit) != 5:
                continue
            values[seed] = float(np.mean([_f(r, key) for r in hit]))
            epochs[seed] = '76-80'
        else:
            raise ValueError(mode)
    return values, epochs


def mean_std(vals):
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    if arr.size == 1:
        return float(arr[0]), float('nan')
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_p(diff):
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.size
    if n < 2:
        return float('nan')
    std = float(diff.std(ddof=1))
    if std == 0.0:
        return float('nan')
    mean = float(diff.mean())
    t_stat = mean / (std / np.sqrt(n))
    try:
        from scipy.stats import t as student_t
        return float(2 * student_t.sf(abs(t_stat), n - 1))
    except Exception:
        # Regularized incomplete beta, two-sided Student t survival, df = n-1.
        from math import lgamma, exp, log
        df = n - 1
        x = df / (df + t_stat * t_stat)
        # continued fraction for the incomplete beta (Numerical Recipes).
        a = 0.5 * df
        b = 0.5
        # betai
        def betacf(aa, bb, xx):
            max_it = 200
            eps = 3e-14
            qab = aa + bb
            qap = aa + 1
            qam = aa - 1
            c = 1.0
            d = 1.0 - qab * xx / qap
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            h = d
            for m in range(1, max_it + 1):
                m2 = 2 * m
                aa_m = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
                d = 1.0 + aa_m * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa_m / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                h *= d * c
                aa_m = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
                d = 1.0 + aa_m * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa_m / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h
        if x == 0 or x == 1:
            bt = 0.0
        else:
            bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
        if x < (a + 1) / (a + b + 2):
            ib = bt * betacf(a, b, x) / a
        else:
            ib = 1 - bt * betacf(b, a, 1 - x) / b
        # two-sided: sf(|t|) = 0.5 * ibeta(df/2, 1/2, df/(df+t^2))
        return float(ib)


def fmt_pm(mean, std, n):
    if n == 0 or mean != mean:
        return 'n/a'
    if n == 1 or std != std:
        return f'{mean:.3f} (n=1)'
    return f'{mean:.3f} ± {std:.3f}'


METRIC_KEYS = (
    ('entire Car', 'entire_car'),
    ('entire Ped', 'entire_ped'),
    ('entire Cyc', 'entire_cyc'),
    ('entire mAP', 'entire_mAP'),
    ('corridor Car', 'corr_car'),
    ('corridor Ped', 'corr_ped'),
    ('corridor Cyc', 'corr_cyc'),
    ('corridor mAP', 'corr_mAP'),
)
MODES = (
    ('(a) best epoch by R11 entire mAP', 'best'),
    ('(b) epoch 80', 'ep80'),
    ('(c) mean of epochs 76-80', 'last5'),
)


def print_summary(rows):
    by_model = {}
    for row in rows:
        by_model.setdefault(row['model'], []).append(row)

    paper = [name for name, _ in SUBMITTED if name in by_model]
    others = [name for name in sorted(by_model) if name not in PAPER_LABEL]

    def label(name):
        return PAPER_LABEL.get(name, name)

    for ap_name in ('r11', 'r40'):
        print(f'\n======== {ap_name.upper()} ========', flush=True)
        for mode_title, mode in MODES:
            print(f'\n{mode_title}', flush=True)
            for model in paper + others:
                for metric_name, suffix in METRIC_KEYS:
                    key = f'{ap_name}_{suffix}'
                    values, epochs = series_for(by_model[model], mode, key)
                    mean, std = mean_std(values.values())
                    if mode == 'best':
                        ep_txt = ' '.join(f's{s}:{epochs[s]}' for s in sorted(epochs))
                    else:
                        ep_txt = ''
                    extra = f'  epochs {ep_txt}' if ep_txt and suffix == 'entire_mAP' else ''
                    print(
                        f'  {label(model):16} {metric_name:14} '
                        f'{fmt_pm(mean, std, len(values))}{extra}',
                        flush=True,
                    )
            print('  paired differences (left - right), two-sided paired t-test', flush=True)
            for pair_name, left, right in PAIRS:
                if left not in by_model or right not in by_model:
                    print(f'  {pair_name}: missing model', flush=True)
                    continue
                for metric_name, suffix in METRIC_KEYS:
                    key = f'{ap_name}_{suffix}'
                    lv, _ = series_for(by_model[left], mode, key)
                    rv, _ = series_for(by_model[right], mode, key)
                    seeds = sorted(set(lv) & set(rv))
                    if not seeds:
                        print(f'  {pair_name:16} {metric_name:14} no paired seeds', flush=True)
                        continue
                    diff = [lv[s] - rv[s] for s in seeds]
                    mean, std = mean_std(diff)
                    p = paired_p(diff)
                    seed_txt = ' '.join(f's{s}:{diff[i]:+.3f}' for i, s in enumerate(seeds))
                    p_txt = 'nan' if p != p else f'{p:.4g}'
                    print(
                        f'  {pair_name:16} {metric_name:14} {seed_txt}  '
                        f'mean {fmt_pm(mean, std, len(diff))}  p={p_txt}',
                        flush=True,
                    )


def main():
    curves = load_logged_curves()
    runs = discover_runs(curves)
    done, rows = load_done()
    print(f'CSV {OUT_CSV} already has {len(done)} epoch rows', flush=True)
    pending = []
    for run in runs:
        need = [e for e in run['epochs'] if (run['run_id'], e) not in done]
        if not need:
            print(f"skip already in CSV {run['model']} seed {run['seed']} {run['run_id']}", flush=True)
        else:
            pending.append((run, need))
            print(
                f"queue {run['model']} seed {run['seed']} {run['run_id']} "
                f"best {run['best_epoch']} epochs {need}",
                flush=True,
            )

    if pending:
        print('Loading VoD devkit (CPU) and val infos...', flush=True)
        vod_eval = load_vod_eval()
        gt, gt_ids = load_gt()
        overlaps = min_overlaps_audit()
        print(f'val frames {len(gt)}', flush=True)
        for run, need in pending:
            for epoch in need:
                path = result_path(run['run_id'], epoch)
                print(f"EVAL {run['model']} seed {run['seed']} epoch {epoch}", flush=True)
                det = load_det(path, gt_ids)
                scores = eval_epoch(vod_eval, gt, det, overlaps)
                row = {
                    'run_id': run['run_id'],
                    'model': run['model'],
                    'seed': run['seed'],
                    'epoch': epoch,
                }
                row.update(scores)
                check_audit(row)
                append_row({k: row[k] for k in FIELDS})
                rows.append({k: str(row[k]) for k in FIELDS})
                print(
                    f"  R11 entire {scores['r11_entire_car']:.3f}/"
                    f"{scores['r11_entire_ped']:.3f}/{scores['r11_entire_cyc']:.3f} "
                    f"mAP {scores['r11_entire_mAP']:.3f}  "
                    f"R40 mAP {scores['r40_entire_mAP']:.3f}  "
                    f"corr R11 {scores['r11_corr_mAP']:.3f} R40 {scores['r40_corr_mAP']:.3f}",
                    flush=True,
                )

    print_summary(rows)


if __name__ == '__main__':
    main()
