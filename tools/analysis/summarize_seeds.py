#!/usr/bin/env python3
"""Summarize per-epoch logs: best / final / last-5, mean±std, paired t-test (n=4).

Default inputs: results/per_epoch_vod.csv and results/per_epoch_vod_r40.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_vod(path: Path, map_key: str):
    """Return dict[(model, seed)] -> list of (epoch, entire_mAP, corr_mAP)."""
    rows = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            model = row['model']
            # E* rows encode seed in the name; normalize
            seed = int(row['seed'])
            ep = int(row['epoch'])
            if map_key == 'r11':
                entire = float(row.get('entire_mAP') or row['r11_entire_mAP'])
                corr = float(row.get('corr_mAP') or row['r11_corr_mAP'])
            else:
                entire = float(row['r40_entire_mAP'])
                corr = float(row['r40_corr_mAP'])
            rows[(model, seed)].append((ep, entire, corr))
    for k in rows:
        rows[k].sort()
    return rows


def rules(series):
    """series: list (epoch, entire, corr) for epochs ideally 51-80."""
    by_ep = {e: (m, c) for e, m, c in series}
    best_ep = max(series, key=lambda t: t[1])[0]
    final_ep = 80 if 80 in by_ep else max(by_ep)
    last5 = [by_ep[e][0] for e in range(76, 81) if e in by_ep]
    return {
        'best_ep': best_ep,
        'best': by_ep[best_ep][0],
        'best_corr': by_ep[best_ep][1],
        'final': by_ep[final_ep][0],
        'last5': float(np.mean(last5)) if last5 else float('nan'),
    }


def paired_ttest(a, b):
    """Two-sided paired t-test; returns mean_diff, t, p (n-1 df)."""
    from math import sqrt
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    t = mean / (sd / sqrt(n)) if sd > 0 else float('inf')
    # Student-t survival (two-sided) via scipy if present
    try:
        from scipy.stats import t as student_t
        p = float(2 * student_t.sf(abs(t), df=n - 1))
    except Exception:
        p = float('nan')
    return mean, t, p


def summarize_model(name, seeds_data, label):
    bests = [seeds_data[s]['best'] for s in sorted(seeds_data)]
    corrs = [seeds_data[s]['best_corr'] for s in sorted(seeds_data)]
    finals = [seeds_data[s]['final'] for s in sorted(seeds_data)]
    last5s = [seeds_data[s]['last5'] for s in sorted(seeds_data)]
    print(f'\n[{label}] {name}  (n={len(bests)})')
    for s in sorted(seeds_data):
        d = seeds_data[s]
        print(f'  seed {s}: best@{d["best_ep"]}={d["best"]:.3f}  '
              f'final={d["final"]:.3f}  last5={d["last5"]:.3f}  corr={d["best_corr"]:.3f}')
    print(f'  best   mean±std = {np.mean(bests):.2f} ± {np.std(bests, ddof=1):.2f}')
    print(f'  final  mean±std = {np.mean(finals):.2f} ± {np.std(finals, ddof=1):.2f}')
    print(f'  last5  mean±std = {np.mean(last5s):.2f} ± {np.std(last5s, ddof=1):.2f}')
    print(f'  corr   mean±std = {np.mean(corrs):.2f} ± {np.std(corrs, ddof=1):.2f}')
    return bests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vod_csv', type=Path, default=REPO / 'results/per_epoch_vod.csv')
    ap.add_argument('--r40_csv', type=Path, default=REPO / 'results/per_epoch_vod_r40.csv')
    ap.add_argument('--tj4d_csv', type=Path, default=REPO / 'results/per_epoch_tj4d.csv')
    args = ap.parse_args()

    print('=== VoD R11 (entire-area) ===')
    vod = load_vod(args.vod_csv, 'r11')
    # Main paper models stored as MambaNet-B / M3-Net / E4c_* 
    groups = {
        'Radar M3-Net (repro.)': [(m, s) for (m, s) in vod if m == 'M3-Net'],
        '4D-MambaNet w/o RCS input': [(m, s) for (m, s) in vod if m.startswith('E4c_')],
        '4D-MambaNet': [(m, s) for (m, s) in vod if m == 'MambaNet-B'],
    }
    best_r11 = {}
    for label, keys in groups.items():
        data = {}
        for m, s in keys:
            data[s] = rules(vod[(m, s)])
        best_r11[label] = summarize_model(label, data, 'R11')

    print('\n=== VoD R40 (entire-area, best epoch by R11) ===')
    if args.r40_csv.exists():
        r40 = load_vod(args.r40_csv, 'r40')
        # Align R40 to R11-selected best epoch
        for label, keys in groups.items():
            vals = []
            for m, s in sorted(keys, key=lambda t: t[1]):
                series = r40.get((m, s))
                if series is None and m.startswith('E4c_'):
                    # r40 csv may use run_id; try model field variants
                    series = next((v for (mm, ss), v in r40.items() if ss == s and 'E4c' in mm), None)
                if series is None:
                    continue
                # best epoch from R11
                best_ep = rules(vod[(m, s)])['best_ep']
                by_ep = {e: ent for e, ent, _ in series}
                if best_ep in by_ep:
                    vals.append(by_ep[best_ep])
                    print(f'  {label} seed {s}: R40@{best_ep}={by_ep[best_ep]:.3f}')
            if vals:
                print(f'  {label}: {np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}')

    print('\n=== Paired t-test: 4D-MambaNet vs Radar M3-Net repro (R11 best) ===')
    a = best_r11['4D-MambaNet']
    b = best_r11['Radar M3-Net (repro.)']
    md, t, p = paired_ttest(a, b)
    print(f'  mean diff = {md:+.2f}  t={t:.3f}  p={p:.3f}  (paper: +1.43, p=0.059)')

    if args.r40_csv.exists():
        # R40 paired using best-by-R11 epochs
        r40 = load_vod(args.r40_csv, 'r40')
        a40, b40 = [], []
        for s in range(4):
            be = rules(vod[('MambaNet-B', s)])['best_ep']
            br = rules(vod[('M3-Net', s)])['best_ep']
            a40.append(dict((e, m) for e, m, _ in r40[('MambaNet-B', s)])[be])
            b40.append(dict((e, m) for e, m, _ in r40[('M3-Net', s)])[br])
        md, t, p = paired_ttest(a40, b40)
        print(f'\n=== Paired t-test R40 best ===')
        print(f'  mean diff = {md:+.2f}  t={t:.3f}  p={p:.3f}  (paper: +1.55, p=0.017)')

    if args.tj4d_csv.exists():
        print('\n=== TJ4D (best by mAP3D) ===')
        rows = defaultdict(list)
        with open(args.tj4d_csv) as f:
            for row in csv.DictReader(f):
                rows[(row['model'], int(row['seed']))].append(
                    (int(row['epoch']), float(row['mAP3D']), float(row['mAPBEV']))
                )
        for label, model in [('Radar M3-Net (repro.)', None), ('4D-MambaNet', 'MambaNet-B')]:
            if label.startswith('Radar'):
                keys = [k for k in rows if k[0] in ('Baseline',) or k[0].startswith('E5_')]
            else:
                keys = [k for k in rows if k[0] == 'MambaNet-B']
            m3, mb = [], []
            for m, s in sorted(keys, key=lambda t: t[1]):
                best = max(rows[(m, s)], key=lambda t: t[1])
                print(f'  {label} seed {s}: ep{best[0]} mAP3D={best[1]:.3f} mAPBEV={best[2]:.3f}')
                m3.append(best[1]); mb.append(best[2])
            if m3:
                print(f'  -> 3D {np.mean(m3):.2f}±{np.std(m3, ddof=1):.2f}  '
                      f'BEV {np.mean(mb):.2f}±{np.std(mb, ddof=1):.2f}')


if __name__ == '__main__':
    main()
