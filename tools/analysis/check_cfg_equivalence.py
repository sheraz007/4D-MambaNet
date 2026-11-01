#!/usr/bin/env python3
"""Assert public configs match the training configs (ignoring paths and names).

Compares resolved MODEL + OPTIMIZATION (+ DATA_CONFIG structure except path-like keys)
between an "old" saved/tools config and a "new" public config.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import yaml
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__import__('os').environ.get('OLD_OPENPCDET_ROOT', str(REPO.parent / 'myopenpcdet')))
RCS = Path(__import__('os').environ.get('OLD_RCSFIX_ROOT', str(REPO.parent / 'rcs_fix')))

IGNORE_KEYS = {
    'DATA_PATH', 'INFO_PATH', 'lidar_path', 'clip_path', 'DB_INFO_PATH',
    'ROOT_DIR', 'TAG', 'EXP_GROUP_PATH', '_BASE_CONFIG_',
}
PATHISH = ('path', 'dir', 'root', 'file')


def _py_eval(loader, node):
    return eval(loader.construct_scalar(node), {'np': np, 'math': math, '__builtins__': {}})


yaml.SafeLoader.add_constructor('!python/eval', _py_eval)


def _merge(base, new):
    for k, v in new.items():
        if k == '_BASE_CONFIG_':
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_path(base_ref: str, cur_file: Path, search_roots):
    p = Path(base_ref)
    if p.is_absolute() and p.exists():
        return p
    cands = [
        cur_file.parent / base_ref,
        *(root / base_ref for root in search_roots),
        *(root / 'tools' / base_ref for root in search_roots),
    ]
    # strip leading tools/
    if base_ref.startswith('tools/'):
        cands += [root / base_ref[len('tools/'):] for root in search_roots]
        cands += [root / 'tools' / base_ref[len('tools/'):] for root in search_roots]
    if base_ref.startswith('cfgs/'):
        cands += [root / 'tools' / base_ref for root in search_roots]
    for c in cands:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f'Cannot resolve base config {base_ref} from {cur_file}')


def resolve_cfg(path: Path, search_roots):
    path = path.resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(path)
    if '_BASE_CONFIG_' in raw:
        base_path = _resolve_path(str(raw['_BASE_CONFIG_']), path, search_roots)
        base = resolve_cfg(base_path, search_roots)
        return _merge(base, {k: v for k, v in raw.items() if k != '_BASE_CONFIG_'})
    return raw


def normalize(obj, key=None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in IGNORE_KEYS:
                continue
            if key in IGNORE_KEYS:
                continue
            if any(t in str(k).lower() for t in ('path',)) and k not in (
                'POINT_CLOUD_RANGE', 'POST_CENTER_LIMIT_RANGE',
            ):
                # keep structural non-filesystem keys; drop filesystem-like
                if k.endswith('_PATH') or k.endswith('_path') or k in ('DATA_PATH', 'INFO_PATH'):
                    continue
            out[k] = normalize(v, k)
        return out
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    return obj


def _eq(a, b, prefix=''):
    if type(a) != type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        # allow list/tuple
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return [f'{prefix}: type {type(a)} vs {type(b)}']
    diffs = []
    if isinstance(a, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            if k not in a:
                diffs.append(f'{prefix}.{k}: missing in old')
            elif k not in b:
                diffs.append(f'{prefix}.{k}: missing in new')
            else:
                diffs.extend(_eq(a[k], b[k], f'{prefix}.{k}'))
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f'{prefix}: len {len(a)} vs {len(b)}']
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_eq(x, y, f'{prefix}[{i}]'))
        return diffs
    if isinstance(a, float) or isinstance(b, float):
        if abs(float(a) - float(b)) > 1e-9:
            return [f'{prefix}: {a} vs {b}']
        return []
    if a != b:
        return [f'{prefix}: {a!r} vs {b!r}']
    return []


PAIRS = [
    # (label, old, new, search roots for old)
    ('4d_mambanet',
     SRC / 'tools/cfgs/vod_models/last_model/uni_ptv3_mamba_only.yaml',
     REPO / 'tools/cfgs/vod/4d_mambanet.yaml'),
    ('m3net_repro',
     SRC / 'output/uni_ptv3_xcout_seed_sweep/seed_0/uni_ptv3_xcout.yaml',
     REPO / 'tools/cfgs/vod/m3net_repro.yaml'),
    ('4d_mambanet_no_rcs_input',
     SRC / 'tools/cfgs/rev1/E4c_no_input_rcs_no_prior.yaml',
     REPO / 'tools/cfgs/vod/4d_mambanet_no_rcs_input.yaml'),
    ('per_frame_native',
     SRC / 'tools/cfgs/rev1/E0_per_frame.yaml',
     REPO / 'tools/cfgs/vod/ablation_scan/per_frame_native.yaml'),
    ('shuffle',
     SRC / 'tools/cfgs/rev1/E1_shuffle.yaml',
     REPO / 'tools/cfgs/vod/ablation_scan/shuffle.yaml'),
    ('hilbert',
     SRC / 'tools/cfgs/rev1/E1_hilbert.yaml',
     REPO / 'tools/cfgs/vod/ablation_scan/hilbert.yaml'),
    ('bidirectional',
     SRC / 'tools/cfgs/rev1/E1_bidir.yaml',
     REPO / 'tools/cfgs/vod/ablation_scan/bidirectional.yaml'),
    ('global_attention',
     SRC / 'tools/cfgs/rev1/E3_attention.yaml',
     REPO / 'tools/cfgs/vod/ablation_attention/global_attention.yaml'),
    ('placement_xcout',
     SRC / 'tools/cfgs/rev1/E6_xcout.yaml',
     REPO / 'tools/cfgs/vod/ablation_placement/xcout.yaml'),
    ('placement_xc4_xcout',
     SRC / 'tools/cfgs/rev1/E6_xc4_xcout.yaml',
     REPO / 'tools/cfgs/vod/ablation_placement/xc4_xcout.yaml'),
    ('tj4d_4d_mambanet',
     SRC / 'tools/cfgs/tj4d_models/uni_ptv3_mamba_only_tj4d.yaml',
     REPO / 'tools/cfgs/tj4d/4d_mambanet.yaml'),
    ('tj4d_m3net_repro',
     SRC / 'tools/cfgs/tj4d_models/msvfe_ptv3.yaml',
     REPO / 'tools/cfgs/tj4d/m3net_repro.yaml'),
    ('rcs_bev_4d_mambanet',
     RCS / 'tools/cfgs/rcsfix/uni_ptv3_mamba_rcs_fixed.yaml',
     REPO / 'tools/cfgs/vod/ablation_rcs_bev/4d_mambanet_bev_prior.yaml'),
    ('rcs_bev_no_input',
     RCS / 'tools/cfgs/rcsfix/uni_ptv3_mamba_rcs_fixed_noinput.yaml',
     REPO / 'tools/cfgs/vod/ablation_rcs_bev/no_rcs_input_bev_prior.yaml'),
    ('rcs_bev_m3net',
     RCS / 'tools/cfgs/rcsfix/uni_ptv3_xcout_rcs_fixed.yaml',
     REPO / 'tools/cfgs/vod/ablation_rcs_bev/m3net_bev_prior.yaml'),
]


def compare_pair(label, old_path, new_path):
    old_roots = [SRC, RCS, SRC / 'tools', RCS / 'tools']
    new_roots = [REPO, REPO / 'tools']
    # For old configs that base through tools/cfgs paths, also allow REPO? no
    old = resolve_cfg(old_path, old_roots + [old_path.parent])
    new = resolve_cfg(new_path, new_roots + [new_path.parent])
    # Compare MODEL and OPTIMIZATION primarily; DATA_CONFIG sans paths
    sections = ['MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG']
    diffs = []
    for sec in sections:
        if sec not in old and sec not in new:
            continue
        oa = normalize(old.get(sec, {}))
        na = normalize(new.get(sec, {}))
        diffs.extend(_eq(oa, na, sec))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default=None, help='optional single pair label')
    args = ap.parse_args()
    failed = 0
    for label, old_p, new_p in PAIRS:
        if args.pair and label != args.pair:
            continue
        print(f'=== {label} ===')
        print(f'  old: {old_p}')
        print(f'  new: {new_p}')
        try:
            diffs = compare_pair(label, old_p, new_p)
        except Exception as e:
            print(f'  ERROR: {e}')
            failed += 1
            continue
        if not diffs:
            print('  OK (equivalent ignoring paths/names)')
        else:
            print(f'  DIFF ({len(diffs)}):')
            for d in diffs[:40]:
                print('   ', d)
            failed += 1
    if failed:
        print(f'\nFAILED pairs: {failed}')
        sys.exit(1)
    print('\nAll pairs equivalent.')


if __name__ == '__main__':
    main()
