#!/usr/bin/env python3
"""VoD val-split FPS benchmark (batch size 1, one idle RTX 3090).

Does not modify model code. Post-processing is skipped only by temporarily
replacing bound methods on the loaded module for the network-only timer.

Each model repeat is a separate Python process so allocator and cuDNN state
cannot leak into the next model. cudnn.benchmark is set False in that process.

Measurements (first 10% of frames are warm-up and are excluded from FPS):
  (a) network-only
      Batch already on GPU. torch.cuda.synchronize() before and after each frame.
      Real eval forward, with these calls stubbed:
        - VoxelNeXtHead.generate_predicted_boxes (box decode + NMS)
        - AnchorHeadSingle.generate_predicted_boxes (anchor box decode)
        - detector.post_processing (anchor NMS, head merge, recall)
      Backbone and head convolutions still run. No data loading in the timer.
  (b) end-to-end
      Wall clock of DataLoader (load + preprocess, NUM_WORKERS) + H2D + full
      model forward + post-processing (NMS). Synchronize at the start and end
      of the timed region. No KITTI metric evaluation.
  (c) model latency
      Batch already on GPU. Full forward including decode + NMS + post_processing.
      Synchronize per frame. No data loading in the timer.

Outer repeats alternate model order:
  repeat 1: 4D-MambaNet, MambaNet-B, M3-Net Repro
  repeat 2: reverse
  repeat 3: original
FPS for a repeat is measured_frames / elapsed_seconds.
mean ± std uses the sample standard deviation (ddof=1) across the 3 repeats.

Active parameters: parameters whose owning module fired a forward hook during
one real (unpatched) eval forward. Counted once via parameters(recurse=False).
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

MODELS = [
    {
        'name': '4D-MambaNet',
        'cfg': 'tools/cfgs/vod_models/last_model/uni_ptv3_mamba_rcs.yaml',
        'ckpt': 'output/mamba_rcs_s0/ckpt/checkpoint_epoch_74.pth',
    },
    {
        'name': 'MambaNet-B',
        'cfg': 'tools/cfgs/vod_models/last_model/uni_ptv3_mamba_only.yaml',
        'ckpt': 'output/mamba_only_s0_gpu1/ckpt/checkpoint_epoch_78.pth',
    },
    {
        'name': 'M3-Net Repro',
        'cfg': 'tools/cfgs/vod_models/last_model/uni_ptv3_xcout.yaml',
        'ckpt': 'output/uni_ptv3_xcout_seed_sweep/seed_0/ckpt/checkpoint_epoch_78.pth',
    },
]

REPEATS = 3
WARMUP_FRAC = 0.10
BATCH_SIZE = 1
NUM_WORKERS = 4
CSV_PATH = ROOT / 'docs' / 'rev1' / 'fps_table6.csv'


def pick_idle_3090():
    """Return a physical GPU index that is an idle RTX 3090. Abort otherwise."""
    print('=== nvidia-smi (before CUDA init) ===')
    subprocess.run(['nvidia-smi'], check=False)
    gpu_txt = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    )
    app_txt = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
            '--format=csv,noheader',
        ],
        text=True,
    )
    busy = set()
    for line in app_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        busy.add(line.split(',')[0].strip())
    idle = []
    for line in gpu_txt.splitlines():
        idx_s, uuid, name, mem_s, util_s = [x.strip() for x in line.split(',')]
        if 'RTX 3090' not in name:
            continue
        mem = int(mem_s)
        util = int(util_s)
        if uuid in busy or mem > 256 or util > 0:
            print(f'skip GPU {idx_s} ({name}): mem={mem} MiB util={util}% busy={uuid in busy}')
            continue
        idle.append(int(idx_s))
    if not idle:
        raise SystemExit('No idle RTX 3090 (compute process, memory, or utilization). Aborting.')
    chosen = idle[0]
    print(f'Using physical GPU {chosen} (idle RTX 3090). Other idle 3090s: {idle[1:]}')
    return chosen


def _stub_anchor_decode(batch_size, cls_preds, box_preds, dir_cls_preds=None):
    return (
        cls_preds.new_zeros((batch_size, 1, cls_preds.shape[-1])),
        box_preds.new_zeros((batch_size, 1, box_preds.shape[-1])),
    )


def _stub_voxelnext_decode(batch_size, pred_dicts, voxel_indices, spatial_shape, points, add_features=None):
    device = voxel_indices.device
    out = []
    for _ in range(int(batch_size)):
        out.append({
            'pre_nms_pred_boxes': torch.zeros((0, 7), device=device),
            'pre_nms_pred_scores': torch.zeros((0,), device=device),
            'pre_nms_pred_labels': torch.zeros((0,), dtype=torch.long, device=device),
            'pred_boxes': torch.zeros((0, 7), device=device),
            'pred_scores': torch.zeros((0,), device=device),
            'pred_labels': torch.zeros((0,), dtype=torch.long, device=device),
            'pred_ious': torch.zeros((0,), device=device),
        })
    return out


def _stub_post_processing(batch_dict):
    return [], {}


class _SkipPostprocess:
    """Instance-level stubs. Restored on exit. Does not edit class source."""

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        self.saved.append(('model', self.model, 'post_processing', self.model.post_processing))
        self.model.post_processing = _stub_post_processing
        for mod in self.model.modules():
            cls_name = type(mod).__name__
            if cls_name == 'VoxelNeXtHead':
                self.saved.append((cls_name, mod, 'generate_predicted_boxes', mod.generate_predicted_boxes))
                mod.generate_predicted_boxes = _stub_voxelnext_decode
            elif cls_name == 'AnchorHeadSingle':
                self.saved.append((cls_name, mod, 'generate_predicted_boxes', mod.generate_predicted_boxes))
                mod.generate_predicted_boxes = _stub_anchor_decode
        return self

    def __exit__(self, exc_type, exc, tb):
        for _cls, obj, attr, orig in reversed(self.saved):
            setattr(obj, attr, orig)
        self.saved.clear()
        return False


def _count_active_params(model, batch):
    called = set()
    hooks = []

    def _hook(mod, _inp, _out):
        called.add(id(mod))

    for mod in model.modules():
        hooks.append(mod.register_forward_hook(_hook))
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()

    seen = set()
    active = 0
    for mod in model.modules():
        if id(mod) not in called:
            continue
        for p in mod.parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            active += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return total, active


def _make_loader(cfg, logger):
    from pcdet.datasets import build_dataloader
    dataset, loader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=False,
        workers=NUM_WORKERS,
        logger=logger,
        training=False,
    )
    return dataset, loader


def _load_model(cfg, dataset, ckpt, logger):
    from pcdet.models import build_network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return model


def _batch_frames(batch):
    return int(batch['batch_size'])


def _measure_synced(model, loader, warmup_frames, skip_post):
    """Per-frame synchronized GPU time. Data loading and H2D are outside the timer."""
    from pcdet.models import load_data_to_gpu
    from contextlib import nullcontext
    elapsed = 0.0
    measured = 0
    seen = 0
    ctx = _SkipPostprocess(model) if skip_post else nullcontext()
    with ctx:
        for batch in loader:
            load_data_to_gpu(batch)
            n = _batch_frames(batch)
            seen += n
            with torch.no_grad():
                if seen <= warmup_frames:
                    model(batch)
                    continue
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - t0
            measured += n
            if seen % 256 == 0:
                print(f'    synced {seen}/{warmup_frames + measured} frames', flush=True)
    return measured, elapsed


def measure_forward(model, loader, warmup_frames):
    return _measure_synced(model, loader, warmup_frames, skip_post=True)


def measure_latency(model, loader, warmup_frames):
    return _measure_synced(model, loader, warmup_frames, skip_post=False)


def measure_e2e(model, loader, warmup_frames):
    from pcdet.models import load_data_to_gpu
    seen = 0
    measured = 0
    it = iter(loader)
    for batch in it:
        load_data_to_gpu(batch)
        n = _batch_frames(batch)
        seen += n
        with torch.no_grad():
            model(batch)
        if warmup_frames > 0 and seen >= warmup_frames:
            break
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in it:
        load_data_to_gpu(batch)
        n = _batch_frames(batch)
        with torch.no_grad():
            model(batch)
        measured += n
        if (seen + measured) % 256 == 0:
            print(f'    e2e {seen + measured} frames', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return measured, elapsed, seen


def _fps(measured, elapsed):
    if measured <= 0 or elapsed <= 0:
        raise RuntimeError(f'bad timing: frames={measured} elapsed={elapsed}')
    return measured / elapsed


def _mean_std(xs):
    import numpy as np
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _precision_label(model):
    dtypes = sorted({str(p.dtype).replace('torch.', '') for p in model.parameters()})
    return ','.join(dtypes) if dtypes else 'none'


def _fresh_cuda():
    torch.backends.cudnn.benchmark = False
    print(f'torch.backends.cudnn.benchmark = {torch.backends.cudnn.benchmark}', flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_one(spec, logger):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file

    cfg_path = ROOT / spec['cfg']
    ckpt_path = ROOT / spec['ckpt']
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    cfg = EasyDict()
    cfg.ROOT_DIR = ROOT
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(str(cfg_path), cfg)
    split_name = cfg.DATA_CONFIG.DATA_SPLIT['test']

    print(f'\n===== {spec["name"]} =====')
    print(f'cfg:  {cfg_path}')
    print(f'ckpt: {ckpt_path}')
    print(f"DATA_SPLIT['test'] = {split_name}")

    dataset, loader = _make_loader(cfg, logger)
    n_frames = len(dataset)
    warmup = int(n_frames * WARMUP_FRAC)
    print(f'val frames = {n_frames}  warmup = {warmup}  measured = {n_frames - warmup}')

    model = _load_model(cfg, dataset, ckpt_path, logger)
    precision = _precision_label(model)
    _fresh_cuda()

    # One real forward so hooks see modules used with post-processing enabled.
    first = next(iter(loader))
    from pcdet.models import load_data_to_gpu
    load_data_to_gpu(first)
    total_params, active_params = _count_active_params(model, first)
    print(f'params total = {total_params}  active = {active_params}  dtype = {precision}', flush=True)
    del first, loader

    def _check_frames(dataset):
        if len(dataset) != n_frames:
            raise RuntimeError(f'frame count changed: {len(dataset)} vs {n_frames}')

    _dataset, loader_a = _make_loader(cfg, logger)
    _check_frames(_dataset)
    m_a, t_a = measure_forward(model, loader_a, warmup)
    if m_a != n_frames - warmup:
        raise RuntimeError(f'forward measured {m_a}, expected {n_frames - warmup}')
    fps_a = _fps(m_a, t_a)
    del loader_a

    _dataset, loader_c = _make_loader(cfg, logger)
    _check_frames(_dataset)
    m_c, t_c = measure_latency(model, loader_c, warmup)
    if m_c != n_frames - warmup:
        raise RuntimeError(f'latency measured {m_c}, expected {n_frames - warmup}')
    fps_c = _fps(m_c, t_c)
    del loader_c

    _dataset, loader_b = _make_loader(cfg, logger)
    _check_frames(_dataset)
    m_b, t_b, warmed = measure_e2e(model, loader_b, warmup)
    if warmed < warmup:
        raise RuntimeError(f'e2e warmup saw {warmed} frames, need {warmup}')
    if m_b != n_frames - warmed:
        raise RuntimeError(
            f'e2e measured {m_b}, expected {n_frames - warmed} (warmup consumed {warmed})'
        )
    fps_b = _fps(m_b, t_b)
    del loader_b

    peak = int(torch.cuda.max_memory_allocated())
    print(
        f'  network {fps_a:.3f} FPS  e2e {fps_b:.3f} FPS  latency {fps_c:.3f} FPS  '
        f'peak {peak / (1024 ** 2):.1f} MiB',
        flush=True,
    )
    return {
        'model': spec['name'],
        'cfg': spec['cfg'],
        'ckpt': spec['ckpt'],
        'split': split_name,
        'frames': n_frames,
        'warmup_frames': warmup,
        'measured_frames': n_frames - warmup,
        'fps_forward': fps_a,
        'fps_e2e': fps_b,
        'fps_latency': fps_c,
        'peak_gpu_mem_bytes': peak,
        'params_total': total_params,
        'params_active': active_params,
        'dtype': precision,
        'num_workers': NUM_WORKERS,
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'env': _env_info(),
    }


def _env_info():
    return {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': str(torch.backends.cudnn.version()),
        'cudnn_benchmark': str(bool(torch.backends.cudnn.benchmark)),
        'cudnn_enabled': str(bool(torch.backends.cudnn.enabled)),
        'gpu': torch.cuda.get_device_name(0),
        'precision': 'fp32' if not torch.is_autocast_enabled() else 'autocast',
    }


def _summarize(samples):
    """samples: list of per-process result dicts for one model, in repeat order."""
    fwd = [s['fps_forward'] for s in samples]
    e2e = [s['fps_e2e'] for s in samples]
    lat = [s['fps_latency'] for s in samples]
    fwd_mean, fwd_std = _mean_std(fwd)
    e2e_mean, e2e_std = _mean_std(e2e)
    lat_mean, lat_std = _mean_std(lat)
    peak = max(s['peak_gpu_mem_bytes'] for s in samples)
    base = samples[0]
    row = {
        'model': base['model'],
        'cfg': base['cfg'],
        'ckpt': base['ckpt'],
        'split': base['split'],
        'frames': base['frames'],
        'warmup_frames': base['warmup_frames'],
        'measured_frames': base['measured_frames'],
        'repeats': len(samples),
        'num_workers': base['num_workers'],
        'fps_forward_mean': fwd_mean,
        'fps_forward_std': fwd_std,
        'fps_e2e_mean': e2e_mean,
        'fps_e2e_std': e2e_std,
        'fps_latency_mean': lat_mean,
        'fps_latency_std': lat_std,
        'peak_gpu_mem_bytes': peak,
        'peak_gpu_mem_mib': peak / (1024 ** 2),
        'params_total': base['params_total'],
        'params_active': base['params_active'],
        'dtype': base['dtype'],
        'cudnn_benchmark': base['cudnn_benchmark'],
    }
    for i, s in enumerate(samples, start=1):
        row[f'fps_forward_r{i}'] = s['fps_forward']
        row[f'fps_e2e_r{i}'] = s['fps_e2e']
        row[f'fps_latency_r{i}'] = s['fps_latency']
        row[f'peak_gpu_mem_bytes_r{i}'] = s['peak_gpu_mem_bytes']
    return row


def _write_csv(rows, env):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model', 'cfg', 'ckpt', 'split', 'frames', 'warmup_frames', 'measured_frames', 'repeats',
        'num_workers',
        'fps_forward_mean', 'fps_forward_std',
        'fps_forward_r1', 'fps_forward_r2', 'fps_forward_r3',
        'fps_e2e_mean', 'fps_e2e_std',
        'fps_e2e_r1', 'fps_e2e_r2', 'fps_e2e_r3',
        'fps_latency_mean', 'fps_latency_std',
        'fps_latency_r1', 'fps_latency_r2', 'fps_latency_r3',
        'peak_gpu_mem_bytes', 'peak_gpu_mem_mib',
        'peak_gpu_mem_bytes_r1', 'peak_gpu_mem_bytes_r2', 'peak_gpu_mem_bytes_r3',
        'params_total', 'params_active', 'dtype',
        'torch', 'cuda', 'cudnn', 'cudnn_benchmark', 'cudnn_enabled', 'gpu', 'amp',
    ]
    with CSV_PATH.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = dict(row)
            out.update({
                'torch': env['torch'],
                'cuda': env['cuda'],
                'cudnn': env['cudnn'],
                'cudnn_benchmark': env['cudnn_benchmark'],
                'cudnn_enabled': env['cudnn_enabled'],
                'gpu': env['gpu'],
                'amp': env['precision'],
            })
            w.writerow({k: out.get(k, '') for k in fieldnames})
    print(f'\nWrote {CSV_PATH}', flush=True)


def _print_markdown(rows, env):
    print('\nRuntime', flush=True)
    print(f"- torch {env['torch']}", flush=True)
    print(f"- CUDA {env['cuda']}", flush=True)
    print(
        f"- cuDNN {env['cudnn']}  enabled={env['cudnn_enabled']}  "
        f"benchmark={env['cudnn_benchmark']}",
        flush=True,
    )
    print(f"- GPU {env['gpu']}", flush=True)
    print(f"- precision {env['precision']} (parameter dtype is per model; no autocast)", flush=True)
    print(
        f"- batch size {BATCH_SIZE}  DataLoader num_workers for (b) = {NUM_WORKERS}  "
        f"repeats {REPEATS}  warmup {WARMUP_FRAC:.0%}",
        flush=True,
    )
    print('- order: r1 Mamba-RCS, Mamba-B, M3-Net; r2 reverse; r3 original', flush=True)
    print('- each model repeat is a fresh Python process', flush=True)
    print('- FPS std is sample std (ddof=1) over repeats', flush=True)
    print(flush=True)
    header = (
        '| Model | Frames | (a) network FPS | (b) e2e FPS | (c) model latency FPS | '
        'workers (b) | Peak mem (MiB) | Params total | Params active | dtype |'
    )
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|'
    print(header, flush=True)
    print(sep, flush=True)
    for r in rows:
        print(
            '| {model} | {frames} | {fm:.2f} ± {fs:.2f} | {em:.2f} ± {es:.2f} | '
            '{lm:.2f} ± {ls:.2f} | {workers} | {mem:.1f} | {pt} | {pa} | {dtype} |'.format(
                model=r['model'],
                frames=r['frames'],
                fm=r['fps_forward_mean'],
                fs=r['fps_forward_std'],
                em=r['fps_e2e_mean'],
                es=r['fps_e2e_std'],
                lm=r['fps_latency_mean'],
                ls=r['fps_latency_std'],
                workers=r['num_workers'],
                mem=r['peak_gpu_mem_mib'],
                pt=r['params_total'],
                pa=r['params_active'],
                dtype=r['dtype'],
            ),
            flush=True,
        )


def _repeat_orders():
    idx = list(range(len(MODELS)))
    return [idx, list(reversed(idx)), list(idx)]


def _run_worker(model_index):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    global torch
    import torch
    from pcdet.config_ext import apply_yaml_ext
    apply_yaml_ext()
    torch.backends.cudnn.benchmark = False
    print('torch', torch.__version__, flush=True)
    print('CUDA', torch.version.cuda, flush=True)
    print(
        'cuDNN', torch.backends.cudnn.version(),
        'benchmark', torch.backends.cudnn.benchmark,
        flush=True,
    )
    print('GPU', torch.cuda.get_device_name(0), flush=True)
    print('autocast', torch.is_autocast_enabled(), flush=True)
    if torch.backends.cudnn.benchmark:
        raise RuntimeError('cudnn.benchmark must be False')

    logger = logging.getLogger('bench_fps')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    result = run_one(MODELS[model_index], logger)
    print('BENCH_JSON ' + json.dumps(result), flush=True)


def _launch_worker(model_index, gpu, repeat):
    spec = MODELS[model_index]
    print(
        f'\n######## repeat {repeat}  {spec["name"]}  (subprocess, physical GPU {gpu}) ########',
        flush=True,
    )
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', str(model_index)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    payload = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith('BENCH_JSON '):
            payload = json.loads(line[len('BENCH_JSON '):])
    code = proc.wait()
    if code != 0 or payload is None:
        raise RuntimeError(f'worker failed for {spec["name"]} repeat {repeat} (exit {code})')
    payload['repeat'] = repeat
    return payload


def main():
    gpu = pick_idle_3090()
    by_model = {spec['name']: [] for spec in MODELS}
    env = None
    for repeat, order in enumerate(_repeat_orders(), start=1):
        names = [MODELS[i]['name'] for i in order]
        print(f'\n===== outer repeat {repeat}/{REPEATS} order: {names} =====', flush=True)
        for model_index in order:
            sample = _launch_worker(model_index, gpu, repeat)
            by_model[sample['model']].append(sample)
            env = sample['env']
            if sample['cudnn_benchmark'] is not False:
                raise RuntimeError(f'cudnn.benchmark was {sample["cudnn_benchmark"]}')
    rows = [_summarize(by_model[spec['name']]) for spec in MODELS]
    _write_csv(rows, env)
    _print_markdown(rows, env)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        _run_worker(int(sys.argv[2]))
    else:
        main()
