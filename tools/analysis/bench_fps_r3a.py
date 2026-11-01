#!/usr/bin/env python3
"""R3-A forward-only FPS on the full VoD val split.

Does not train and does not edit model code. Refuses to start unless
CUDA_VISIBLE_DEVICES is exactly 1 (GPU 0 is reserved for training).

Models (seed 0), each loaded from its best checkpoint under
output/rev1/<EXP>/seed0/ckpt/. Best epoch is the highest entire-area 3D mAP
in docs/rev1/per_epoch_vod.csv; ties keep the earliest epoch.

  E0_per_frame   per-frame Mamba, native scan
  E1_shuffle     per-frame Mamba, shuffled scan
  E1_hilbert     per-frame Mamba, Hilbert scan
  E1_bidir       per-frame bidirectional Mamba, native scan
  E3_attention   per-frame global attention

Forward-only: the batch is already on GPU, and box decode / NMS /
post_processing are stubbed for the timer (same boundary as bench_fps
network-only). Backbone and head convolutions still run. Data loading is
outside the timer.

The timed pass covers every val frame. An untimed warm-up of 10% of the
frames (at least 20) runs first and is not removed from the denominator.
Three repeats, each in a fresh process. mean ± sample std (ddof=1).

  CUDA_VISIBLE_DEVICES=1 python tools/rev1/bench_fps_r3a.py
"""
import csv
import json
import os
import subprocess
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
    sys.stderr.write(
        'Refusing to run: CUDA_VISIBLE_DEVICES must be 1. GPU 0 is training.\n'
    )
    sys.exit(2)

import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

PER_EPOCH_CSV = ROOT / 'docs' / 'rev1' / 'per_epoch_vod.csv'
CSV_PATH = ROOT / 'docs' / 'rev1' / 'fps_forward_r3a.csv'
REPEATS = 3
BATCH_SIZE = 1
NUM_WORKERS = 4
WARMUP_FRAC = 0.10
MIN_WARMUP = 20

EXPERIMENTS = (
    {
        'exp': 'E0_per_frame',
        'label': 'E0_per_frame s0 (native)',
        'cfg': 'tools/cfgs/rev1/E0_per_frame.yaml',
        'expect': {
            'per_frame_sequence': True,
            'scan_order': 'native',
            'bidirectional': False,
            'mixer': 'mamba',
        },
    },
    {
        'exp': 'E1_shuffle',
        'label': 'E1_shuffle s0',
        'cfg': 'tools/cfgs/rev1/E1_shuffle.yaml',
        'expect': {
            'per_frame_sequence': True,
            'scan_order': 'shuffle',
            'bidirectional': False,
            'mixer': 'mamba',
        },
    },
    {
        'exp': 'E1_hilbert',
        'label': 'E1_hilbert s0',
        'cfg': 'tools/cfgs/rev1/E1_hilbert.yaml',
        'expect': {
            'per_frame_sequence': True,
            'scan_order': 'hilbert',
            'bidirectional': False,
            'mixer': 'mamba',
        },
    },
    {
        'exp': 'E1_bidir',
        'label': 'E1_bidir s0',
        'cfg': 'tools/cfgs/rev1/E1_bidir.yaml',
        'expect': {
            'per_frame_sequence': True,
            'scan_order': 'native',
            'bidirectional': True,
            'mixer': 'mamba',
        },
    },
    {
        'exp': 'E3_attention',
        'label': 'E3_attention s0',
        'cfg': 'tools/cfgs/rev1/E3_attention.yaml',
        'expect': {
            'per_frame_sequence': True,
            'scan_order': 'native',
            'bidirectional': False,
            'mixer': 'attention',
        },
    },
)

FIELDS = [
    'model', 'exp', 'seed', 'cfg', 'ckpt', 'best_epoch', 'best_entire_mAP',
    'split', 'frames', 'warmup_frames', 'measured_frames', 'repeats', 'num_workers',
    'fps_mean', 'fps_std',
    'fps_r1', 'fps_r2', 'fps_r3',
    'peak_gpu_mem_bytes', 'peak_gpu_mem_mib',
    'peak_gpu_mem_bytes_r1', 'peak_gpu_mem_bytes_r2', 'peak_gpu_mem_bytes_r3',
    'params_total', 'params_active', 'dtype',
    'torch', 'cuda', 'cudnn', 'cudnn_benchmark', 'cudnn_enabled', 'gpu', 'gpu_uuid',
    'cuda_visible_devices', 'amp',
]


def _norm_uuid(value):
    text = str(value).strip().lower().replace('-', '')
    if text.startswith('gpu'):
        text = text[3:]
    return text


def assert_physical_gpu1():
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(f'expected 1 visible GPU, found {torch.cuda.device_count()}')
    want = subprocess.check_output(
        ['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True,
    ).strip()
    got = torch.cuda.get_device_properties(0).uuid
    if _norm_uuid(got) != _norm_uuid(want):
        raise SystemExit(
            f'visible GPU uuid {got} is not nvidia-smi GPU 1 ({want}). Refusing to run.'
        )
    print(
        f'visible device is nvidia-smi GPU 1 uuid={want} name={torch.cuda.get_device_name(0)}',
        flush=True,
    )
    return want


def best_checkpoint(exp):
    """Earliest epoch with the highest logged entire-area mAP for <exp>_s0."""
    model_key = f'{exp}_s0'
    best_ap = None
    best_epoch = None
    with PER_EPOCH_CSV.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['model'] != model_key:
                continue
            ap = float(row['entire_mAP'])
            epoch = int(row['epoch'])
            if best_ap is None or ap > best_ap or (ap == best_ap and epoch < best_epoch):
                best_ap = ap
                best_epoch = epoch
    if best_epoch is None:
        raise FileNotFoundError(f'no {model_key} rows in {PER_EPOCH_CSV}')
    ckpt = ROOT / 'output' / 'rev1' / exp / 'seed0' / 'ckpt' / f'checkpoint_epoch_{best_epoch}.pth'
    if not ckpt.is_file():
        raise FileNotFoundError(f'missing best checkpoint {ckpt} (entire mAP {best_ap:.4f})')
    return ckpt, best_epoch, best_ap


def resolved_models():
    specs = []
    for spec in EXPERIMENTS:
        ckpt, epoch, ap = best_checkpoint(spec['exp'])
        item = dict(spec)
        item['ckpt'] = str(ckpt.relative_to(ROOT))
        item['best_epoch'] = epoch
        item['best_entire_mAP'] = ap
        item['seed'] = 0
        specs.append(item)
    return specs


def _stub_anchor_decode(batch_size, cls_preds, box_preds, dir_cls_preds=None):
    return (
        cls_preds.new_zeros((batch_size, 1, cls_preds.shape[-1])),
        box_preds.new_zeros((batch_size, 1, box_preds.shape[-1])),
    )


def _stub_voxelnext_decode(batch_size, pred_dicts, voxel_indices, spatial_shape, points, add_features=None):
    import torch
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
    import torch
    called = set()
    hooks = []

    def _hook(mod, _inp, _out):
        called.add(id(mod))

    for mod in model.modules():
        hooks.append(mod.register_forward_hook(_hook))
    with torch.no_grad():
        model(batch)
    for hook in hooks:
        hook.remove()

    seen = set()
    active = 0
    for mod in model.modules():
        if id(mod) not in called:
            continue
        for param in mod.parameters(recurse=False):
            if id(param) in seen:
                continue
            seen.add(id(param))
            active += param.numel()
    total = sum(param.numel() for param in model.parameters())
    return total, active


def _assert_variant(model, spec):
    blocks = [mod for mod in model.modules() if type(mod).__name__ == 'MambaVoxelBlock']
    if len(blocks) != 3:
        raise RuntimeError(f'{spec["exp"]} has {len(blocks)} MambaVoxelBlocks, expected 3 (xc3/xc4/xcout)')
    for block in blocks:
        for key, value in spec['expect'].items():
            got = getattr(block, key)
            if got != value:
                raise RuntimeError(f'{spec["exp"]} MambaVoxelBlock {key}={got!r}, expected {value!r}')


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


def _precision_label(model):
    dtypes = sorted({str(param.dtype).replace('torch.', '') for param in model.parameters()})
    return ','.join(dtypes) if dtypes else 'none'


def _env_info(gpu_uuid):
    import torch
    return {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': str(torch.backends.cudnn.version()),
        'cudnn_benchmark': str(bool(torch.backends.cudnn.benchmark)),
        'cudnn_enabled': str(bool(torch.backends.cudnn.enabled)),
        'gpu': torch.cuda.get_device_name(0),
        'gpu_uuid': gpu_uuid,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'precision': 'fp32' if not torch.is_autocast_enabled() else 'autocast',
    }


def _warmup(model, loader, n_warm):
    import torch
    from pcdet.models import load_data_to_gpu
    seen = 0
    with _SkipPostprocess(model):
        for batch in loader:
            load_data_to_gpu(batch)
            with torch.no_grad():
                model(batch)
            seen += int(batch['batch_size'])
            if seen >= n_warm:
                break
    torch.cuda.synchronize()
    return seen


def _measure_full(model, loader):
    import torch
    from pcdet.models import load_data_to_gpu
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = 0.0
    measured = 0
    with _SkipPostprocess(model):
        for batch in loader:
            load_data_to_gpu(batch)
            n_frames = int(batch['batch_size'])
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - t0
            measured += n_frames
            if measured % 256 == 0:
                print(f'    forward {measured} frames', flush=True)
    peak = int(torch.cuda.max_memory_allocated())
    return measured, elapsed, peak


def run_one(spec, logger, gpu_uuid):
    import torch
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
    if str(split_name) != 'val':
        raise RuntimeError(f'{spec["exp"]} DATA_SPLIT test is {split_name}, expected val')

    print(f'\n===== {spec["label"]} =====', flush=True)
    print(f'cfg:  {cfg_path}', flush=True)
    print(f'ckpt: {ckpt_path}', flush=True)
    print(
        f"best epoch {spec['best_epoch']}  entire mAP {spec['best_entire_mAP']:.4f}",
        flush=True,
    )
    print(f"DATA_SPLIT['test'] = {split_name}", flush=True)

    dataset, loader = _make_loader(cfg, logger)
    n_frames = len(dataset)
    warmup = max(MIN_WARMUP, int(n_frames * WARMUP_FRAC))
    print(f'val frames = {n_frames}  untimed warmup = {warmup}  measured = {n_frames}', flush=True)

    model = _load_model(cfg, dataset, ckpt_path, logger)
    _assert_variant(model, spec)
    precision = _precision_label(model)
    if precision != 'float32':
        raise RuntimeError(f'{spec["exp"]} dtype {precision}, expected float32')
    if torch.is_autocast_enabled():
        raise RuntimeError('autocast is enabled')
    torch.backends.cudnn.benchmark = False
    print(f'torch.backends.cudnn.benchmark = {torch.backends.cudnn.benchmark}', flush=True)

    from pcdet.models import load_data_to_gpu
    first = next(iter(loader))
    load_data_to_gpu(first)
    total_params, active_params = _count_active_params(model, first)
    print(f'params total = {total_params}  active = {active_params}  dtype = {precision}', flush=True)
    del first, loader
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    _dataset, warm_loader = _make_loader(cfg, logger)
    if len(_dataset) != n_frames:
        raise RuntimeError(f'frame count changed: {len(_dataset)} vs {n_frames}')
    warmed = _warmup(model, warm_loader, warmup)
    if warmed < warmup:
        raise RuntimeError(f'warmup saw {warmed} frames, need {warmup}')
    del warm_loader
    torch.cuda.empty_cache()

    _dataset, timed_loader = _make_loader(cfg, logger)
    if len(_dataset) != n_frames:
        raise RuntimeError(f'frame count changed: {len(_dataset)} vs {n_frames}')
    measured, elapsed, peak = _measure_full(model, timed_loader)
    if measured != n_frames:
        raise RuntimeError(f'forward measured {measured}, expected {n_frames}')
    fps = measured / elapsed
    print(
        f'  forward {fps:.3f} FPS over {measured} frames  '
        f'peak {peak / (1024 ** 2):.1f} MiB',
        flush=True,
    )
    return {
        'model': spec['label'],
        'exp': spec['exp'],
        'seed': spec['seed'],
        'cfg': spec['cfg'],
        'ckpt': spec['ckpt'],
        'best_epoch': spec['best_epoch'],
        'best_entire_mAP': spec['best_entire_mAP'],
        'split': split_name,
        'frames': n_frames,
        'warmup_frames': warmup,
        'measured_frames': measured,
        'fps_forward': fps,
        'peak_gpu_mem_bytes': peak,
        'params_total': total_params,
        'params_active': active_params,
        'dtype': precision,
        'num_workers': NUM_WORKERS,
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'env': _env_info(gpu_uuid),
    }


def _mean_std(values):
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _summarize(samples):
    fps_values = [sample['fps_forward'] for sample in samples]
    fps_mean, fps_std = _mean_std(fps_values)
    peak = max(sample['peak_gpu_mem_bytes'] for sample in samples)
    base = samples[0]
    for sample in samples[1:]:
        if sample['params_total'] != base['params_total'] or sample['params_active'] != base['params_active']:
            raise RuntimeError(f'parameter count changed across repeats for {base["model"]}')
    row = {
        'model': base['model'],
        'exp': base['exp'],
        'seed': base['seed'],
        'cfg': base['cfg'],
        'ckpt': base['ckpt'],
        'best_epoch': base['best_epoch'],
        'best_entire_mAP': base['best_entire_mAP'],
        'split': base['split'],
        'frames': base['frames'],
        'warmup_frames': base['warmup_frames'],
        'measured_frames': base['measured_frames'],
        'repeats': len(samples),
        'num_workers': base['num_workers'],
        'fps_mean': fps_mean,
        'fps_std': fps_std,
        'peak_gpu_mem_bytes': peak,
        'peak_gpu_mem_mib': peak / (1024 ** 2),
        'params_total': base['params_total'],
        'params_active': base['params_active'],
        'dtype': base['dtype'],
        'cudnn_benchmark': base['cudnn_benchmark'],
    }
    for index, sample in enumerate(samples, start=1):
        row[f'fps_r{index}'] = sample['fps_forward']
        row[f'peak_gpu_mem_bytes_r{index}'] = sample['peak_gpu_mem_bytes']
    return row


def _write_csv(rows, env):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.update({
                'torch': env['torch'],
                'cuda': env['cuda'],
                'cudnn': env['cudnn'],
                'cudnn_benchmark': env['cudnn_benchmark'],
                'cudnn_enabled': env['cudnn_enabled'],
                'gpu': env['gpu'],
                'gpu_uuid': env['gpu_uuid'],
                'cuda_visible_devices': env['cuda_visible_devices'],
                'amp': env['precision'],
            })
            writer.writerow({key: out.get(key, '') for key in FIELDS})
    print(f'wrote {CSV_PATH}', flush=True)


def _print_markdown(rows, env):
    print('\nForward-only FPS (full VoD val, batch 1)', flush=True)
    print(f"- torch {env['torch']}", flush=True)
    print(f"- CUDA {env['cuda']}", flush=True)
    print(
        f"- cuDNN {env['cudnn']}  enabled={env['cudnn_enabled']}  benchmark={env['cudnn_benchmark']}",
        flush=True,
    )
    print(f"- GPU {env['gpu']}  uuid {env['gpu_uuid']}", flush=True)
    print('- CUDA_VISIBLE_DEVICES=1 (nvidia-smi GPU 1)', flush=True)
    print(f"- precision {env['precision']} (parameter dtype; no autocast)", flush=True)
    print(
        f'- batch size {BATCH_SIZE}  repeats {REPEATS}  '
        f'untimed warmup {WARMUP_FRAC:.0%} of frames (min {MIN_WARMUP}), not removed from the split',
        flush=True,
    )
    print('- FPS std is the sample std (ddof=1) across repeats', flush=True)
    print('- each repeat is a fresh process; order is forward, reversed, forward', flush=True)
    print(flush=True)
    print(
        '| Model | Best epoch | Frames | Forward FPS | Peak mem (MiB) | Params total | Params active | dtype |',
        flush=True,
    )
    print('|---|---:|---:|---:|---:|---:|---:|---|', flush=True)
    for row in rows:
        print(
            '| {model} | {epoch} | {frames} | {mean:.2f} ± {std:.2f} | {mem:.1f} | {total} | {active} | {dtype} |'.format(
                model=row['model'],
                epoch=row['best_epoch'],
                frames=row['frames'],
                mean=row['fps_mean'],
                std=row['fps_std'],
                mem=row['peak_gpu_mem_mib'],
                total=row['params_total'],
                active=row['params_active'],
                dtype=row['dtype'],
            ),
            flush=True,
        )


def _repeat_orders(n_models):
    order = list(range(n_models))
    return [list(order), list(reversed(order)), list(order)]


def _run_worker(model_index):
    import torch
    from pcdet.config_ext import apply_yaml_ext
    apply_yaml_ext()
    torch.backends.cudnn.benchmark = False
    gpu_uuid = assert_physical_gpu1()
    print('torch', torch.__version__, flush=True)
    print('CUDA', torch.version.cuda, flush=True)
    print('cuDNN', torch.backends.cudnn.version(), 'benchmark', torch.backends.cudnn.benchmark, flush=True)
    print('GPU', torch.cuda.get_device_name(0), flush=True)
    print('autocast', torch.is_autocast_enabled(), flush=True)
    specs = resolved_models()
    logger = logging.getLogger('bench_fps_r3a')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    result = run_one(specs[model_index], logger, gpu_uuid)
    print('BENCH_JSON ' + json.dumps(result), flush=True)


def _launch_worker(model_index, repeat, spec):
    print(
        f'\n######## repeat {repeat}  {spec["label"]}  (subprocess, CUDA_VISIBLE_DEVICES=1) ########',
        flush=True,
    )
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
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
        raise RuntimeError(f'worker failed for {spec["label"]} repeat {repeat} (exit {code})')
    payload['repeat'] = repeat
    return payload


def main():
    specs = resolved_models()
    for spec in specs:
        print(
            f'{spec["label"]}: epoch {spec["best_epoch"]}  '
            f'entire mAP {spec["best_entire_mAP"]:.4f}  ckpt {spec["ckpt"]}',
            flush=True,
        )
    print('=== nvidia-smi before sweep (no CUDA init in this process) ===', flush=True)
    subprocess.run(['nvidia-smi'], check=False)
    by_model = {spec['label']: [] for spec in specs}
    env = None
    finished = []
    for repeat, order in enumerate(_repeat_orders(len(specs)), start=1):
        names = [specs[index]['label'] for index in order]
        print(f'\n===== outer repeat {repeat}/{REPEATS} order: {names} =====', flush=True)
        for model_index in order:
            sample = _launch_worker(model_index, repeat, specs[model_index])
            by_model[sample['model']].append(sample)
            env = sample['env']
            if sample['cudnn_benchmark'] is not False:
                raise RuntimeError(f'cudnn.benchmark was {sample["cudnn_benchmark"]}')
            done = [spec['label'] for spec in specs if len(by_model[spec['label']]) == REPEATS]
            if done != finished:
                finished = done
                _write_csv([_summarize(by_model[label]) for label in finished], env)
    rows = [_summarize(by_model[spec['label']]) for spec in specs]
    _write_csv(rows, env)
    _print_markdown(rows, env)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        _run_worker(int(sys.argv[2]))
    else:
        main()
