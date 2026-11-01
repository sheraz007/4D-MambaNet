#!/usr/bin/env python3
"""R3-A mixer scaling: Mamba vs global attention vs M3-Net PTv3 at xc4 width.

Does not train and does not edit model code. Refuses to start unless
CUDA_VISIBLE_DEVICES is exactly 1 (GPU 0 is reserved for training).

One stage module, batch 1, fp32, N in {1k, 2k, 4k, 8k, 16k, 32k, 64k}:
  mamba      MambaVoxelBlock d_model=128 mixer=mamba
             (the xc4 Mamba block; xc4 channels are 128)
  attention  MambaVoxelBlock d_model=128 mixer=attention
             (global MHSA over the frame, the E3 mixer)
  ptv3       VoxelAttnPTv3 used by the M3-Net baseline at xcout, which is the
             same stage size as the xc4 slot:
             enc_depths=(2,), enc_channels=(128,), enc_num_head=(8,),
             enc_patch_size=(48,), enable_flash=False

Forward latency is the median of 100 synchronized runs after 20 warm-up runs.
Peak memory is torch.cuda.max_memory_allocated over those timed runs.
OOM is recorded and the sweep continues (each cell is its own process).

  CUDA_VISIBLE_DEVICES=1 python tools/rev1/bench_mixer_r3a.py
"""
import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
    sys.stderr.write(
        'Refusing to run: CUDA_VISIBLE_DEVICES must be 1. GPU 0 is training.\n'
    )
    sys.exit(2)

import csv
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / 'docs' / 'rev1' / 'mixer_scaling_r3a.csv'
D_MODEL = 128
N_VOXELS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
MIXERS = ('mamba', 'attention', 'ptv3')
WARMUP = 20
ITERS = 100
SPATIAL_SHAPE = [200, 200, 20]
# M3-Net baseline calls voxel_attn[2] (xcout). That module matches the xc4 slot.
PTV3_SPEC = dict(
    enc_depths=(2,),
    enc_channels=(128,),
    enc_num_head=(8,),
    enc_patch_size=(48,),
    enable_flash=False,
    indice_key='ptv3_4',
)

FIELDS = [
    'mixer', 'd_model', 'n_blocks', 'num_heads', 'patch_size', 'enable_flash',
    'N', 'batch_size', 'dtype', 'warmup', 'iters', 'median_ms', 'mean_ms',
    'peak_bytes', 'peak_mib', 'params', 'status',
    'spatial_shape', 'torch', 'cuda', 'cudnn', 'cudnn_benchmark',
    'allow_tf32_matmul', 'allow_tf32_cudnn', 'gpu', 'gpu_uuid',
    'cuda_visible_devices', 'notes',
]


def _norm_uuid(value):
    text = str(value).strip().lower().replace('-', '')
    if text.startswith('gpu'):
        text = text[3:]
    return text


def assert_physical_gpu1():
    """Abort unless the only visible CUDA device is nvidia-smi GPU 1."""
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f'expected 1 visible GPU, found {torch.cuda.device_count()}'
        )
    want = subprocess.check_output(
        ['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True,
    ).strip()
    got = torch.cuda.get_device_properties(0).uuid
    if _norm_uuid(got) != _norm_uuid(want):
        raise SystemExit(
            f'visible GPU uuid {got} is not nvidia-smi GPU 1 ({want}). '
            'Refusing to run.'
        )
    print(
        f'visible device is nvidia-smi GPU 1 uuid={want} '
        f'name={torch.cuda.get_device_name(0)}',
        flush=True,
    )
    return want


def _env_info(gpu_uuid):
    import torch
    return {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': str(torch.backends.cudnn.version()),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'gpu': torch.cuda.get_device_name(0),
        'gpu_uuid': gpu_uuid,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    }


def make_sparse(n, channels, device):
    import torch
    import spconv.pytorch as spconv
    volume = SPATIAL_SHAPE[0] * SPATIAL_SHAPE[1] * SPATIAL_SHAPE[2]
    if n > volume:
        raise ValueError(f'N={n} exceeds spatial volume {volume}')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(0)
    pick = torch.randperm(volume, generator=generator)[:n]
    stride_yx = SPATIAL_SHAPE[1] * SPATIAL_SHAPE[2]
    z = torch.div(pick, stride_yx, rounding_mode='floor')
    rem = pick % stride_yx
    y = torch.div(rem, SPATIAL_SHAPE[2], rounding_mode='floor')
    x = rem % SPATIAL_SHAPE[2]
    indices = torch.stack([
        torch.zeros(n, dtype=torch.int32),
        z.to(torch.int32),
        y.to(torch.int32),
        x.to(torch.int32),
    ], dim=1)
    feats = torch.randn(n, channels, generator=generator)
    return spconv.SparseConvTensor(
        features=feats.to(device),
        indices=indices.to(device),
        spatial_shape=list(SPATIAL_SHAPE),
        batch_size=1,
    )


def clone_sparse(x):
    import spconv.pytorch as spconv
    return spconv.SparseConvTensor(
        features=x.features.detach().clone(),
        indices=x.indices.clone(),
        spatial_shape=list(x.spatial_shape),
        batch_size=x.batch_size,
    )


def build_block(mixer, device):
    import torch
    torch.manual_seed(0)
    if mixer == 'mamba':
        from pcdet.models.backbones_3d.mamba_voxel_block import MambaVoxelBlock
        block = MambaVoxelBlock(d_model=D_MODEL, mixer='mamba')
        meta = dict(
            n_blocks=1, num_heads='', patch_size='', enable_flash='',
            notes='MambaVoxelBlock d_model=128 mixer=mamba (xc4 channel width)',
        )
    elif mixer == 'attention':
        from pcdet.models.backbones_3d.mamba_voxel_block import MambaVoxelBlock
        block = MambaVoxelBlock(d_model=D_MODEL, mixer='attention')
        meta = dict(
            n_blocks=1, num_heads=int(block.num_heads), patch_size='', enable_flash='',
            notes=(
                f'MambaVoxelBlock d_model=128 mixer=attention global MHSA '
                f'd_inner={block.d_inner} heads={block.num_heads} (xc4 width)'
            ),
        )
    elif mixer == 'ptv3':
        from pcdet.models.sparse_attention.point_transformer_v3 import VoxelAttnPTv3
        block = VoxelAttnPTv3(**PTV3_SPEC)
        meta = dict(
            n_blocks=int(PTV3_SPEC['enc_depths'][0]),
            num_heads=int(PTV3_SPEC['enc_num_head'][0]),
            patch_size=int(PTV3_SPEC['enc_patch_size'][0]),
            enable_flash=bool(PTV3_SPEC['enable_flash']),
            notes=(
                'VoxelAttnPTv3 M3-Net xcout/xc4 stage: enc_depths=2, '
                'enc_channels=128, enc_num_head=8, enc_patch_size=48, '
                'enable_flash=False, upcast_attention=False, upcast_softmax=False'
            ),
        )
    else:
        raise ValueError(mixer)
    block = block.to(device).eval()
    params = sum(p.numel() for p in block.parameters())
    dtypes = {str(p.dtype) for p in block.parameters()}
    if dtypes != {'torch.float32'}:
        raise RuntimeError(f'{mixer} parameter dtypes {dtypes}, expected fp32')
    meta['params'] = params
    return block, meta


def _is_oom(exc):
    name = type(exc).__name__
    text = str(exc).lower()
    return name == 'OutOfMemoryError' or 'out of memory' in text


def bench_forward(block, sample, device, warmup, iters):
    import torch
    with torch.no_grad():
        for i in range(warmup):
            out = block(clone_sparse(sample))
            del out
            if (i + 1) % 10 == 0:
                print(f'  warmup {i + 1}/{warmup}', flush=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples_ms = []
        for i in range(iters):
            inp = clone_sparse(sample)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = block(inp)
            torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
            del out, inp
            if (i + 1) % 20 == 0:
                print(f'  iter {i + 1}/{iters}', flush=True)
    median_ms = float(statistics.median(samples_ms))
    mean_ms = float(statistics.fmean(samples_ms))
    peak = int(torch.cuda.max_memory_allocated(device))
    return median_ms, mean_ms, peak


def run_cell(mixer, n, warmup, iters):
    import torch
    torch.backends.cudnn.benchmark = False
    gpu_uuid = assert_physical_gpu1()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    env = _env_info(gpu_uuid)
    print(
        f'torch {env["torch"]}  CUDA {env["cuda"]}  cuDNN {env["cudnn"]}  '
        f'benchmark {env["cudnn_benchmark"]}  '
        f'tf32 matmul {env["allow_tf32_matmul"]} cudnn {env["allow_tf32_cudnn"]}',
        flush=True,
    )
    print(f'cell mixer={mixer} N={n} warmup={warmup} iters={iters} fp32', flush=True)
    row = {
        'mixer': mixer,
        'd_model': D_MODEL,
        'N': int(n),
        'batch_size': 1,
        'dtype': 'float32',
        'warmup': int(warmup),
        'iters': int(iters),
        'median_ms': '',
        'mean_ms': '',
        'peak_bytes': '',
        'peak_mib': '',
        'status': 'ok',
        'spatial_shape': 'x'.join(str(v) for v in SPATIAL_SHAPE),
        'notes': '',
        'n_blocks': '',
        'num_heads': '',
        'patch_size': '',
        'enable_flash': '',
        'params': '',
    }
    row.update(env)
    try:
        if torch.is_autocast_enabled():
            raise RuntimeError('autocast is enabled; fp32 bench requires it off')
        block, meta = build_block(mixer, device)
        row.update(meta)
        sample = make_sparse(int(n), D_MODEL, device)
        if sample.features.dtype != torch.float32:
            raise RuntimeError(f'feature dtype {sample.features.dtype}')
        median_ms, mean_ms, peak = bench_forward(block, sample, device, warmup, iters)
        row['median_ms'] = median_ms
        row['mean_ms'] = mean_ms
        row['peak_bytes'] = peak
        row['peak_mib'] = peak / (1024 ** 2)
        print(
            f'RESULT {mixer} N={n} median_ms={median_ms:.4f} '
            f'peak_mib={row["peak_mib"]:.2f} params={row["params"]}',
            flush=True,
        )
    except Exception as exc:
        torch.cuda.empty_cache()
        if _is_oom(exc):
            row['status'] = 'OOM'
            print(f'RESULT {mixer} N={n} OOM', flush=True)
        else:
            traceback.print_exc()
            row['status'] = 'ERR:' + type(exc).__name__
            row['notes'] = (row['notes'] + ' | ' if row['notes'] else '') + str(exc)[:400]
            print(f'RESULT {mixer} N={n} {row["status"]}', flush=True)
    print('CELL_JSON ' + json.dumps(row), flush=True)


def _blank_row(mixer, n, status, note):
    row = {key: '' for key in FIELDS}
    row.update({
        'mixer': mixer,
        'd_model': D_MODEL,
        'N': n,
        'batch_size': 1,
        'dtype': 'float32',
        'warmup': WARMUP,
        'iters': ITERS,
        'status': status,
        'spatial_shape': 'x'.join(str(v) for v in SPATIAL_SHAPE),
        'notes': note,
        'cuda_visible_devices': '1',
    })
    return row


def _launch_cell(mixer, n):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        '--cell', mixer, str(n),
        '--warmup', str(WARMUP),
        '--iters', str(ITERS),
    ]
    print(f'\n##### {mixer} N={n} #####', flush=True)
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
    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
        if line.startswith('CELL_JSON '):
            payload = json.loads(line[len('CELL_JSON '):])
    code = proc.wait()
    if payload is not None:
        return payload
    text = ''.join(captured)
    lowered = text.lower()
    if 'out of memory' in lowered or 'outofmemory' in lowered or code < 0:
        print(f'cell process exited {code} without JSON; recording OOM', flush=True)
        return _blank_row(mixer, n, 'OOM', f'process exit {code}')
    tail = text[-800:].replace('\n', ' ')
    print(f'cell process exited {code} without JSON; recording error', flush=True)
    return _blank_row(mixer, n, f'ERR:exit{code}', tail)


def _write_csv(rows):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in FIELDS})
    print(f'wrote {CSV_PATH}', flush=True)


def _fmt_ms(row):
    if row.get('status') != 'ok':
        return row.get('status', '')
    return f'{float(row["median_ms"]):.3f}'


def _fmt_mib(row):
    if row.get('status') != 'ok':
        return row.get('status', '')
    return f'{float(row["peak_mib"]):.1f}'


def _print_markdown(rows):
    env = next((row for row in rows if row.get('torch')), rows[0] if rows else {})
    print('\nMixer scaling (xc4 width, batch 1, fp32)', flush=True)
    print(f"- torch {env.get('torch', '')}", flush=True)
    print(f"- CUDA {env.get('cuda', '')}", flush=True)
    print(
        f"- cuDNN {env.get('cudnn', '')}  benchmark={env.get('cudnn_benchmark', '')}",
        flush=True,
    )
    print(
        f"- tf32 matmul={env.get('allow_tf32_matmul', '')}  "
        f"cudnn={env.get('allow_tf32_cudnn', '')}",
        flush=True,
    )
    print(f"- GPU {env.get('gpu', '')}  uuid {env.get('gpu_uuid', '')}", flush=True)
    print('- CUDA_VISIBLE_DEVICES=1 (nvidia-smi GPU 1)', flush=True)
    print(
        f'- median of {ITERS} synchronized forwards after {WARMUP} warm-up; '
        'peak is max_memory_allocated over the timed runs',
        flush=True,
    )
    print(
        '- mamba / attention: one MambaVoxelBlock, d_model=128. '
        'ptv3: M3-Net VoxelAttnPTv3 stage (depth 2, 128 channels, 8 heads, patch 48, flash off)',
        flush=True,
    )
    print(flush=True)
    print('| Mixer | N | median latency (ms) | peak memory (MiB) | params | status |', flush=True)
    print('|---|---:|---:|---:|---:|---|', flush=True)
    order = {(mixer, n): i for i, (mixer, n) in enumerate(
        (mixer, n) for mixer in MIXERS for n in N_VOXELS
    )}
    for row in sorted(rows, key=lambda r: order.get((r['mixer'], int(r['N'])), 99)):
        params = row.get('params', '')
        print(
            f'| {row["mixer"]} | {int(row["N"])} | {_fmt_ms(row)} | {_fmt_mib(row)} | '
            f'{params} | {row.get("status", "")} |',
            flush=True,
        )


def run_sweep():
    print('=== nvidia-smi before sweep (no CUDA init in this process) ===', flush=True)
    subprocess.run(['nvidia-smi'], check=False)
    rows = []
    # Mamba and PTv3 first so a slow or OOM global-attention cell cannot block them.
    schedule = [(mixer, n) for mixer in ('mamba', 'ptv3', 'attention') for n in N_VOXELS]
    for mixer, n in schedule:
        rows.append(_launch_cell(mixer, n))
        _write_csv(rows)
    _print_markdown(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', nargs=2, metavar=('MIXER', 'N'))
    parser.add_argument('--warmup', type=int, default=WARMUP)
    parser.add_argument('--iters', type=int, default=ITERS)
    args = parser.parse_args()
    if args.cell:
        mixer, n_text = args.cell
        if mixer not in MIXERS:
            raise SystemExit(f'unknown mixer {mixer}')
        run_cell(mixer, int(n_text), args.warmup, args.iters)
        return
    if args.warmup != WARMUP or args.iters != ITERS:
        raise SystemExit('the sweep uses warmup=20 and iters=100; pass those only to --cell')
    run_sweep()


if __name__ == '__main__':
    main()
