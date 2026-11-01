#!/usr/bin/env python3
"""Microbench: one attention mixer vs one Mamba mixer (forward time + peak memory).

Does not train. Run manually, e.g.:
  python tools/rev1/bench_mixer.py

Reports d_model in {64, 128} and N in {2k, 8k, 16k, 32k} voxels (batch_size=1).
OOM is caught and recorded; other errors abort that cell.
"""
import sys
import time
from pathlib import Path

import torch
import spconv.pytorch as spconv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pcdet.models.backbones_3d.mamba_voxel_block import MambaVoxelBlock

D_MODELS = (64, 128)
N_VOXELS = (2000, 8000, 16000, 32000)
WARMUP = 5
ITERS = 20
SPATIAL_SHAPE = [200, 200, 20]


def make_sparse(n, channels, device):
    g = torch.Generator(device='cpu')
    g.manual_seed(0)
    batch = torch.zeros(n, 1, dtype=torch.int32)
    z = torch.randint(0, SPATIAL_SHAPE[0], (n, 1), generator=g, dtype=torch.int32)
    y = torch.randint(0, SPATIAL_SHAPE[1], (n, 1), generator=g, dtype=torch.int32)
    x = torch.randint(0, SPATIAL_SHAPE[2], (n, 1), generator=g, dtype=torch.int32)
    indices = torch.cat([batch, z, y, x], dim=1)
    feats = torch.randn(n, channels, generator=g)
    return spconv.SparseConvTensor(
        features=feats.to(device),
        indices=indices.to(device),
        spatial_shape=list(SPATIAL_SHAPE),
        batch_size=1,
    )


def clone_sparse(x):
    return spconv.SparseConvTensor(
        features=x.features.detach().clone(),
        indices=x.indices.clone(),
        spatial_shape=list(x.spatial_shape),
        batch_size=x.batch_size,
    )


@torch.no_grad()
def bench_one(block, x, device):
    """Return (mean_ms, peak_mem_bytes) or raise torch.cuda.OutOfMemoryError."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for _ in range(WARMUP):
        _ = block(clone_sparse(x))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = block(clone_sparse(x))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    mean_ms = 1000.0 * elapsed / ITERS
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return mean_ms, peak


def main():
    if not torch.cuda.is_available():
        raise SystemExit('bench_mixer.py requires CUDA')
    device = torch.device('cuda')

    print(
        f'{"mixer":<12} {"d_model":>7} {"N":>7} {"ms":>10} {"peak_MiB":>10}'
    )
    print('-' * 50)

    for d_model in D_MODELS:
        for mixer in ('mamba', 'attention'):
            torch.manual_seed(0)
            block = MambaVoxelBlock(d_model=d_model, mixer=mixer).to(device).eval()
            for n in N_VOXELS:
                x = make_sparse(n, d_model, device)
                try:
                    mean_ms, peak = bench_one(block, x, device)
                    print(
                        f'{mixer:<12} {d_model:>7} {n:>7} {mean_ms:>10.3f} '
                        f'{peak / (1024 ** 2):>10.1f}'
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f'{mixer:<12} {d_model:>7} {n:>7} {"OOM":>10} {"OOM":>10}')
                except Exception as exc:
                    torch.cuda.empty_cache()
                    print(f'{mixer:<12} {d_model:>7} {n:>7} ERR:{type(exc).__name__}')
            del block
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
