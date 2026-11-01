# Reproduce paper tables

All VoD models: **80 epochs**, batch size **8**, seeds **0–3**. Evaluate epochs **51–80**; report best (R11 entire), final (80), mean of 76–80. Both R11 and R40.

Paper tables use the **training-time evaluation protocol with `--batch_size 8`**. Example (VoD 4D-MambaNet seed 0, epoch 78):

```bash
cd tools
python test.py --cfg_file cfgs/vod/4d_mambanet.yaml \
  --batch_size 8 --workers 4 \
  --ckpt ../checkpoints/vod_4d_mambanet_seed0_ep78_weights.pth
```

Expected: **R11 ≈ 56.226**. For M3-Net repro seed 0 epoch 78 with the same batch size, expect **R11 ≈ 55.168** (M3-Net re-evaluation can vary; see note below).

### Batch size 1 reference

`--batch_size 1` is useful for debugging. Numbers differ slightly from the paper (bs=8) tables:

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean R11 / R40 |
|---|---|---|---|---|---|
| 4D-MambaNet | 56.235 / 54.066 | 56.222 / 54.036 | 56.152 / 53.979 | 55.830 / 53.636 | 56.11 / 53.93 |
| M3-Net repro | 55.197 / 52.709 | 51.864 / 51.255 | 52.568 / 51.684 | 55.104 / 52.655 | 53.68 / 52.08 |

R11 is sensitive near the maximum recall of a class, so results with batch size 1 and 8 differ slightly; the gain of 4D-MambaNet holds under both.

Note on the Radar M3-Net baseline: its PTv3 stage uses a random serialization order also at inference (shuffle_orders=True). Because R11 samples precision at only 11 recall points, repeated evaluations of the same checkpoint can change R11 noticeably for some seeds (in our re-evaluation by up to about 2 mAP). The paper reports the training-time evaluation (best epoch) for all models; a lower re-evaluated baseline only increases the gain. 4D-MambaNet does not use PTv3 and gives identical results in every evaluation.

## One command per experiment (train)

From `tools/`:

```bash
bash scripts/train_4seeds.sh cfgs/vod/4d_mambanet.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/m3net_repro.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/4d_mambanet_no_rcs_input.yaml 0
# Table 4–6 ablations
bash scripts/train_4seeds.sh cfgs/vod/ablation_scan/per_frame_native.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_scan/shuffle.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_scan/hilbert.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_scan/bidirectional.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_attention/global_attention.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_placement/xcout.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_placement/xc4_xcout.yaml 0
# Table 7 ablation (RCS injection stage)
bash scripts/train_4seeds.sh cfgs/vod/ablation_rcs_bev/4d_mambanet_bev_prior.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_rcs_bev/no_rcs_input_bev_prior.yaml 0
bash scripts/train_4seeds.sh cfgs/vod/ablation_rcs_bev/m3net_bev_prior.yaml 0
# TJ4D
bash scripts/train_4seeds.sh cfgs/tj4d/4d_mambanet.yaml 0
bash scripts/train_4seeds.sh cfgs/tj4d/m3net_repro.yaml 0
```

## Evaluate a released checkpoint

```bash
cd tools
python test.py --cfg_file cfgs/vod/4d_mambanet.yaml \
  --batch_size 8 --workers 4 \
  --ckpt ../checkpoints/vod_4d_mambanet_seed0_ep78_weights.pth
```

Helper (defaults to batch size 1 — override as needed):

```bash
bash scripts/eval_checkpoint.sh cfgs/vod/4d_mambanet.yaml \
  ../checkpoints/vod_4d_mambanet_seed0_ep78_weights.pth 0
```

## Summarize seeds / t-test from logs

```bash
bash scripts/reproduce_table.sh table1
# or
python tools/analysis/summarize_seeds.py
```

Expected numbers: [../results/expected_results.md](../results/expected_results.md).

## Analysis tools

| Script | Role |
|---|---|
| `tools/analysis/summarize_seeds.py` | best / final / last-5, mean±std, paired t-test |
| `tools/analysis/compute_r40.py` | R40 from `result.pkl` |
| `tools/analysis/ap_by_distance.py` | AP by forward distance |
| `tools/analysis/eval_sparsity.py` | 25% / 50% point removal, shared mask, bs=1, R40 |
| `tools/analysis/bench_fps.py` | FPS / memory |
| `tools/analysis/bench_block_scaling.py` | Mamba / PTv3 / attention vs. #voxels |
