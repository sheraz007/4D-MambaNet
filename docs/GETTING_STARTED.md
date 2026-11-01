# Getting started

OpenPCDet-style quick path from a fresh clone to a first evaluation.

## 1. Install

Follow [INSTALL.md](INSTALL.md) (Python 3.10, PyTorch 2.5.1+cu121, spconv 2.3.8, mamba-ssm 2.3.1).

## 2. Prepare data

Follow [DATA.md](DATA.md): download VoD (5-scan radar) and/or TJ4DRadSet, place them under `data/`, and generate the info pickles.

```bash
# Example overrides
export MAMBANET4D_DATA=/path/to/datasets
```

## 3. Download checkpoints (optional)

```bash
export RELEASE_URL=https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0
bash tools/scripts/download_checkpoints.sh
```

Verify with `checkpoints.sha256`. See [MODEL_ZOO.md](MODEL_ZOO.md).

## 4. Evaluate (paper protocol)

```bash
cd tools
python test.py --cfg_file cfgs/vod/4d_mambanet.yaml \
  --batch_size 8 --workers 4 \
  --ckpt ../checkpoints/vod_4d_mambanet_seed0_ep78_weights.pth
```

Expected VoD 4D-MambaNet seed 0: **R11 ≈ 56.226**. For batch-size notes and other experiments, see [REPRODUCE.md](REPRODUCE.md).

## 5. Train four seeds

```bash
bash tools/scripts/train_4seeds.sh cfgs/vod/4d_mambanet.yaml 0
```

Configs for every paper experiment live under `tools/cfgs/`. Summarize with:

```bash
python tools/analysis/summarize_seeds.py
```
