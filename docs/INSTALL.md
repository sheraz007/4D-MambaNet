# Install

Use a CUDA **12.1** environment matching the paper runs (do not mix wheel versions).

## Environment (paper)

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| spconv-cu121 | 2.3.8 |
| mamba-ssm | 2.3.1 |
| causal-conv1d | 1.6.1 |
| OpenPCDet / pcdet | 0.6.0 (this tree) |

## Copy-paste install

```bash
# 1) Conda env
conda create -n mambanet4d python=3.10 -y
conda activate mambanet4d

# 2) PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Sparse convolution
pip install spconv-cu121==2.3.8

# 4) Mamba stack (prefer prebuilt wheels; building from source needs a matching CUDA toolkit)
pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1
# If the pip build fails, download official prebuilt wheels and install locally:
#   mamba-ssm v2.3.1: https://github.com/state-spaces/mamba/releases
#   causal-conv1d v1.6.1: https://github.com/Dao-AILab/causal-conv1d/releases
# Pick the wheel matching cu12, torch 2.5, cp310 and the cxx11abi of your torch build, then:
#   pip install <downloaded .whl>

# 5) Remaining Python deps + editable package
cd /path/to/4D-MambaNet
pip install -r requirements.txt
python setup.py develop
```

## Import check

```bash
python -c "import torch, spconv, mamba_ssm, pcdet; print(torch.__version__, spconv.__version__, mamba_ssm.__version__, pcdet.__version__)"
```

Expected torch string contains `2.5.1` and `cu121`; `mamba_ssm` reports `2.3.1`.

Next: [DATA.md](DATA.md), then [GETTING_STARTED.md](GETTING_STARTED.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `mamba_ssm` / `causal_conv1d` compile errors | Use **prebuilt** cu121 wheels for Python 3.10; avoid building from sdist without a full CUDA toolkit. |
| `spconv` CUDA mismatch | Reinstall `spconv-cu121==2.3.8` in the same env as `torch==2.5.1+cu121`. |
| `import pcdet` points elsewhere | Uninstall other editable OpenPCDet installs, or set `PYTHONPATH=/path/to/4D-MambaNet`. |
| `CUDA out of memory` on eval | Lower `--batch_size` (paper tables use 8; bs=1 works but numbers differ slightly — see [REPRODUCE.md](REPRODUCE.md)). |
| Missing CUDA ops (`.so`) | Run `python setup.py develop` after torch/spconv are installed so `pcdet/ops` extensions build. |
