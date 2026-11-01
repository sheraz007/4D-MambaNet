# Third-party notices

## OpenPCDet

This repository is derived from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache License 2.0).
See `LICENSE` for the full Apache-2.0 text.

## Radar M3-Net

4D-MambaNet is built on the Radar M3-Net framework (MSVFE / multi-scale VFE, temporal fusion via UniConv, sparse dual-head detection). Files taken from that codebase are marked with:

```text
# Based on Radar M3-Net (Y. Yang et al., Expert Systems with Applications, 2025). Released with permission.
```

The contribution of the authors is **MambaVoxelBlock** (Mamba state-space mixing on sparse voxels) and the RCS analysis / optional RCS-aware BEV ablation.

```bibtex
@article{yang2025radarm3net,
  author  = {Yang, Yunting and Liu, Jun and Liu, Huanyu and Jiang, Guangyu},
  title   = {Radar {M}$^3$-Net: Multi-scale, multi-layer, multi-frame network with a large receptive field for 3{D} object detection},
  journal = {Expert Systems with Applications},
  volume  = {286},
  pages   = {127515},
  year    = {2025}
}
```

## Point Transformer V3 / Pointcept

PTv3 attention stages and Point structures are adapted from [Pointcept](https://github.com/Pointcept/Pointcept) (MIT License). Original author headers are retained in the vendored files under `pcdet/models/backbones_3d/attn_layer/ptv3/` and `pcdet/models/model_utils/pointcept_utils/`.

Hilbert / z-order serialization utilities include code licensed under the **MIT License** (Peng-Shuai Wang; see header in `pcdet/models/model_utils/pointcept_utils/serialization/z_order.py`).

## Mamba (mamba-ssm)

Training and inference use [mamba-ssm](https://github.com/state-spaces/mamba) **v2.3.1** (Apache License 2.0) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) **v1.6.1** (BSD-3-Clause).
