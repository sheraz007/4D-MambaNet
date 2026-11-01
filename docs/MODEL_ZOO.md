# Model zoo

Checkpoints are distributed via **GitHub Releases** (`v1.0`), not git.

```bash
export RELEASE_URL=https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0
bash tools/scripts/download_checkpoints.sh
```

Verify against `checkpoints.sha256` at the repo root. Release page: https://github.com/sheraz007/4D-MambaNet/releases/tag/v1.0

## VoD — best epoch by R11 entire-area mAP (seeds 0–3, eval batch size 8)

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean R11 |
|---|---|---|---|---|---|
| Radar M3-Net (repro.) | [ep78](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_m3net_seed0_ep78_weights.pth) | [ep61](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_m3net_seed1_ep61_weights.pth) | [ep73](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_m3net_seed2_ep73_weights.pth) | [ep73](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_m3net_seed3_ep73_weights.pth) | 54.65 ± 0.88 |
| **4D-MambaNet** | [ep78 (56.226)](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_4d_mambanet_seed0_ep78_weights.pth) | [ep79 (56.156)](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_4d_mambanet_seed1_ep79_weights.pth) | [ep80 (56.154)](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_4d_mambanet_seed2_ep80_weights.pth) | [ep71 (55.797)](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/vod_4d_mambanet_seed3_ep71_weights.pth) | **56.08 ± 0.19** |

## TJ4D — best by mAP_3D

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean mAP_3D / mAP_BEV |
|---|---|---|---|---|---|
| **4D-MambaNet** | [ep71](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/tj4d_4d_mambanet_seed0_ep71_weights.pth) | [ep74](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/tj4d_4d_mambanet_seed1_ep74_weights.pth) | [ep76](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/tj4d_4d_mambanet_seed2_ep76_weights.pth) | [ep58](https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0/tj4d_4d_mambanet_seed3_ep58_weights.pth) | **34.60 ± 0.57** / **39.63 ± 0.30** |

## SHA256 (weights-only)

```
8a3ad1cbfda38f00060a6179af2de055446faf69e9224fcf1590c00c237aaae9  vod_4d_mambanet_seed0_ep78_weights.pth
0af342d669569f33c12f00b57c63c44cd5a03fa652a4ec30d3820a824251ea42  vod_4d_mambanet_seed1_ep79_weights.pth
d75a1962ba9ef7bceaf2a0ba0e60c3fe9fce4a997f83f9b5c39255c1c0143e21  vod_4d_mambanet_seed2_ep80_weights.pth
d276fa41ee7a5537f6a253669615db4871a9403d363bafc50d07a70d3e084518  vod_4d_mambanet_seed3_ep71_weights.pth
0bd500cd4cc8ba79c06e7820da658b8c5e26f515ca6f1bb414604e7a7c59e13b  vod_m3net_seed0_ep78_weights.pth
476495b9f5968c2bbb392349a2450c9e5b7fd2ec0f255ec994049dac9691f1c2  vod_m3net_seed1_ep61_weights.pth
9a2f7304b6f1923d060fea2a95981afbdd01d275692b424dc6a988c3869efc8a  vod_m3net_seed2_ep73_weights.pth
823663b9470def078d4681f95bc7457cee4a50d32699cc11cc0d631ed3aee81a  vod_m3net_seed3_ep73_weights.pth
4e5d3ac995f0325da7b05791ab43b5243b84e2e18227520240238493385c58c3  tj4d_4d_mambanet_seed0_ep71_weights.pth
18f6536deb4109c367a5407a726a83221940574c1fa53e95371accb5448cdd86  tj4d_4d_mambanet_seed1_ep74_weights.pth
53a005711829fe87735fdc97748ba94475a26324f881c3bff71ffea97aa516ce  tj4d_4d_mambanet_seed2_ep76_weights.pth
5904868be6bdf5a47a7cb23602a5975a4caa53e66c0fdd9039cdd66926993ad5  tj4d_4d_mambanet_seed3_ep58_weights.pth
```
