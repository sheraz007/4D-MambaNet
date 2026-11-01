# Data layout

Default paths are **repository-relative**. Override with `MAMBANET4D_DATA` (parent of `vod/` and `tj4d/`) or `DATA_PATH` (exact dataset root), or edit the YAML.

## View-of-Delft (VoD), 5-scan radar

**Download:** [view-of-delft-dataset](https://github.com/tudelft-iv/view-of-delft-dataset) (official GitHub) and the [View-of-Delft Dataset](https://intelligent-vehicles.org/datasets/view-of-delft/) release page. Obtain the **radar_5frames** split from the VoD release; the temporal model also needs the **`clips/`** directory (multi-frame clip indices used by `vodDataset_Temporal` to load neighbouring radar scans).

Expected layout after linking/copying into this repo:

```text
data/vod/
├── lidar/                          # optional; used by temporal loader defaults
└── radar_5frames/
    ├── clips/
    ├── training/                   # VoD radar frames (dataset release layout)
    ├── ImageSets/
    │   ├── train.txt
    │   └── val.txt
    ├── vod_infos_train_temporal.pkl
    └── vod_infos_val_temporal.pkl
```

### Create VoD temporal info pickles

From the repository root (after install):

```bash
python -m pcdet.datasets.kitti.vod_dataset_temporal create_vod_infos \
  tools/cfgs/dataset_configs/radar_5frames_tem_dataset.yaml
```

Or, with an explicit data root (edit the script paths / set `MAMBANET4D_DATA` so `DATA_PATH` resolves to your `radar_5frames` folder):

```bash
export MAMBANET4D_DATA=/path/to/parent   # parent that contains vod/radar_5frames
python -m pcdet.datasets.kitti.vod_dataset_temporal create_vod_infos \
  tools/cfgs/dataset_configs/radar_5frames_tem_dataset.yaml
```

This writes `vod_infos_train_temporal.pkl` and `vod_infos_val_temporal.pkl` next to the radar frames (see `create_vod_infos` in `pcdet/datasets/kitti/vod_dataset_temporal.py`).

## TJ4DRadSet

**Download:** [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet) (official GitHub release / project page).

```text
data/tj4d/
├── training/                       # TJ4D release layout
├── ImageSets/
├── pcdtj_infos_train.pkl           # or kitti_infos_*.pkl from the creator below
└── pcdtj_infos_val.pkl
```

### Create TJ4D info pickles

```bash
python -m pcdet.datasets.kitti.tj4d \
  # uses tools/cfgs/dataset_configs/tj4d.yaml and data/tj4d by default
```

See `create_kitti_infos` in `pcdet/datasets/kitti/tj4d.py`. Rename or point YAML `INFO_PATH` entries to the generated pickles if your filenames differ.

## Environment overrides

```bash
# Parent directory that contains vod/ and tj4d/
export MAMBANET4D_DATA=/path/to/datasets

# Or set the exact DATA_PATH for the active config
export DATA_PATH=/path/to/data/vod/radar_5frames
```
