#!/usr/bin/env bash
# Train one config for seeds 0-3 (80 epochs, batch 8, eval epochs 51-80).
# Usage: bash tools/scripts/train_4seeds.sh <cfg_relative_to_tools> <gpu_id>
# Example: bash tools/scripts/train_4seeds.sh cfgs/vod/4d_mambanet.yaml 0
set -euo pipefail
CFG=${1:?config path relative to tools/, e.g. cfgs/vod/4d_mambanet.yaml}
GPU=${2:?gpu id}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/tools"
PY="${PYTHON:-python}"
TAG=$(basename "$CFG" .yaml)
for SEED in 0 1 2 3; do
  echo "=== seed ${SEED} gpu ${GPU} cfg ${CFG} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" ${PY} train.py \
    --cfg_file "${CFG}" \
    --batch_size 8 \
    --epochs 80 \
    --workers 0 \
    --extra_tag "seed${SEED}" \
    --fix_random_seed \
    --seed "${SEED}" \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 30 \
    --num_epochs_to_eval 30 \
    --eval_interval -1 \
    --work_dir "${ROOT}/output/${TAG}_s${SEED}"
done
