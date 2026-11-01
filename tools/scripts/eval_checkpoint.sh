#!/usr/bin/env bash
# Evaluate one checkpoint → R11 (official VoD) via tools/test.py.
# Usage: bash tools/scripts/eval_checkpoint.sh <cfg> <ckpt> [gpu] [batch_size]
# Paper tables use batch_size=8; default here is 8.
set -euo pipefail
CFG=${1:?cfg}
CKPT=${2:?checkpoint}
GPU=${3:-0}
BS=${4:-8}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/tools"
PY="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${GPU}" ${PY} test.py \
  --cfg_file "${CFG}" \
  --batch_size "${BS}" \
  --workers 4 \
  --ckpt "${CKPT}" \
  --eval_tag "public_eval"
echo "For R40 from result.pkl, run: python tools/analysis/compute_r40.py (see docs/REPRODUCE.md)"
