#!/usr/bin/env bash
# Reproduce paper table summaries from checked-in per-epoch logs / released ckpts.
# Usage: bash tools/scripts/reproduce_table.sh table1
set -euo pipefail
TABLE=${1:?table name, e.g. table1}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python}"
case "${TABLE}" in
  table1|main|vod)
    ${PY} "${ROOT}/tools/analysis/summarize_seeds.py"
    ;;
  *)
    echo "Supported: table1|main|vod (extend as needed). Got: ${TABLE}"
    exit 1
    ;;
esac
