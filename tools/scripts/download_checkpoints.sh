#!/usr/bin/env bash
# Download released checkpoints (GitHub Release v1.0) and verify SHA256.
# Fill RELEASE_URL after Phase 4 publish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RELEASE_URL="${RELEASE_URL:-}"
OUT="${ROOT}/checkpoints"
mkdir -p "${OUT}"
if [[ -z "${RELEASE_URL}" ]]; then
  echo "Set RELEASE_URL to the GitHub Release asset base URL, e.g.:"
  echo "  export RELEASE_URL=https://github.com/sheraz007/4D-MambaNet/releases/download/v1.0"
  echo "Then re-run this script."
  exit 1
fi
# Asset names will match checkpoints.sha256
while read -r hash name; do
  [[ -z "${name:-}" ]] && continue
  [[ "${hash}" =~ ^# ]] && continue
  echo "Fetching ${name} ..."
  curl -L --fail -o "${OUT}/${name}" "${RELEASE_URL}/${name}"
done < "${ROOT}/checkpoints.sha256"
echo "Verifying checksums..."
(cd "${OUT}" && sha256sum -c "${ROOT}/checkpoints.sha256")
echo "OK → ${OUT}"
