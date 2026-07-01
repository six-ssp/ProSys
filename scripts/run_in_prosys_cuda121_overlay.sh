#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
shift || true

OVERLAY_ROOT="${OVERLAY_ROOT:-$REPO_ROOT/runtime/overlay_cuda121}"
BASE_ENV_ROOT="${BASE_ENV_ROOT:-/home/six_ssp/miniconda3/envs/ProSys}"
SITE_PACKAGES="$OVERLAY_ROOT/lib/python3.9/site-packages"
TORCH_LIB="$SITE_PACKAGES/torch/lib"
ITT_STUB="$OVERLAY_ROOT/lib/libittnotify.so"

if [[ ! -d "$SITE_PACKAGES/torch" ]]; then
  echo "[overlay] missing torch overlay at $SITE_PACKAGES/torch"
  echo "[overlay] run scripts/prepare_prosys_cuda121_overlay.sh first"
  exit 1
fi

if [[ ! -f "$ITT_STUB" ]]; then
  echo "[overlay] missing ITT stub at $ITT_STUB"
  echo "[overlay] run scripts/prepare_prosys_cuda121_overlay.sh first"
  exit 1
fi

export PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$OVERLAY_ROOT/lib:$TORCH_LIB:$BASE_ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ITT_STUB${LD_PRELOAD:+:$LD_PRELOAD}"

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"
