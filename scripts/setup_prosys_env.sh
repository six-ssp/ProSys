#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu124}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+cu124}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1+cu124}"

cd "$REPO_ROOT"

if [[ "$FORCE_TORCH_REINSTALL" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --force-reinstall \
    --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"
fi

"$PYTHON_BIN" scripts/check_runtime.py --repo_root "$REPO_ROOT"

cd "$REPO_ROOT/stage1_retrosynthesis/fairseq"

CUDA_HOME_VALUE="$("$PYTHON_BIN" - <<'PY'
import os
import shutil
from torch.utils.cpp_extension import CUDA_HOME

value = os.environ.get('CUDA_HOME') or CUDA_HOME or ''
if not value and shutil.which('nvcc'):
    value = os.path.dirname(os.path.dirname(shutil.which('nvcc')))
print(value)
PY
)"

if [[ -n "$CUDA_HOME_VALUE" ]]; then
  export CUDA_HOME="$CUDA_HOME_VALUE"
fi

"$REPO_ROOT/stage1_retrosynthesis/scripts/ensure_fairseq_extensions.sh" "$REPO_ROOT"
