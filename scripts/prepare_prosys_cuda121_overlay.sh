#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
OVERLAY_ROOT="${OVERLAY_ROOT:-$REPO_ROOT/runtime/overlay_cuda121}"
CLEAN_OVERLAY="${CLEAN_OVERLAY:-0}"
CONDA_BASE_PYTHON="${CONDA_BASE_PYTHON:-$(dirname "$(dirname "$(command -v conda)")")/bin/python}"

if [[ ! -x "$CONDA_BASE_PYTHON" ]]; then
  echo "[overlay] cannot find conda base python: $CONDA_BASE_PYTHON"
  exit 1
fi

mkdir -p "$OVERLAY_ROOT"

if [[ "$CLEAN_OVERLAY" == "1" ]]; then
  "$CONDA_BASE_PYTHON" - <<'PY' "$OVERLAY_ROOT"
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True, exist_ok=True)
PY
fi

"$CONDA_BASE_PYTHON" - <<'PY' "$OVERLAY_ROOT"
import sys
import tarfile
from pathlib import Path

from conda_package_handling.api import extract

overlay = Path(sys.argv[1])
overlay.mkdir(parents=True, exist_ok=True)

packages = [
    '/home/six_ssp/miniconda3/pkgs/pytorch-2.5.1-py3.9_cuda12.1_cudnn9.1.0_0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/mkl-2025.0.0-hacee8c2_941.conda',
    '/home/six_ssp/miniconda3/pkgs/cuda-cudart-12.1.105-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/cuda-nvrtc-12.1.105-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libcublas-12.1.0.26-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libcufft-11.2.1.3-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libcusolver-11.6.1.9-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libcusparse-12.3.1.170-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libnvjitlink-12.4.127-0.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/libcurand-10.4.2.55-h91de7bb_0.conda',
    '/home/six_ssp/miniconda3/pkgs/intel-openmp-2025.0.0-h06a4308_1171.conda',
    '/home/six_ssp/miniconda3/pkgs/torchvision-0.20.1-py39_cu121.tar.bz2',
    '/home/six_ssp/miniconda3/pkgs/torchaudio-2.5.1-py39_cu121.tar.bz2',
]

missing = [path for path in packages if not Path(path).exists()]
if missing:
    raise SystemExit('missing cached packages:\n' + '\n'.join(missing))

for raw_path in packages:
    path = Path(raw_path)
    print(f'[overlay] extract {path.name}')
    if path.suffix == '.conda':
        extract(str(path), dest_dir=str(overlay))
        continue

    with tarfile.open(path, 'r:*') as handle:
        handle.extractall(path=overlay)
PY

gcc -shared -fPIC -O2 \
  -o "$OVERLAY_ROOT/lib/libittnotify.so" \
  "$REPO_ROOT/scripts/libittnotify_stub.c"

echo "[overlay] prepared at $OVERLAY_ROOT"
