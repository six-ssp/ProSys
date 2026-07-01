#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FAIRSEQ_ROOT="$REPO_ROOT/stage1/fairseq"

check_libnat() {
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

try:
    import fairseq.libnat  # noqa: F401
except Exception as exc:
    print(f'fairseq.libnat unavailable: {exc}')
    sys.exit(1)

print('fairseq.libnat ready')
PY
}

cd "$FAIRSEQ_ROOT"

if check_libnat; then
  exit 0
fi

echo "[stage1] build fairseq extensions"
"$PYTHON_BIN" setup.py build_ext --inplace

check_libnat
