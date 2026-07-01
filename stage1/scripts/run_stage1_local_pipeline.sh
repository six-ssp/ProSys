#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/stage1/results}"
AUDIT_LOG="${AUDIT_LOG:-$RESULTS_ROOT/stage1_split_audit.log}"
PIPELINE_LOG="${PIPELINE_LOG:-$RESULTS_ROOT/stage1_pipeline.log}"

mkdir -p "$RESULTS_ROOT"

cd "$REPO_ROOT"

{
  echo "[stage1] rebuild route datasets"
  "$PYTHON_BIN" data_preprocess/rebuild_stage1_routes.py \
    --input_dir data/reaxys_input \
    --output_dir data

  echo "[stage1] audit data splits"
  "$PYTHON_BIN" scripts/audit_data_splits.py --strict | tee "$AUDIT_LOG"

  echo "[stage1] start family finetune batch"
  "$REPO_ROOT/stage1/scripts/run_family_finetune_batch.sh" "$REPO_ROOT"
} 2>&1 | tee "$PIPELINE_LOG"
