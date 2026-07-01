#!/bin/bash
#
# ProSys end-to-end pipeline: Stage 1 family finetune -> Stage 2 V2 (train + Oracle eval) -> hit-rate summary.
#
# Designed to saturate a single-GPU host: Stage 1 runs multiple families concurrently on one GPU,
# Stage 2 uses the abundant CPU cores for candidate-pool building. All paths are derived from the
# repo root argument; nothing is hardcoded.
#
# Usage:
#   conda activate ProSys
#   bash scripts/run_full_pipeline.sh [repo_root]
#
# Tunables (env, with defaults):
#   GPU_IDS              slots on the GPU for Stage 1 (repeat a device id to co-locate families)
#   MAX_TOKENS           per-family max tokens/batch for Stage 1
#   NUM_WORKERS          Stage 1 dataloader workers per family
#   PATIENCE             Stage 1 early-stop patience (validations w/o improvement)
#   KEEP_LAST_EPOCHS     Stage 1 per-epoch checkpoints to keep (disk guard)
#   STAGE2_EPOCHS        Stage 2 V2 training epochs
#   STAGE2_PREPROCESS    Stage 2A candidate-pool build parallelism (CPU)
#   TRAIN_DEVICE         device for Stage 2 training + eval
#   SKIP_STAGE1          set to 1 to skip Stage 1 (e.g. Stage 2 only)

set -uo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
GPU_IDS="${GPU_IDS:-0,0}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-15}"
KEEP_LAST_EPOCHS="${KEEP_LAST_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE2_PREPROCESS="${STAGE2_PREPROCESS:-16}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

RESULTS_ROOT="$REPO_ROOT/stage1/results/family_finetune"
PIPE_DIR="$REPO_ROOT/stage1/results/full_pipeline"
mkdir -p "$PIPE_DIR"

echo "[pipeline] repo_root=$REPO_ROOT python=$PYTHON_BIN"
echo "[pipeline] start $(date)"

if [[ "$SKIP_STAGE1" != "1" ]]; then
  echo "[pipeline] === Stage 1 family finetune (GPU_IDS=$GPU_IDS, patience=$PATIENCE) ==="
  GPU_IDS="$GPU_IDS" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  MAX_TOKENS="$MAX_TOKENS" \
  NUM_WORKERS="$NUM_WORKERS" \
  PATIENCE="$PATIENCE" \
  KEEP_LAST_EPOCHS="$KEEP_LAST_EPOCHS" \
    bash "$REPO_ROOT/stage1/scripts/run_family_finetune_batch.sh" "$REPO_ROOT"
  echo "[pipeline] Stage 1 finished exit=$? $(date)"
else
  echo "[pipeline] SKIP_STAGE1=1, skipping Stage 1"
fi

echo "[pipeline] === Stage 2 V2 full-family batch (train + Oracle eval) ==="
"$PYTHON_BIN" scripts/run_stage2_v2_family_batch.py \
  --repo_root "$REPO_ROOT" \
  --families all \
  --output_root outputs/stage2_v2 \
  --candidate_device cpu \
  --parallel_preprocess "$STAGE2_PREPROCESS" \
  --train_devices "$TRAIN_DEVICE" \
  --parallel_train 1 \
  --epochs "$STAGE2_EPOCHS" \
  --eval_device "$TRAIN_DEVICE"
echo "[pipeline] Stage 2 finished exit=$? $(date)"

echo "[pipeline] === hit-rate summary ==="
"$PYTHON_BIN" scripts/summarize_hitrates.py \
  --repo_root "$REPO_ROOT" \
  --json_out "$PIPE_DIR/hitrate_summary.json" | tee "$PIPE_DIR/hitrate_summary.txt"

echo "[pipeline] done $(date)"
