#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATASET="${2:-}"

if [[ -z "$DATASET" ]]; then
  echo "usage: $0 <repo_root> <dataset>"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/stage1_retrosynthesis/checkpoints/checkpoint_UPSTO_full_best.pt}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/stage1_retrosynthesis/results/family_finetune}"
AUGMENTATION="${AUGMENTATION:-10}"
PROCESSES="${PROCESSES:-8}"
MAX_EPOCH="${MAX_EPOCH:-200}"
MAX_UPDATE="${MAX_UPDATE:-200000}"
LR="${LR:-0.0003}"
WARMUP="${WARMUP:-10000}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
USE_FP16="${USE_FP16:-1}"
# Throughput / disk knobs.
NUM_WORKERS="${NUM_WORKERS:-8}"          # dataloader workers (host has many cores)
PATIENCE="${PATIENCE:-15}"               # early-stop after N validations w/o val-loss improvement (-1 disables)
KEEP_LAST_EPOCHS="${KEEP_LAST_EPOCHS:-2}"  # cap per-epoch checkpoints (each ~470MB) to protect disk

run_name="$(date "+%Y%m%d_%H%M%S")"
family_root="$RESULTS_ROOT/$DATASET/$run_name"
model_dir="$family_root/checkpoints"
mkdir -p "$model_dir"

cd "$REPO_ROOT"

"$PYTHON_BIN" stage1_retrosynthesis/scripts/prepare_family_binarized.py \
  --dataset "$DATASET" \
  --augmentation "$AUGMENTATION" \
  --processes "$PROCESSES" \
  --repo_root "$REPO_ROOT"

databin="$REPO_ROOT/data/editretro/datasets/$DATASET/aug$AUGMENTATION/data-bin"

train_cmd=(
  fairseq-train
  "$databin"
  --user-dir "$REPO_ROOT/stage1_retrosynthesis/editretro"
  -s src
  -t tgt
  --save-dir "$model_dir"
  --ddp-backend=no_c10d
  --task translation_retro
  --criterion nat_loss
  --arch editretro_nat
  --noise random_delete_shuffle
  --optimizer adam
  --adam-betas "(0.9,0.98)"
  --lr "$LR"
  --lr-scheduler inverse_sqrt
  --min-lr 1e-09
  --warmup-updates "$WARMUP"
  --warmup-init-lr 1e-07
  --label-smoothing 0.1
  --dropout 0.2
  --attention-dropout 0.2
  --weight-decay 0.01
  --share-all-embeddings
  --decoder-learned-pos
  --encoder-learned-pos
  --max-tokens-valid 4000
  --log-format simple
  --log-interval 200
  --fixed-validation-seed 7
  --max-tokens "$MAX_TOKENS"
  --num-workers "$NUM_WORKERS"
  --patience "$PATIENCE"
  --save-interval-updates 5000
  --keep-last-epochs "$KEEP_LAST_EPOCHS"
  --max-epoch "$MAX_EPOCH"
  --max-update "$MAX_UPDATE"
  --alpha-ratio 0.5
  --dae-ratio 0.5
  --update-freq "$UPDATE_FREQ"
  --restore-file "$BASE_CKPT"
  --reset-optimizer
  --reset-lr-scheduler
  --reset-meters
  --reset-dataloader
)

if [[ "$USE_FP16" == "1" ]]; then
  train_cmd+=(--fp16)
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "${train_cmd[@]}" > "$family_root/train.log" 2>&1
