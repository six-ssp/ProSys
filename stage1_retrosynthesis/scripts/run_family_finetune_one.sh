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
DEFAULT_ALIAS_CKPT="$REPO_ROOT/stage1_retrosynthesis/checkpoints/checkpoint_USPTO_STAGE2_FILTERED_best.pt"
if [[ -z "${BASE_CKPT:-}" ]]; then
  if [[ -f "$DEFAULT_ALIAS_CKPT" ]]; then
    BASE_CKPT="$DEFAULT_ALIAS_CKPT"
  else
    BASE_CKPT="$(find "$REPO_ROOT/stage1_retrosynthesis/results/base_train/USPTO_STAGE2_FILTERED" \
      -mindepth 3 -maxdepth 3 -path '*/checkpoints/checkpoint_best.pt' -print | sort | tail -n 1)"
  fi
fi
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
KEEP_LAST_EPOCHS="${KEEP_LAST_EPOCHS:-1}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-0}"
NO_EPOCH_CHECKPOINTS="${NO_EPOCH_CHECKPOINTS:-1}"

run_name="$(date "+%Y%m%d_%H%M%S")"
family_root="$RESULTS_ROOT/$DATASET/$run_name"
model_dir="$family_root/checkpoints"
prepare_log="$family_root/prepare.log"
train_log="$family_root/train.log"
mkdir -p "$model_dir"

if [[ -z "$BASE_CKPT" || ! -f "$BASE_CKPT" ]]; then
  echo "[stage1] base checkpoint not found: ${BASE_CKPT:-<empty>}" >&2
  exit 1
fi

if [[ "${OMP_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ "${MKL_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${MKL_NUM_THREADS:-0}" -le 0 ]]; then
  export MKL_NUM_THREADS=8
fi

cd "$REPO_ROOT"

echo "[stage1] preparing $DATASET -> $prepare_log"
"$PYTHON_BIN" stage1_retrosynthesis/scripts/prepare_family_binarized.py \
  --dataset "$DATASET" \
  --augmentation "$AUGMENTATION" \
  --processes "$PROCESSES" \
  --repo_root "$REPO_ROOT" \
  > "$prepare_log" 2>&1

databin="$REPO_ROOT/data/editretro/datasets/$DATASET/aug$AUGMENTATION/data-bin"

train_cmd=(
  "$PYTHON_BIN"
  "$REPO_ROOT/stage1_retrosynthesis/fairseq/fairseq_cli/train.py"
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
  --save-interval-updates "$SAVE_INTERVAL_UPDATES"
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

if [[ "$NO_EPOCH_CHECKPOINTS" == "1" ]]; then
  train_cmd+=(--no-epoch-checkpoints)
fi

export PYTHONPATH="$REPO_ROOT/stage1_retrosynthesis/fairseq${PYTHONPATH:+:$PYTHONPATH}"
echo "[stage1] training $DATASET -> $train_log"
CUDA_VISIBLE_DEVICES="$GPU_ID" "${train_cmd[@]}" > "$train_log" 2>&1

find "$model_dir" -maxdepth 1 -type f \
  \( -name 'checkpoint[0-9]*.pt' -o -name 'checkpoint_*_*.pt' -o -name 'checkpoint.best_*.pt' \) \
  ! -name 'checkpoint_best.pt' ! -name 'checkpoint_last.pt' -delete

echo "[stage1] finished $DATASET"
