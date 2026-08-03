#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATASET="${2:-USPTO_STAGE2_FILTERED}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/stage1_retrosynthesis/results/base_train}"
AUGMENTATION="${AUGMENTATION:-10}"
PROCESSES="${PROCESSES:-8}"
MAX_EPOCH="${MAX_EPOCH:-50}"
MAX_UPDATE="${MAX_UPDATE:-200000}"
LR="${LR:-0.0003}"
WARMUP="${WARMUP:-10000}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
USE_FP16="${USE_FP16:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-10}"
KEEP_LAST_EPOCHS="${KEEP_LAST_EPOCHS:-1}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-0}"
NO_EPOCH_CHECKPOINTS="${NO_EPOCH_CHECKPOINTS:-1}"
RESTORE_CKPT="${RESTORE_CKPT:-}"
ALIAS_NAME="${ALIAS_NAME:-checkpoint_${DATASET}_best.pt}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

run_name="$(date "+%Y%m%d_%H%M%S")"
run_root="$RESULTS_ROOT/$DATASET/$run_name"
model_dir="$run_root/checkpoints"
mkdir -p "$model_dir"

# Some runtime environments export OMP/MKL thread counts as 0, which causes
# libgomp / OpenMP initialization to fail immediately when Python imports
# native extensions. Clamp them before any Python process starts.
if [[ "${OMP_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ "${MKL_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${MKL_NUM_THREADS:-0}" -le 0 ]]; then
  export MKL_NUM_THREADS=8
fi

cd "$REPO_ROOT"

"$REPO_ROOT/stage1_retrosynthesis/scripts/ensure_fairseq_extensions.sh" "$REPO_ROOT"

databin="$REPO_ROOT/data/editretro/datasets/$DATASET/aug$AUGMENTATION/data-bin"
if [[ "$SKIP_PREPARE" != "1" ]]; then
  "$PYTHON_BIN" stage1_retrosynthesis/scripts/prepare_family_binarized.py \
    --dataset "$DATASET" \
    --augmentation "$AUGMENTATION" \
    --processes "$PROCESSES" \
    --repo_root "$REPO_ROOT"
elif [[ ! -f "$databin/dict.src.txt" ]]; then
  echo "SKIP_PREPARE=1 but missing data-bin: $databin" >&2
  exit 1
fi

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
)

if [[ -n "$RESTORE_CKPT" ]]; then
  train_cmd+=(
    --restore-file "$RESTORE_CKPT"
    --reset-optimizer
    --reset-lr-scheduler
    --reset-meters
    --reset-dataloader
  )
fi

if [[ "$USE_FP16" == "1" ]]; then
  train_cmd+=(--fp16)
fi

if [[ "$NO_EPOCH_CHECKPOINTS" == "1" ]]; then
  train_cmd+=(--no-epoch-checkpoints)
fi

export PYTHONPATH="$REPO_ROOT/stage1_retrosynthesis/fairseq${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "${train_cmd[@]}" > "$run_root/train.log" 2>&1

best_ckpt="$model_dir/checkpoint_best.pt"
if [[ -f "$best_ckpt" ]]; then
  mkdir -p "$REPO_ROOT/stage1_retrosynthesis/checkpoints"
  alias_path="$REPO_ROOT/stage1_retrosynthesis/checkpoints/$ALIAS_NAME"
  rm -f "$alias_path"
  ln -s "$best_ckpt" "$alias_path"
  echo "best checkpoint: $best_ckpt"
  echo "alias updated: $alias_path"
else
  echo "warning: checkpoint_best.pt not found under $model_dir" >&2
fi

find "$model_dir" -maxdepth 1 -type f \
  \( -name 'checkpoint[0-9]*.pt' -o -name 'checkpoint_*_*.pt' -o -name 'checkpoint.best_*.pt' \) \
  ! -name 'checkpoint_best.pt' ! -name 'checkpoint_last.pt' -delete
