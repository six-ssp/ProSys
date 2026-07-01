#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
GPU_IDS="${GPU_IDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/stage1/checkpoints/checkpoint_UPSTO_full_best.pt}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/stage1/results/family_finetune}"
AUGMENTATION="${AUGMENTATION:-10}"
PROCESSES="${PROCESSES:-8}"
SKIP_REMAP="${SKIP_REMAP:-1}"
MAX_EPOCH="${MAX_EPOCH:-200}"
MAX_UPDATE="${MAX_UPDATE:-200000}"
LR="${LR:-0.0003}"
WARMUP="${WARMUP:-10000}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
ALLOW_CPU_STAGE1="${ALLOW_CPU_STAGE1:-0}"

FAMILIES=(
  "REAXYS_Beckmann_SINGLE_CATMERGE"
  "REAXYS_Buchwald-HartwigCross-Coupling_SINGLE_CATMERGE"
  "REAXYS_Chan_LamCoupling_SINGLE_CATMERGE"
  "REAXYS_DielsAlder_SINGLE_CATMERGE"
  "REAXYS_FischerIndoleSynthesis_SINGLE_CATMERGE"
  "REAXYS_Friedel-CraftsAcylation_SINGLE_CATMERGE"
  "REAXYS_Friedel-CraftsAlkylation_SINGLE_CATMERGE"
  "REAXYS_GrignardReaction_SINGLE_CATMERGE"
  "REAXYS_KumadaCoupling_SINGLE_CATMERGE"
  "REAXYS_NegishiCoupling_SINGLE_CATMERGE"
)

mkdir -p "$RESULTS_ROOT"

cd "$REPO_ROOT"

if [[ "$SKIP_REMAP" != "1" ]]; then
  "$PYTHON_BIN" stage1/scripts/remap_reaxys_routes.py --datasets_root data/editretro/datasets
fi

"$REPO_ROOT/stage1/scripts/ensure_fairseq_extensions.sh" "$REPO_ROOT"

cuda_status="$("$PYTHON_BIN" - <<'PY'
import torch

cuda_available = bool(torch.cuda.is_available())
cuda_version = torch.version.cuda or ''
print(f"{int(cuda_available)}|{cuda_version}")
PY
)"
cuda_available="${cuda_status%%|*}"
cuda_version="${cuda_status#*|}"

if [[ "$cuda_available" != "1" && "$ALLOW_CPU_STAGE1" != "1" ]]; then
  echo "[stage1] CUDA-capable PyTorch is not available in the current environment."
  echo "[stage1] torch.version.cuda=${cuda_version:-none}"
  echo "[stage1] Refusing to start full family finetune on CPU."
  echo "[stage1] If you intentionally want CPU smoke runs, set ALLOW_CPU_STAGE1=1."
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
gpu_slot_count="${#GPU_ARRAY[@]}"
if [[ "$gpu_slot_count" -lt 1 ]]; then
  echo "[stage1] no GPU slots configured"
  exit 1
fi

running_jobs=0
for dataset in "${FAMILIES[@]}"; do
  gpu_index=$(( running_jobs % gpu_slot_count ))
  GPU_ID="${GPU_ARRAY[$gpu_index]}" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_CKPT="$BASE_CKPT" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  AUGMENTATION="$AUGMENTATION" \
  PROCESSES="$PROCESSES" \
  MAX_EPOCH="$MAX_EPOCH" \
  MAX_UPDATE="$MAX_UPDATE" \
  LR="$LR" \
  WARMUP="$WARMUP" \
  MAX_TOKENS="$MAX_TOKENS" \
  UPDATE_FREQ="$UPDATE_FREQ" \
  "$REPO_ROOT/stage1/scripts/run_family_finetune_one.sh" "$REPO_ROOT" "$dataset" &

  running_jobs=$(( running_jobs + 1 ))
  if [[ "$running_jobs" -ge "$gpu_slot_count" ]]; then
    wait -n
    running_jobs=$(( running_jobs - 1 ))
  fi
done

wait
