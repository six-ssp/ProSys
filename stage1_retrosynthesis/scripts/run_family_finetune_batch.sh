#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
GPU_IDS="${GPU_IDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
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
SKIP_REMAP="${SKIP_REMAP:-1}"
MAX_EPOCH="${MAX_EPOCH:-200}"
MAX_UPDATE="${MAX_UPDATE:-200000}"
LR="${LR:-0.0003}"
WARMUP="${WARMUP:-10000}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
ALLOW_CPU_STAGE1="${ALLOW_CPU_STAGE1:-0}"

if [[ -n "${FAMILIES:-}" ]]; then
  IFS=', ' read -r -a FAMILIES <<< "$FAMILIES"
else
  mapfile -t FAMILIES < <(
    find "$REPO_ROOT/data/editretro/datasets" -maxdepth 1 -mindepth 1 -type d -name 'REAXYS_*_SINGLE_CATMERGE' \
    -printf '%f\n' | sort
  )
fi

mkdir -p "$RESULTS_ROOT"

cd "$REPO_ROOT"

if [[ "${OMP_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ "${MKL_NUM_THREADS:-0}" =~ ^[0-9]+$ ]] && [[ "${MKL_NUM_THREADS:-0}" -le 0 ]]; then
  export MKL_NUM_THREADS=8
fi

if [[ "$SKIP_REMAP" != "1" ]]; then
  "$PYTHON_BIN" stage1_retrosynthesis/scripts/remap_reaxys_routes.py --datasets_root data/editretro/datasets
fi

"$REPO_ROOT/stage1_retrosynthesis/scripts/ensure_fairseq_extensions.sh" "$REPO_ROOT"

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
declare -A JOB_DATASET=()
failed_families=()
completed_families=()

# One family's failure (e.g. a bad SMILES in preprocessing) must not abort the
# whole batch or kill its concurrent sibling. Track PID->dataset, disable -e
# around the waits, and record per-family exit status for a final summary.
# (bash 5.0 has no `wait -n -p`, so block on one pid at a time.)
reap_one() {
  local pid rc
  for pid in "${!JOB_DATASET[@]}"; do
    wait "$pid"
    rc=$?
    local ds="${JOB_DATASET[$pid]}"
    if [[ "$rc" -eq 0 ]]; then
      completed_families+=("$ds")
      echo "[stage1] finished $ds"
    else
      failed_families+=("$ds")
      echo "[stage1] FAILED $ds (exit $rc) — continuing with remaining families"
    fi
    unset "JOB_DATASET[$pid]"
    return 0
  done
}

set +e
for dataset in "${FAMILIES[@]}"; do
  gpu_index=$(( running_jobs % gpu_slot_count ))
  echo "[stage1] launch $dataset on GPU ${GPU_ARRAY[$gpu_index]}"
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
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  PATIENCE="${PATIENCE:-15}" \
  KEEP_LAST_EPOCHS="${KEEP_LAST_EPOCHS:-1}" \
  SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-0}" \
  NO_EPOCH_CHECKPOINTS="${NO_EPOCH_CHECKPOINTS:-1}" \
  "$REPO_ROOT/stage1_retrosynthesis/scripts/run_family_finetune_one.sh" "$REPO_ROOT" "$dataset" &
  JOB_DATASET[$!]="$dataset"

  running_jobs=$(( running_jobs + 1 ))
  if [[ "$running_jobs" -ge "$gpu_slot_count" ]]; then
    reap_one
    running_jobs=$(( running_jobs - 1 ))
  fi
done

# Drain remaining jobs.
while [[ "${#JOB_DATASET[@]}" -gt 0 ]]; do
  reap_one
done

echo "[stage1] batch done: ${#completed_families[@]} ok, ${#failed_families[@]} failed"
if [[ "${#failed_families[@]}" -gt 0 ]]; then
  echo "[stage1] failed families: ${failed_families[*]}"
  exit 1
fi
