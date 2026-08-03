#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ProSys/bin/python}"
FAMILIES="${FAMILIES:-all}"
GPU_IDS="${GPU_IDS:-0}"
GEN_DEVICE="${GEN_DEVICE:-0}"
AUG="${AUG:-10}"
TOPK="${TOPK:-10}"
REPOS_BEAM="${REPOS_BEAM:-5}"
TOKEN_BEAM="${TOKEN_BEAM:-2}"
MASK_BEAM="${MASK_BEAM:-1}"
N_BEST="${N_BEST:-10}"
CANDIDATE_PROCESSES="${CANDIDATE_PROCESSES:-8}"
CLEAN_LEGACY="${CLEAN_LEGACY:-1}"
RESET_PROCESSED="${RESET_PROCESSED:-1}"
RESET_STAGE1_RESULTS="${RESET_STAGE1_RESULTS:-0}"
SKIP_STAGE1_FINETUNE="${SKIP_STAGE1_FINETUNE:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
FORCE_DATABIN_REBUILD="${FORCE_DATABIN_REBUILD:-0}"
FORCE_ROUTE_REBUILD="${FORCE_ROUTE_REBUILD:-0}"
RUN_BASE_ROUTE_EVAL="${RUN_BASE_ROUTE_EVAL:-1}"
FORCE_STAGE23_REBUILD="${FORCE_STAGE23_REBUILD:-1}"
MIN_LABEL_FREQ="${MIN_LABEL_FREQ:-6}"
LABEL_FREQ_SCOPE="${LABEL_FREQ_SCOPE:-family}"
MIN_YIELD="${MIN_YIELD:-25}"
RUN_BASE_TRAIN="${RUN_BASE_TRAIN:-0}"
RESET_BASE_RESULTS="${RESET_BASE_RESULTS:-0}"
BASE_DATASET="${BASE_DATASET:-USPTO_STAGE2_FILTERED}"
BASE_ALIAS_NAME="${BASE_ALIAS_NAME:-checkpoint_UPSTO_full_best.pt}"
BASE_RESTORE_CKPT="${BASE_RESTORE_CKPT:-}"

echo "[reproduce] repo_root=$REPO_ROOT"
echo "[reproduce] python=$PYTHON_BIN"
echo "[reproduce] families=$FAMILIES"
echo "[reproduce] min_yield=$MIN_YIELD"
echo "[reproduce] lowfreq scope=$LABEL_FREQ_SCOPE min_freq=$MIN_LABEL_FREQ"

if [[ "$CLEAN_LEGACY" == "1" ]]; then
  echo "[reproduce] remove legacy stage2_v2 / historical stage23 outputs"
  rm -rf "$REPO_ROOT/outputs/stage2_v2"
  rm -rf "$REPO_ROOT/outputs/stage23_non_oracle_all10"
fi

if [[ "$RESET_PROCESSED" == "1" ]]; then
  echo "[reproduce] reset processed datasets and current mainline outputs"
  rm -rf "$REPO_ROOT"/data/reaction_processed_*_catmerge
  rm -rf "$REPO_ROOT"/data/editretro/datasets/REAXYS_*_SINGLE_CATMERGE
  rm -rf "$REPO_ROOT/data/editretro/datasets/USPTO_STAGE2_FILTERED"
  rm -rf "$REPO_ROOT/outputs/stage1_routes"
  rm -rf "$REPO_ROOT/outputs/stage1_routes_base"
  rm -rf "$REPO_ROOT/outputs/stage23_mainline"
  rm -rf "$REPO_ROOT/outputs/checklist_stats"
fi

if [[ "$RESET_STAGE1_RESULTS" == "1" ]]; then
  echo "[reproduce] reset family finetune results"
  rm -rf "$REPO_ROOT/stage1_retrosynthesis/results/family_finetune"
fi

if [[ "$RESET_BASE_RESULTS" == "1" ]]; then
  echo "[reproduce] reset base-train results"
  rm -rf "$REPO_ROOT/stage1_retrosynthesis/results/base_train"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  echo "[reproduce] preprocess raw Reaxys + Stage1 route raw csv + USPTO filtered split"
  "$PYTHON_BIN" data_preprocess/preprocess.py \
    --input_dir "$REPO_ROOT/data/reaxys_input" \
    --output_dir "$REPO_ROOT/data" \
    --min_yield "$MIN_YIELD" \
    --min_label_freq "$MIN_LABEL_FREQ" \
    --label_freq_scope "$LABEL_FREQ_SCOPE" \
    --do_stage1 \
    --process_uspto
else
  echo "[reproduce] SKIP_PREPROCESS=1, reuse existing processed datasets"
fi

if [[ "$SKIP_AUDIT" != "1" ]]; then
  echo "[reproduce] audit Stage1/Stage2 splits"
  "$PYTHON_BIN" data_preprocess/audit_data_splits.py --strict
else
  echo "[reproduce] SKIP_AUDIT=1, skip split audit"
fi

family_base_ckpt="${BASE_CKPT:-$REPO_ROOT/stage1_retrosynthesis/checkpoints/$BASE_ALIAS_NAME}"

if [[ "$RUN_BASE_TRAIN" == "1" ]]; then
  echo "[reproduce] Stage1 base training on $BASE_DATASET"
  first_gpu="${GPU_IDS%%,*}"
  GPU_ID="${BASE_GPU_ID:-$first_gpu}" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESULTS_ROOT="${BASE_RESULTS_ROOT:-$REPO_ROOT/stage1_retrosynthesis/results/base_train}" \
  AUGMENTATION="$AUG" \
  PROCESSES="$CANDIDATE_PROCESSES" \
  MAX_EPOCH="${BASE_MAX_EPOCH:-50}" \
  MAX_UPDATE="${BASE_MAX_UPDATE:-200000}" \
  LR="${BASE_LR:-0.0003}" \
  WARMUP="${BASE_WARMUP:-10000}" \
  MAX_TOKENS="${BASE_MAX_TOKENS:-16384}" \
  UPDATE_FREQ="${BASE_UPDATE_FREQ:-1}" \
  NUM_WORKERS="${BASE_NUM_WORKERS:-8}" \
  PATIENCE="${BASE_PATIENCE:-10}" \
  KEEP_LAST_EPOCHS="${BASE_KEEP_LAST_EPOCHS:-2}" \
  VALIDATE_INTERVAL_UPDATES="${BASE_VALIDATE_INTERVAL_UPDATES:-5000}" \
  SAVE_INTERVAL_UPDATES="${BASE_SAVE_INTERVAL_UPDATES:-5000}" \
  RESTORE_CKPT="$BASE_RESTORE_CKPT" \
  ALIAS_NAME="$BASE_ALIAS_NAME" \
  bash "$REPO_ROOT/stage1_retrosynthesis/scripts/run_base_train.sh" "$REPO_ROOT" "$BASE_DATASET"
  family_base_ckpt="$REPO_ROOT/stage1_retrosynthesis/checkpoints/$BASE_ALIAS_NAME"
fi

IFS=', ' read -r -a FAMILY_ARRAY <<< "$FAMILIES"
if [[ "$FAMILIES" == "all" ]]; then
  FAMILY_ARRAY=(
    "Beckmann"
    "Buchwald-HartwigCross-Coupling"
    "Chan_LamCoupling"
    "DielsAlder"
    "Friedel-CraftsAcylation"
    "Friedel-CraftsAlkylation"
  )
fi

if [[ "$SKIP_STAGE1_FINETUNE" != "1" ]]; then
  echo "[reproduce] setup runtime extensions"
  bash "$REPO_ROOT/scripts/setup_prosys_env.sh" "$REPO_ROOT"

  echo "[reproduce] Stage1 family finetune"
  GPU_IDS="$GPU_IDS" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_CKPT="$family_base_ckpt" \
  bash "$REPO_ROOT/stage1_retrosynthesis/scripts/run_family_finetune_batch.sh" "$REPO_ROOT"
else
  echo "[reproduce] SKIP_STAGE1_FINETUNE=1, reuse existing family checkpoints"
  echo "[reproduce] rebuild family EditRetro data-bin for inference"
  for fam in "${FAMILY_ARRAY[@]}"; do
    dataset="REAXYS_${fam}_SINGLE_CATMERGE"
    databin="$REPO_ROOT/data/editretro/datasets/$dataset/aug$AUG/data-bin"
    if [[ "$FORCE_DATABIN_REBUILD" != "1" && -f "$databin/dict.src.txt" ]]; then
      echo "[reproduce] skip existing data-bin: $dataset"
      continue
    fi
    "$PYTHON_BIN" "$REPO_ROOT/stage1_retrosynthesis/scripts/prepare_family_binarized.py" \
      --dataset "$dataset" \
      --augmentation "$AUG" \
      --processes "$CANDIDATE_PROCESSES" \
      --repo_root "$REPO_ROOT"
  done
fi

echo "[reproduce] Stage1 tuned route caches"
for fam in "${FAMILY_ARRAY[@]}"; do
  tuned_cache="$REPO_ROOT/outputs/stage1_routes/$fam/route_cache.json"
  if [[ "$FORCE_ROUTE_REBUILD" != "1" && -f "$tuned_cache" ]]; then
    echo "[reproduce] skip existing tuned route cache: $fam"
    continue
  fi
  "$PYTHON_BIN" stage1_retrosynthesis/build_route_cache.py \
    --repo_root "$REPO_ROOT" \
    --family "$fam" \
    --aug "$AUG" \
    --topk "$TOPK" \
    --repos_beam "$REPOS_BEAM" \
    --token_beam "$TOKEN_BEAM" \
    --mask_beam "$MASK_BEAM" \
    --n_best "$N_BEST" \
    --device "$GEN_DEVICE"
done

if [[ "$RUN_BASE_ROUTE_EVAL" == "1" ]]; then
  echo "[reproduce] Stage1 base route caches"
  for fam in "${FAMILY_ARRAY[@]}"; do
    base_cache="$REPO_ROOT/outputs/stage1_routes_base/$fam/route_cache.json"
    if [[ "$FORCE_ROUTE_REBUILD" != "1" && -f "$base_cache" ]]; then
      echo "[reproduce] skip existing base route cache: $fam"
      continue
    fi
    "$PYTHON_BIN" stage1_retrosynthesis/build_route_cache.py \
      --repo_root "$REPO_ROOT" \
      --family "$fam" \
      --checkpoint "$family_base_ckpt" \
      --output "$REPO_ROOT/outputs/stage1_routes_base/$fam" \
      --aug "$AUG" \
      --topk "$TOPK" \
      --repos_beam "$REPOS_BEAM" \
      --token_beam "$TOKEN_BEAM" \
      --mask_beam "$MASK_BEAM" \
      --n_best "$N_BEST" \
      --device "$GEN_DEVICE"
  done
fi

echo "[reproduce] Stage2/Stage3 mainline"
PYTHON_BIN="$PYTHON_BIN" \
FAMILIES="$FAMILIES" \
FORCE_REBUILD="$FORCE_STAGE23_REBUILD" \
bash "$REPO_ROOT/scripts/run_stage23_non_oracle_suite.sh" "$REPO_ROOT"

echo "[reproduce] collect checklist statistics"
"$PYTHON_BIN" scripts/collect_checklist_stats.py \
  --repo_root "$REPO_ROOT" \
  --families "$FAMILIES" \
  --output_root "$REPO_ROOT/outputs/checklist_stats" \
  --route_root "$REPO_ROOT/outputs/stage1_routes" \
  --base_route_root "$REPO_ROOT/outputs/stage1_routes_base" \
  --mainline_root "$REPO_ROOT/outputs/stage23_mainline"

echo "[reproduce] done"
