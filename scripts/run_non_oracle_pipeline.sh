#!/bin/bash
#
# Legacy ProSys neural-V2 Non-Oracle pipeline.
# For the current KNN+XGBoost mainline, use `scripts/run_stage23_non_oracle_suite.sh`.
#
# Prereqs: Stage 1 family checkpoints (stage1_retrosynthesis/results/family_finetune/) and
# Stage 2 artifacts (outputs/stage2_v2/<family>/{memory,train}) already exist.
#
# Usage:
#   conda activate ProSys
#   bash scripts/run_non_oracle_pipeline.sh [repo_root]
#
# Tunables (env): FAMILIES (comma/space list; default all with a finetune ckpt),
#   AUG, TOPK, REPOS_BEAM, TOKEN_BEAM, N_BEST, GEN_DEVICE, EVAL_DEVICE, MAX_PRODUCTS.

set -uo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
AUG="${AUG:-10}"
TOPK="${TOPK:-10}"
REPOS_BEAM="${REPOS_BEAM:-5}"
TOKEN_BEAM="${TOKEN_BEAM:-2}"
MASK_BEAM="${MASK_BEAM:-1}"
N_BEST="${N_BEST:-10}"
GEN_DEVICE="${GEN_DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
MAX_PRODUCTS="${MAX_PRODUCTS:-}"

if [[ -n "${FAMILIES:-}" ]]; then
  IFS=', ' read -r -a FAMS <<< "$FAMILIES"
else
  FAMS=()
  for d in stage1_retrosynthesis/results/family_finetune/REAXYS_*_SINGLE_CATMERGE; do
    [[ -d "$d" ]] || continue
    fam="$(basename "$d")"; fam="${fam#REAXYS_}"; fam="${fam%_SINGLE_CATMERGE}"
    FAMS+=("$fam")
  done
fi

echo "[non-oracle-pipeline] families: ${FAMS[*]}"
echo "[non-oracle-pipeline] start $(date)"

max_products_arg=()
[[ -n "$MAX_PRODUCTS" ]] && max_products_arg=(--max_products "$MAX_PRODUCTS")

# Phase 1: Stage 1 route caches (GPU inference), one family at a time.
for fam in "${FAMS[@]}"; do
  cache="outputs/stage1_routes/$fam/route_cache.json"
  if [[ -f "$cache" ]]; then
    echo "[non-oracle-pipeline] reuse route cache: $cache"
    continue
  fi
  echo "[non-oracle-pipeline] generating route cache for $fam $(date)"
  "$PYTHON_BIN" stage1_retrosynthesis/build_route_cache.py \
    --repo_root "$REPO_ROOT" --family "$fam" \
    --aug "$AUG" --topk "$TOPK" --repos_beam "$REPOS_BEAM" \
    --token_beam "$TOKEN_BEAM" --mask_beam "$MASK_BEAM" --n_best "$N_BEST" \
    --device "$GEN_DEVICE" "${max_products_arg[@]}" \
    || echo "[non-oracle-pipeline] WARN: route cache failed for $fam (continuing)"
done

# Phase 2: Stage 2 Non-Oracle evaluation (all families with a cache).
echo "[non-oracle-pipeline] Non-Oracle evaluation $(date)"
"$PYTHON_BIN" scripts/run_stage2_v2_non_oracle.py \
  --repo_root "$REPO_ROOT" --families all --device "$EVAL_DEVICE"

# Phase 3: combined summary.
echo "[non-oracle-pipeline] summary $(date)"
"$PYTHON_BIN" scripts/summarize_hitrates.py \
  --repo_root "$REPO_ROOT" \
  --json_out stage1_retrosynthesis/results/full_pipeline/hitrate_summary.json \
  | tee stage1_retrosynthesis/results/full_pipeline/hitrate_summary.txt

echo "[non-oracle-pipeline] done $(date)"
