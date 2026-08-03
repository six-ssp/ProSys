#!/bin/bash
#
# Current ProSys Stage-2/Stage-3 Non-Oracle mainline suite.
# Mainline: Stage 1 route cache -> Stage 2 KNN candidate pool -> Stage 3 no-GNN
# XGBoost reranking + validation-gated reaction-GNN temperature regression.
#
# Usage:
#   conda activate ProSys
#   bash scripts/run_stage23_non_oracle_suite.sh [repo_root]
#
# Tunables (env):
#   FAMILIES            comma/space separated families, default all
#   OUTPUT_ROOT         default outputs/stage23_mainline
#   ROUTE_ROOT          default outputs/stage1_routes
#   FPSIZE              default 4096
#   RADIUS              default 2
#   KNN_TOP_K           default 64
#   PREFILTER_CONTEXTS  default 64
#   MAX_CONTEXTS        default 20
#   MAX_TRAIN_ROUTES    default 0 (= full train split)
#   MAX_VAL_ROUTES      default 0 (= full val split)
#   TRAIN_TABLE_MODE    default oracle; one of oracle / mixed_hard_negative / non_oracle
#   TRAIN_ROUTE_ROOT    optional route-cache root for train hard negatives / non-oracle train
#   VAL_ROUTE_ROOT      default outputs/stage1_routes_validation; also gates the GNN temperature branch
#   HARD_NEGATIVE_PER_SAMPLE default 8
#   FORCE_REBUILD       set to 1 to rebuild candidate tables and model artifacts
#   REAFNN_DEVICE       default cpu
#   GNN_DEVICE          default cpu

#   GNN_TEMPERATURE_MIN_VAL_MAE_IMPROVEMENT default 0.25 C
#   REUSE_CANDIDATE_TABLES_ROOT optional existing Stage 2/3 table root, avoiding candidate regeneration
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-}"
if [[ ! "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=8
fi
FAMILIES="${FAMILIES:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage23_mainline}"
ROUTE_ROOT="${ROUTE_ROOT:-outputs/stage1_routes}"
FPSIZE="${FPSIZE:-4096}"
RADIUS="${RADIUS:-2}"
KNN_TOP_K="${KNN_TOP_K:-64}"
PREFILTER_CONTEXTS="${PREFILTER_CONTEXTS:-64}"
MAX_CONTEXTS="${MAX_CONTEXTS:-20}"
MAX_TRAIN_ROUTES="${MAX_TRAIN_ROUTES:-0}"
MAX_VAL_ROUTES="${MAX_VAL_ROUTES:-0}"
TRAIN_TABLE_MODE="${TRAIN_TABLE_MODE:-oracle}"
TRAIN_ROUTE_ROOT="${TRAIN_ROUTE_ROOT:-}"
VAL_ROUTE_ROOT="${VAL_ROUTE_ROOT:-outputs/stage1_routes_validation}"
HARD_NEGATIVE_PER_SAMPLE="${HARD_NEGATIVE_PER_SAMPLE:-8}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
REAFNN_DEVICE="${REAFNN_DEVICE:-cpu}"
GNN_DEVICE="${GNN_DEVICE:-cpu}"

echo "[stage23-suite] repo_root=$REPO_ROOT"
GNN_TEMPERATURE_MIN_VAL_MAE_IMPROVEMENT="${GNN_TEMPERATURE_MIN_VAL_MAE_IMPROVEMENT:-0.25}"
REUSE_CANDIDATE_TABLES_ROOT="${REUSE_CANDIDATE_TABLES_ROOT:-}"
echo "[stage23-suite] output_root=$OUTPUT_ROOT families=$FAMILIES"
export OMP_NUM_THREADS

cmd=(
  "$PYTHON_BIN" scripts/run_stage23_mainline_non_oracle.py
  --repo_root "$REPO_ROOT" \
  --families "$FAMILIES" \
  --output_root "$OUTPUT_ROOT" \
  --route_root "$ROUTE_ROOT" \
  --fpsize "$FPSIZE" \
  --radius "$RADIUS" \
  --knn_top_k "$KNN_TOP_K" \
  --prefilter_contexts "$PREFILTER_CONTEXTS" \
  --max_contexts "$MAX_CONTEXTS" \
  --max_train_routes "$MAX_TRAIN_ROUTES" \
  --max_val_routes "$MAX_VAL_ROUTES" \
  --train_table_mode "$TRAIN_TABLE_MODE" \
  --hard_negative_per_sample "$HARD_NEGATIVE_PER_SAMPLE" \
  --reafnn_device "$REAFNN_DEVICE" \
  --gnn_device "$GNN_DEVICE" \
  --gnn_temperature_min_val_mae_improvement "$GNN_TEMPERATURE_MIN_VAL_MAE_IMPROVEMENT"
)

if [[ -n "$TRAIN_ROUTE_ROOT" ]]; then
  cmd+=(--train_route_root "$TRAIN_ROUTE_ROOT")
fi

if [[ -n "$VAL_ROUTE_ROOT" ]]; then
  cmd+=(--val_route_root "$VAL_ROUTE_ROOT")
fi

if [[ -n "$REUSE_CANDIDATE_TABLES_ROOT" ]]; then
  cmd+=(--reuse_candidate_tables_root "$REUSE_CANDIDATE_TABLES_ROOT")
fi

if [[ "$FORCE_REBUILD" == "1" ]]; then
  cmd+=(--force_rebuild)
fi

"${cmd[@]}"

echo "[stage23-suite] done"
