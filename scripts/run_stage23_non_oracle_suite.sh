#!/bin/bash
#
# Current ProSys Stage-2/Stage-3 Non-Oracle mainline suite.
# Mainline: Stage 1 route cache -> Stage 2 KNN candidate pool -> Stage 3 XGBoost
# tabular reranking + fixed R-GNN temperature regression.
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
#   KNN_RETRIEVAL_MODE default product_morgan
#   KNN_TOP_K           default 64
#   PREFILTER_CONTEXTS  default 64
#   MAX_CONTEXTS        default 20
#   REAFNN_HIDDEN_DIM / REAFNN_HIDDEN_LAYERS / REAFNN_DROPOUT
#                       defaults 512 / 2 / 0.10
#   REAFNN_KNN_ANCHORS / REAFNN_CORRECTION_WEIGHT / REAFNN_CORRECTION_CLIP
#                       defaults 12 / 0.65 / 0.35
#   MAX_TRAIN_ROUTES    default 0 (= full train split)
#   MAX_VAL_ROUTES      default 0 (= full val split)
#   TRAIN_TABLE_MODE    default oracle; one of oracle / mixed_hard_negative / non_oracle
#   TRAIN_ROUTE_ROOT    optional route-cache root for train hard negatives / non-oracle train
#   VAL_ROUTE_ROOT      default outputs/stage1_routes_validation; used when non-oracle validation tables are requested
#   HARD_NEGATIVE_PER_SAMPLE default 8
#   FORCE_REBUILD       set to 1 to rebuild candidate tables and model artifacts
#   REAFNN_DEVICE       default cpu
#   GNN_DEVICE          default cpu
#   REUSE_CANDIDATE_TABLES_ROOT optional matching Stage 2 table root; R-GNN features are always regenerated
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
KNN_RETRIEVAL_MODE="${KNN_RETRIEVAL_MODE:-product_morgan}"
KNN_TOP_K="${KNN_TOP_K:-64}"
PREFILTER_CONTEXTS="${PREFILTER_CONTEXTS:-64}"
MAX_CONTEXTS="${MAX_CONTEXTS:-20}"
REAFNN_HIDDEN_DIM="${REAFNN_HIDDEN_DIM:-512}"
REAFNN_HIDDEN_LAYERS="${REAFNN_HIDDEN_LAYERS:-2}"
REAFNN_DROPOUT="${REAFNN_DROPOUT:-0.10}"
REAFNN_KNN_ANCHORS="${REAFNN_KNN_ANCHORS:-12}"
REAFNN_CORRECTION_WEIGHT="${REAFNN_CORRECTION_WEIGHT:-0.65}"
REAFNN_CORRECTION_CLIP="${REAFNN_CORRECTION_CLIP:-0.35}"
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
  --knn_retrieval_mode "$KNN_RETRIEVAL_MODE" \
  --knn_top_k "$KNN_TOP_K" \
  --prefilter_contexts "$PREFILTER_CONTEXTS" \
  --max_contexts "$MAX_CONTEXTS" \
  --reafnn_hidden_dim "$REAFNN_HIDDEN_DIM" \
  --reafnn_hidden_layers "$REAFNN_HIDDEN_LAYERS" \
  --reafnn_dropout "$REAFNN_DROPOUT" \
  --reafnn_knn_anchor_contexts "$REAFNN_KNN_ANCHORS" \
  --reafnn_correction_weight "$REAFNN_CORRECTION_WEIGHT" \
  --reafnn_correction_clip "$REAFNN_CORRECTION_CLIP" \
  --reafnn_enable_knn_wide_refinement \
  --max_train_routes "$MAX_TRAIN_ROUTES" \
  --max_val_routes "$MAX_VAL_ROUTES" \
  --train_table_mode "$TRAIN_TABLE_MODE" \
  --hard_negative_per_sample "$HARD_NEGATIVE_PER_SAMPLE" \
  --reafnn_device "$REAFNN_DEVICE" \
  --gnn_device "$GNN_DEVICE"
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
