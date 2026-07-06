#!/bin/bash
#
# Current ProSys Stage-2/Stage-3 Non-Oracle suite.
# Mainline: Stage 1 route cache -> Stage 2 KNN candidate pool -> Stage 3 XGBoost rerank + temperature
# Also renders the historical baseline and ablation tables from the same result tree.
#
# Usage:
#   conda activate ProSys
#   bash scripts/run_stage23_non_oracle_suite.sh [repo_root]
#
# Tunables (env):
#   FAMILIES            comma/space separated families, default all
#   OUTPUT_ROOT         default outputs/stage23_non_oracle_all10
#   ROUTE_ROOT          default outputs/stage1_routes
#   FPSIZE              default 4096
#   RADIUS              default 2
#   KNN_TOP_K           default 20
#   MAX_CONTEXTS        default 20
#   CLUSTER_NUM         default 32
#   SVD_DIM             default 64
#   LEGACY_MAX_CONTEXTS default 200
#   MAX_TRAIN_ROUTES    default 2000
#   MAX_VAL_ROUTES      default 500

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-}"
if [[ ! "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=8
fi
FAMILIES="${FAMILIES:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage23_non_oracle_all10}"
ROUTE_ROOT="${ROUTE_ROOT:-outputs/stage1_routes}"
FPSIZE="${FPSIZE:-4096}"
RADIUS="${RADIUS:-2}"
KNN_TOP_K="${KNN_TOP_K:-20}"
MAX_CONTEXTS="${MAX_CONTEXTS:-20}"
CLUSTER_NUM="${CLUSTER_NUM:-32}"
SVD_DIM="${SVD_DIM:-64}"
LEGACY_MAX_CONTEXTS="${LEGACY_MAX_CONTEXTS:-200}"
MAX_TRAIN_ROUTES="${MAX_TRAIN_ROUTES:-2000}"
MAX_VAL_ROUTES="${MAX_VAL_ROUTES:-500}"

echo "[stage23-suite] repo_root=$REPO_ROOT"
echo "[stage23-suite] output_root=$OUTPUT_ROOT families=$FAMILIES"
export OMP_NUM_THREADS

"$PYTHON_BIN" baseline/run_non_oracle_stage23_experiments.py \
  --repo_root "$REPO_ROOT" \
  --families "$FAMILIES" \
  --output_root "$OUTPUT_ROOT" \
  --route_root "$ROUTE_ROOT" \
  --fpsize "$FPSIZE" \
  --radius "$RADIUS" \
  --knn_top_k "$KNN_TOP_K" \
  --max_contexts "$MAX_CONTEXTS" \
  --cluster_num "$CLUSTER_NUM" \
  --svd_dim "$SVD_DIM" \
  --legacy_max_contexts "$LEGACY_MAX_CONTEXTS" \
  --max_train_routes "$MAX_TRAIN_ROUTES" \
  --max_val_routes "$MAX_VAL_ROUTES"

"$PYTHON_BIN" baseline/render_stage23_nonoracle_reports.py \
  --output_root "$OUTPUT_ROOT"

echo "[stage23-suite] done"
