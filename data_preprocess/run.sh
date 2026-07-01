#!/bin/bash
# ============================================================
# Reaxys 反应数据全流程预处理
#
# 用法：
#   bash run.sh                         # 首次运行（Stage 2）
#   bash run.sh --stage1                # 含 Stage 1 逆合成
#   bash run.sh --stage1 --uspto        # Stage 1 + USPTO 过滤
#   bash run.sh --skip-clean            # 跳过原始清理
#   bash run.sh --stage1 --uspto --skip-clean  # 全套跳过清理
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$PROJECT_DIR/data/reaxys_input"
OUTPUT_DIR="$PROJECT_DIR/data"

DO_STAGE1=""
SKIP_CLEAN=""
DO_USPTO=""

# 解析参数
for arg in "$@"; do
    case $arg in
        --stage1)    DO_STAGE1="--do_stage1" ;;
        --skip-clean) SKIP_CLEAN="--skip_raw_clean" ;;
        --uspto)     DO_USPTO="--process_uspto" ;;
        *)           echo "未知参数: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo " Reaxys 数据全流程预处理"
echo "============================================"
echo " 输入目录: $INPUT_DIR"
echo " 输出目录: $OUTPUT_DIR"
echo " Stage 1:  ${DO_STAGE1:+是}${DO_STAGE1:-否}"
echo " USPTO:    ${DO_USPTO:+是}${DO_USPTO:-否}"
echo " 原始清理: ${SKIP_CLEAN:+跳过}${SKIP_CLEAN:-是}"
echo "============================================"
echo ""

cd "$PROJECT_DIR"

python "$SCRIPT_DIR/preprocess.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $SKIP_CLEAN \
    $DO_STAGE1 \
    $DO_USPTO

echo ""
echo "============================================"
echo " 完成！"
echo "============================================"
echo " Stage 2 输出: $OUTPUT_DIR/reaction_processed_{family}_catmerge/"
if [ -n "$DO_STAGE1" ]; then
    echo " Stage 1 输出: $OUTPUT_DIR/editretro/datasets/"
fi
