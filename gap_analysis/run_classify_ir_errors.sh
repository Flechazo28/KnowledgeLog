#!/usr/bin/env bash

set -euo pipefail

# 中文使用说明：
# 1. 先进入项目根目录并激活运行环境：
#      cd /home/zdw25/Project/LogHub/Myself/KnowledgeLog-main
#      conda activate knowledgelog
# 2. 不传参数时，默认分类最新实验：
#      gap_analysis/results/knowledge_gaps/20260723_153030_32shot/IR_results.json
# 3. 使用默认输入运行：
#      bash gap_analysis/run_classify_ir_errors.sh
# 4. 也可以将第一个参数设置为其他 IR_results.json。
# 5. 指定模型：
#      KNOWLEDGELOG_MODEL=Llama-3.3-70B-Instruct \
#        bash gap_analysis/run_classify_ir_errors.sh
#
# 默认输出目录：
#   gap_analysis/results/classification/时间戳_错误数量/
#   classification_results.json

usage() {
    echo "用法：bash gap_analysis/run_classify_ir_errors.sh [IR_results.json] [额外参数]"
    echo "运行目录：/home/zdw25/Project/LogHub/Myself/KnowledgeLog-main"
    echo "默认输入：gap_analysis/results/knowledge_gaps/20260723_153030_32shot/IR_results.json"
    echo
    echo "环境变量："
    echo "  KNOWLEDGELOG_PYTHON  Python 命令或绝对路径，默认：python"
    echo "  KNOWLEDGELOG_MODEL   模型名称，默认：Llama-3.3-70B-Instruct"
    echo
    echo "示例："
    echo "  bash gap_analysis/run_classify_ir_errors.sh"
    echo "  bash gap_analysis/run_classify_ir_errors.sh gap_analysis/results/knowledge_gaps/时间戳_32shot/IR_results.json"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INPUT_JSON="$SCRIPT_DIR/results/knowledge_gaps/20260723_153030_32shot/IR_results.json"
INPUT_JSON="$DEFAULT_INPUT_JSON"
if [[ $# -gt 0 && "$1" != --* ]]; then
    INPUT_JSON="$1"
    shift
fi

if [[ ! -f "$INPUT_JSON" ]]; then
    echo "错误：找不到输入文件：$INPUT_JSON" >&2
    exit 1
fi

PYTHON_BIN="${KNOWLEDGELOG_PYTHON:-python}"
MODEL="${KNOWLEDGELOG_MODEL:-Llama-3.3-70B-Instruct}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：找不到 Python：$PYTHON_BIN" >&2
    exit 1
fi

echo "开始分类 IR 错误"
echo "Python：$PYTHON_BIN"
echo "模型：$MODEL"
echo "输入：$INPUT_JSON"

"$PYTHON_BIN" -u "$SCRIPT_DIR/classify_ir_errors.py" \
    --input "$INPUT_JSON" \
    --model "$MODEL" \
    "$@"
