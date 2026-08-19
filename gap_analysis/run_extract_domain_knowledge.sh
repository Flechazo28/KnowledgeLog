#!/usr/bin/env bash

set -euo pipefail

# 中文使用说明：
# 1. 先进入项目根目录并激活运行环境：
#      cd /home/zdw25/Project/LogHub/Myself/KnowledgeLog-main
#      conda activate knowledgelog
# 2. 脚本会自动读取 results/classification 下最新目录中的
#    classification_results_correction.json，无需传入分类文件路径。
# 3. 正式调用模型提取候选领域知识：
#      bash gap_analysis/run_extract_domain_knowledge.sh
# 4. 只检查数据、批次和 Prompt，不调用模型：
#      bash gap_analysis/run_extract_domain_knowledge.sh --dry-run
# 5. 指定模型：
#      KNOWLEDGELOG_MODEL=Llama-3.3-70B-Instruct \
#        bash gap_analysis/run_extract_domain_knowledge.sh
#
# 默认输出目录：
#   gap_analysis/results/knowledge/时间戳_领域知识数量/
#   domain_knowledge_for_review.json

usage() {
    echo "用法：bash gap_analysis/run_extract_domain_knowledge.sh [--dry-run]"
    echo "运行目录：/home/zdw25/Project/LogHub/Myself/KnowledgeLog-main"
    echo "输入：自动选择 results/classification 下最新的 classification_results_correction.json"
    echo
    echo "环境变量："
    echo "  KNOWLEDGELOG_PYTHON  Python 命令或绝对路径，默认：python"
    echo "  KNOWLEDGELOG_MODEL   模型名称，默认：Llama-3.3-70B-Instruct"
    echo
    echo "示例："
    echo "  bash gap_analysis/run_extract_domain_knowledge.sh"
    echo "  bash gap_analysis/run_extract_domain_knowledge.sh --dry-run"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFICATION_DIR="$SCRIPT_DIR/results/classification"
PYTHON_BIN="${KNOWLEDGELOG_PYTHON:-python}"
MODEL="${KNOWLEDGELOG_MODEL:-Llama-3.3-70B-Instruct}"

INPUT_JSON="$(
    find "$CLASSIFICATION_DIR" -mindepth 2 -maxdepth 2 -type f \
        -name "classification_results_correction.json" -print \
        | sort -r | sed -n '1p'
)"

if [[ -z "$INPUT_JSON" || ! -f "$INPUT_JSON" ]]; then
    echo "错误：$CLASSIFICATION_DIR 中没有找到 classification_results_correction.json" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：找不到 Python：$PYTHON_BIN" >&2
    exit 1
fi

echo "开始提取候选领域知识"
echo "Python：$PYTHON_BIN"
echo "模型：$MODEL"
echo "输入：$INPUT_JSON"

"$PYTHON_BIN" -u "$SCRIPT_DIR/extract_domain_knowledge.py" \
    --input "$INPUT_JSON" \
    --model "$MODEL" \
    "$@"
