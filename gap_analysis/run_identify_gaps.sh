#!/usr/bin/env bash

set -euo pipefail

# 中文使用说明：
# 1. 先进入项目根目录并激活运行环境：
#      cd /home/zdw25/Project/LogHub/Myself/KnowledgeLog-main
#      conda activate knowledgelog
# 2. 如果使用本地 Llama 模型，请先启动 OpenAI 兼容服务，并确保
#    KnowledgeLog-main/openai_key.txt 中配置了对应的模型名称和 API 地址。
# 3. 默认在全部 32shot 数据集上识别知识缺口：
#      bash gap_analysis/run_identify_gaps.sh
# 4. 指定其他 shot：
#      KNOWLEDGELOG_SHOT=64 bash gap_analysis/run_identify_gaps.sh
# 5. 指定模型：
#      KNOWLEDGELOG_MODEL=Llama-3.3-70B-Instruct \
#        bash gap_analysis/run_identify_gaps.sh
# 6. 追加 Python 程序支持的参数，例如覆盖本地 API 地址：
#      bash gap_analysis/run_identify_gaps.sh \
#        --api-base http://127.0.0.1:8001/v1 \
#        --api-key EMPTY
#
# 默认输出目录：
#   gap_analysis/results/knowledge_gaps/时间戳_32shot/IR_results.json

usage() {
    echo "用法：bash gap_analysis/run_identify_gaps.sh [identify_gaps.py 的额外参数]"
    echo "运行目录：/home/zdw25/Project/LogHub/Myself/KnowledgeLog-main"
    echo
    echo "环境变量："
    echo "  KNOWLEDGELOG_PYTHON  Python 命令或绝对路径，默认：python"
    echo "  KNOWLEDGELOG_MODEL   模型名称，默认：Llama-3.3-70B-Instruct"
    echo "  KNOWLEDGELOG_SHOT    shot 数，可选 8/16/32/64/128，默认：32"
    echo
    echo "示例："
    echo "  bash gap_analysis/run_identify_gaps.sh"
    echo "  KNOWLEDGELOG_SHOT=64 bash gap_analysis/run_identify_gaps.sh"
    echo "  bash gap_analysis/run_identify_gaps.sh --api-base http://127.0.0.1:8001/v1 --api-key EMPTY"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${KNOWLEDGELOG_PYTHON:-python}"
MODEL="${KNOWLEDGELOG_MODEL:-Llama-3.3-70B-Instruct}"
SHOT="${KNOWLEDGELOG_SHOT:-32}"

case "$SHOT" in
    8|16|32|64|128) ;;
    *)
        echo "错误：KNOWLEDGELOG_SHOT 只能是 8、16、32、64 或 128。" >&2
        exit 2
        ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：找不到 Python：$PYTHON_BIN" >&2
    exit 1
fi

echo "开始识别知识缺口"
echo "Python：$PYTHON_BIN"
echo "模型：$MODEL"
echo "Shot：$SHOT"

"$PYTHON_BIN" -u "$SCRIPT_DIR/identify_gaps.py" \
    --sampled-examples-dir "$SCRIPT_DIR/sampled_examples" \
    --shot "$SHOT" \
    --model "$MODEL" \
    "$@"
