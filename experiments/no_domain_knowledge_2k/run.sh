#!/usr/bin/env bash
set -euo pipefail

# 使用说明：
#   1. 先确保本地 Llama-3.3-70B 服务运行在 http://127.0.0.1:8001/v1。
#   2. 在 KnowledgeLog-main 根目录执行：
#        bash experiments/no_domain_knowledge_2k/run.sh
#
# 本实验只关闭 gpt_query.py 中的两类领域知识：
#   - 自然语言领域知识；
#   - 预处理正则领域知识。
# 基础提示词、模板缓存、common.json 和现有后处理逻辑保持启用。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/benchmark/evaluation"
PYTHON_BIN="/home/zdw25/miniconda3/envs/knowledgelog/bin/python"
MODEL="Llama-3.3-70B-Instruct"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/result/KnowledgeLog_2k_no_domain_knowledge_${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "$$" > "${OUTPUT_DIR}/run.pid"

cat > "${OUTPUT_DIR}/experiment_config.txt" <<EOF
experiment=no_domain_knowledge_2k
model=${MODEL}
dataset=2k_datasets
prompt_domain_knowledge=disabled
regex_domain_knowledge=disabled
common_json=enabled
postprocessing=enabled
output_dir=${OUTPUT_DIR}
started_at=$(date '+%F %T %z')
EOF

echo "实验：2k_datasets，不使用自然语言和正则领域知识"
echo "模型：${MODEL}"
echo "结果：${OUTPUT_DIR}"

cd "${EVAL_DIR}"
set +e
LILAC_PROMPT_GROUPS=none \
LILAC_PREPROCESSING_GROUPS=none \
"${PYTHON_BIN}" -u KnowledgeLog_eval.py \
    --model "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e

exit "${RUN_STATUS}"
