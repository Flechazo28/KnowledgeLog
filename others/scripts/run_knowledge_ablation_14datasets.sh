#!/usr/bin/env bash
set -euo pipefail

# Run KnowledgeLog ablation experiments 2 and 3 on the 14 full LogHub-2.0
# datasets. The ablations are controlled only through environment variables
# consumed by benchmark/logparser/KnowledgeLog/gpt_query.py.
#
# Active ablation definitions:
# 2. w/o pre-processing:
#    keep prompt-level domain knowledge, disable preprocessing rules.
# 3. w/o domain knowledge in prompt:
#    keep preprocessing rules, disable prompt-level domain knowledge.
#
# The output-format cleanup group "safe_format" is disabled in both active
# settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/benchmark/evaluation"

DEFAULT_PYTHON="/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${DEFAULT_PYTHON}" ]]; then
        PYTHON_BIN="${DEFAULT_PYTHON}"
    else
        PYTHON_BIN="python3"
    fi
fi

MODEL="${MODEL:-gpt-3.5-turbo}"
DATASETS="${DATASETS:-Proxifier,Apache,OpenSSH,HDFS,OpenStack,HPC,Zookeeper,HealthApp,Hadoop,Spark,BGL,Linux,Mac,Thunderbird}"

DEFAULT_PROMPT_GROUPS="core,structure,semantic_role,literal"
DEFAULT_PREPROCESSING_GROUPS="safe_url_endpoint,safe_request_fields,safe_identifier,safe_lifetime_duration,safe_empty_kv_fields"
DEFAULT_POSTPROCESSING_GROUPS="safe_format"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/result/result_datasets/knowledge_ablation_14datasets_${TS}}"
mkdir -p "${RUN_ROOT}"

cat > "${RUN_ROOT}/README.md" <<EOF
# KnowledgeLog 14-Dataset Ablation Run

- Created at: ${TS}
- Project root: ${PROJECT_ROOT}
- Evaluation entry: ${EVAL_DIR}/KnowledgeLog_eval.py
- Python: ${PYTHON_BIN}
- Model: ${MODEL}
- Datasets: ${DATASETS}
- Data split: full

## Experiments

2. \`02_wo_preprocessing\`: prompt domain knowledge on, preprocessing off, safe output-format cleanup off.
3. \`03_wo_domain_knowledge_in_prompt\`: prompt domain knowledge off, preprocessing on, safe output-format cleanup off.

Each experiment writes results under its own \`result/\` directory and records stdout/stderr in \`run.log\`.
EOF

run_experiment() {
    local exp_name="$1"
    local description="$2"
    local prompt_groups="$3"
    local preprocessing_groups="$4"
    local postprocessing_groups="$5"

    local exp_dir="${RUN_ROOT}/${exp_name}"
    local result_dir="${exp_dir}/result"
    mkdir -p "${result_dir}"

    cat > "${exp_dir}/config.env" <<EOF
experiment=${exp_name}
description=${description}
MODEL=${MODEL}
DATASETS=${DATASETS}
LILAC_PROMPT_GROUPS=${prompt_groups}
LILAC_PREPROCESSING_GROUPS=${preprocessing_groups}
LILAC_POSTPROCESSING_GROUPS=${postprocessing_groups}
output_dir=${result_dir}
EOF

    echo "============================================================"
    echo "[$(date '+%F %T')] Start ${exp_name}"
    echo "${description}"
    echo "Output: ${result_dir}"
    echo "============================================================"

    set +e
    (
        cd "${EVAL_DIR}"
        LILAC_PROMPT_GROUPS="${prompt_groups}" \
        LILAC_PREPROCESSING_GROUPS="${preprocessing_groups}" \
        LILAC_POSTPROCESSING_GROUPS="${postprocessing_groups}" \
        "${PYTHON_BIN}" KnowledgeLog_eval.py \
            --full_data \
            --model "${MODEL}" \
            --datasets "${DATASETS}" \
            --output_dir "${result_dir}"
    ) 2>&1 | tee "${exp_dir}/run.log"
    local status="${PIPESTATUS[0]}"
    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] ${exp_name} failed with exit code ${status}."
        echo "See log: ${exp_dir}/run.log"
        exit "${status}"
    fi

    echo "[$(date '+%F %T')] Finished ${exp_name}"
    echo
}

# Experiment 1 has already been run and is intentionally disabled here.
# run_experiment \
#     "01_wo_dual_knowledge_supplementation" \
#     "Disable both prompt-level domain knowledge and preprocessing rules." \
#     "" \
#     "" \
#     "${DEFAULT_POSTPROCESSING_GROUPS}"

run_experiment \
    "02_wo_preprocessing" \
    "Keep prompt-level domain knowledge and disable preprocessing rules and safe output-format cleanup." \
    "${DEFAULT_PROMPT_GROUPS}" \
    "" \
    ""

run_experiment \
    "03_wo_domain_knowledge_in_prompt" \
    "Keep preprocessing rules and disable prompt-level domain knowledge and safe output-format cleanup." \
    "" \
    "${DEFAULT_PREPROCESSING_GROUPS}" \
    ""

{
    echo "Run root: ${RUN_ROOT}"
    echo
    echo "Summary files:"
    find "${RUN_ROOT}" -path "*/result/summary_*.csv" -print | sort
} | tee "${RUN_ROOT}/summary_paths.txt"

echo
echo "All ablation experiments finished."
echo "Run root: ${RUN_ROOT}"
