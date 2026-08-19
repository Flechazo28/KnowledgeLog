#!/usr/bin/env bash
set -euo pipefail

# 使用说明：
# 1. 先启动需要测试的本地模型，例如：
#      bash ./start_llama70b.sh 0-3
#      bash ./start_llama31_8b.sh 4
# 2. 在另一个终端进入 KnowledgeLog-main 目录并运行：
#      bash ./run_knowledgelog_full_14datasets.sh
#      bash ./run_knowledgelog_full_14datasets.sh Llama-3.1-8B-Instruct
#    也可以通过 KNOWLEDGELOG_MODEL 环境变量选择模型。
# 3. 脚本会依次测试 full_dataset 中的14个数据集。
# 4. 正常结束、运行失败或按 Ctrl+C 中断时，脚本都会关闭运行前检测到的
#    当前用户在 8001 端口启动的 vLLM 服务并释放显卡。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/benchmark/evaluation"
PYTHON_BIN="/home/zdw25/miniconda3/envs/knowledgelog/bin/python"
MODEL="${1:-${KNOWLEDGELOG_MODEL:-Llama-3.3-70B-Instruct}}"
API_URL="http://127.0.0.1:8001/v1/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="${MODEL//[^A-Za-z0-9._-]/_}"
OUTPUT_DIR="${SCRIPT_DIR}/result/KnowledgeLog_full_14datasets_${MODEL_TAG}_${TIMESTAMP}"
RUN_LOG="${OUTPUT_DIR}/run.log"

if [[ "$#" -gt 1 ]]; then
    echo "用法：bash $0 [模型名称]" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "错误：找不到 knowledgelog 环境中的 Python：${PYTHON_BIN}" >&2
    exit 1
fi

mapfile -t MODEL_PIDS < <(
    pgrep -u "$(id -u)" -f \
        '[v]llm\.entrypoints\.openai\.api_server.*--port[ =]8001' || true
)

if [[ "${#MODEL_PIDS[@]}" -eq 0 ]]; then
    echo "错误：没有找到当前用户在 8001 端口运行的 vLLM 服务。" >&2
    echo "70B 启动示例：bash ./start_llama70b.sh 0-3" >&2
    echo "8B 启动示例：bash ./start_llama31_8b.sh 4" >&2
    exit 1
fi

declare -A MODEL_START_TIMES=()
for pid in "${MODEL_PIDS[@]}"; do
    if [[ -r "/proc/${pid}/stat" ]]; then
        MODEL_START_TIMES["${pid}"]="$(awk '{print $22}' "/proc/${pid}/stat")"
    fi
done

same_model_process() {
    local pid="$1"
    [[ -r "/proc/${pid}/stat" ]] &&
        [[ "$(awk '{print $22}' "/proc/${pid}/stat")" == "${MODEL_START_TIMES[${pid}]:-}" ]]
}

stop_local_model() {
    local pid
    local -a alive=()

    echo
    echo "正在关闭本次实验使用的 ${MODEL} 服务……"
    for pid in "${MODEL_PIDS[@]}"; do
        if same_model_process "${pid}"; then
            alive+=("${pid}")
        fi
    done

    if [[ "${#alive[@]}" -eq 0 ]]; then
        echo "模型服务已经结束，无需再次关闭。"
        return
    fi

    kill -TERM "${alive[@]}" 2>/dev/null || true
    for _ in {1..30}; do
        alive=()
        for pid in "${MODEL_PIDS[@]}"; do
            if same_model_process "${pid}"; then
                alive+=("${pid}")
            fi
        done
        [[ "${#alive[@]}" -eq 0 ]] && break
        sleep 1
    done

    if [[ "${#alive[@]}" -gt 0 ]]; then
        echo "模型未在30秒内退出，正在强制结束：${alive[*]}"
        kill -KILL "${alive[@]}" 2>/dev/null || true
    fi
    echo "模型服务已关闭，显卡资源已释放。"
}

trap stop_local_model EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! curl -fsS --max-time 10 "${API_URL}" | grep -Fq "${MODEL}"; then
    echo "错误：8001 端口未提供模型 ${MODEL}。" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "模型：${MODEL}"
echo "数据：${SCRIPT_DIR}/full_dataset（14个数据集）"
echo "结果：${OUTPUT_DIR}"
echo "日志：${RUN_LOG}"
echo

set +e
(
    cd "${EVAL_DIR}"
    "${PYTHON_BIN}" -u KnowledgeLog_eval.py \
        --full_data \
        --model "${MODEL}" \
        --output_dir "${OUTPUT_DIR}"
) 2>&1 | tee "${RUN_LOG}"
RUN_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${RUN_STATUS}" -eq 0 ]]; then
    echo
    echo "14个数据集运行完成。"
    echo "实验结果：${OUTPUT_DIR}"
else
    echo
    echo "实验运行失败，退出码：${RUN_STATUS}" >&2
    echo "请查看日志：${RUN_LOG}" >&2
fi

exit "${RUN_STATUS}"
