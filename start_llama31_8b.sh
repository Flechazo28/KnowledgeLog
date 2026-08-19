#!/usr/bin/env bash

set -euo pipefail

# 使用说明：
#   bash ./start_llama31_8b.sh 0
#   bash ./start_llama31_8b.sh 7
#
# 参数必须是 0 到 7 中的一张显卡。模型通过 OpenAI 兼容接口
# http://127.0.0.1:8001/v1 提供服务。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/home/public_models/llm/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
VLLM_PYTHON="/home/zdw25/miniconda3/envs/knowledgelog-vllm/bin/python"
CUDA_HOME="/home/zdw25/miniconda3/envs/knowledgelog-vllm/lib/python3.11/site-packages/nvidia/cu13"
GPU_ID="${1:-0}"
PORT=8001

if [[ ! "${GPU_ID}" =~ ^[0-7]$ ]]; then
    echo "用法：bash $0 {0|1|2|3|4|5|6|7}" >&2
    exit 2
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "模型目录不存在：${MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "vLLM Python 不存在：${VLLM_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "nvcc 不存在：${CUDA_HOME}/bin/nvcc" >&2
    exit 1
fi

if pgrep -u "$(id -u)" -f "[v]llm\.entrypoints\.openai\.api_server.*--port[ =]${PORT}" >/dev/null; then
    echo "端口 ${PORT} 已有当前用户的 vLLM 服务，请先关闭后再启动。" >&2
    exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_FILE="${SCRIPT_DIR}/llama31_8b_vllm_gpu_${GPU_ID}.log"

echo "使用 GPU：${GPU_ID}"
echo "模型目录：${MODEL_DIR}"
echo "服务地址：http://127.0.0.1:${PORT}/v1"
echo "启动日志：${LOG_FILE}"

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.50 \
    --max-model-len 8192 \
    2>&1 | tee "${LOG_FILE}"
