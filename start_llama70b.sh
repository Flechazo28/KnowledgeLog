#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash start_llama70b.sh 0-3
#   bash start_llama70b.sh 4-7
#
# The model is served through an OpenAI-compatible vLLM API on port 8001.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/home/public_models/llm/Llama-3.3-70B-Instruct"
VLLM_PYTHON="/home/zdw25/miniconda3/envs/knowledgelog-vllm/bin/python"
CUDA_HOME="/home/zdw25/miniconda3/envs/knowledgelog-vllm/lib/python3.11/site-packages/nvidia/cu13"
GPU_GROUP="${1:-0-3}"

case "$GPU_GROUP" in
    0-3|0123)
        GPU_IDS="0,1,2,3"
        ;;
    4-7|4567)
        GPU_IDS="4,5,6,7"
        ;;
    *)
        echo "用法: bash $0 {0-3|4-7}" >&2
        exit 2
        ;;
esac

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "模型目录不存在: $MODEL_DIR" >&2
    exit 1
fi

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "vLLM Python 不存在: $VLLM_PYTHON" >&2
    exit 1
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "nvcc 不存在: $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_FILE="$SCRIPT_DIR/llama70b_vllm_gpu_${GPU_GROUP//,/_}.log"

echo "使用 GPU: $GPU_IDS"
echo "模型目录: $MODEL_DIR"
echo "服务地址: http://127.0.0.1:8001/v1"
echo "启动日志: $LOG_FILE"

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    2>&1 | tee "$LOG_FILE"
