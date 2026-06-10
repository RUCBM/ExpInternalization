#!/bin/bash
# Start a vLLM server that exposes the local student model as an OpenAI-compatible API.
#
# Usage:   bash start_vllm.sh [GPU_IDS] [PORT]
#   e.g.:  bash start_vllm.sh 0,1 8100
#
# Override the model / served name via env vars:
#   MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 SERVED_MODEL_NAME=qwen3-4b bash start_vllm.sh 0,1 8100

GPU_IDS="${1:-0,1}"
PORT="${2:-8100}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
# Python interpreter that has vLLM installed (defaults to the active one).
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "=========================================="
echo "  Starting vLLM server"
echo "  Model:        $MODEL_PATH"
echo "  GPUs:         $GPU_IDS ($NUM_GPUS)"
echo "  Port:         $PORT"
echo "  Served name:  $SERVED_MODEL_NAME"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len 32768 \
    --trust-remote-code \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
