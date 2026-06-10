#!/bin/bash
# ============================================================================
# Experience summarization (training-free GRPO style), fully local.
#
# Starts a vLLM server for the student model, then runs the experience-
# extraction loop: rollout (agent + search tools) -> verify -> ExperienceUpdater.
# The SAME local model is used for both rollout and experience update.
#
# Usage:
#   bash scripts/run_experience_summarize.sh
#
# Common overrides (a 4B model fits on 1-2 GPUs):
#   GPU_IDS=0,1 VLLM_PORT=8100 EXPERIMENT_NAME=my_run \
#       bash scripts/run_experience_summarize.sh
#
# Requires (export before running, or set in .env):
#   SERPER_API_KEY  - web search   (https://serper.dev)
#   JINA_API_KEY    - web reader    (https://jina.ai/reader)
# ============================================================================

set -u
set -o pipefail

# Repo root = parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
VLLM_PORT="${VLLM_PORT:-8100}"
LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-alllocal_qwen3_web}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-qwen3-4b}"
DATA_FILE="${DATA_FILE:-data/train/data_15k_shuffle.jsonl}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/exp_${TS}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  Experience summarization (fully local)"
echo "=========================================="
echo "  Model name:   $LOCAL_MODEL_NAME"
echo "  GPUs:         $GPU_IDS"
echo "  Port:         $VLLM_PORT"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  Data file:    $DATA_FILE"
echo "  Logs:         $LOG_DIR"
echo "=========================================="

# ── Start vLLM in the background ──────────────────────────────────────────
echo "[$(date '+%F %T')] Starting vLLM on GPU $GPU_IDS port $VLLM_PORT..."
bash training_free_grpo/start_vllm.sh "$GPU_IDS" "$VLLM_PORT" > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "[$(date '+%F %T')] vLLM PID=$VLLM_PID  log=$LOG_DIR/vllm.log"

cleanup() {
    local code=$?
    echo "[$(date '+%F %T')] Cleanup: killing vLLM PID=$VLLM_PID..."
    pkill -P "$VLLM_PID" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}" 2>/dev/null || true
    echo "[$(date '+%F %T')] GPU $GPU_IDS released. Exit=$code"
}
trap cleanup EXIT INT TERM

# ── Wait for vLLM to be ready (up to 15 min) ──────────────────────────────
echo "[$(date '+%F %T')] Waiting for vLLM at $LOCAL_BASE_URL/models ..."
for i in $(seq 1 90); do
    if curl -s "${LOCAL_BASE_URL}/models" > /dev/null 2>&1; then
        echo "[$(date '+%F %T')] vLLM ready (after ${i}0s)"
        break
    fi
    if [ "$i" -eq 90 ]; then
        echo "ERROR: vLLM not ready after 15 min. Check $LOG_DIR/vllm.log"
        exit 1
    fi
    sleep 10
done

# ── Bypass any local proxy for the vLLM endpoint ──────────────────────────
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0}"
export NO_PROXY="$no_proxy"

# ── Force the agent's internal tool LLM calls onto the local vLLM too ──────
# (overrides whatever is in .env, e.g. a DeepSeek default)
export UTU_LLM_TYPE="chat.completions"
export UTU_LLM_MODEL="$LOCAL_MODEL_NAME"
export UTU_LLM_BASE_URL="$LOCAL_BASE_URL"
export UTU_LLM_API_KEY="not-needed"

# ── Search-tool API keys (read from environment / .env) ───────────────────
# SERPER_API_KEY / JINA_API_KEY must be set for the search + visit tools.

python -m training_free_grpo.train_experience \
    --mode agent \
    --domain web \
    --experience_backend local \
    --experiment_name "$EXPERIMENT_NAME" \
    --data_file "$DATA_FILE" \
    --dataset_truncate 100 \
    --epochs 3 \
    --batchsize 4 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800 \
    --max_pool_size 50 \
    --local_model_name "$LOCAL_MODEL_NAME" \
    --local_base_url "$LOCAL_BASE_URL" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "[$(date '+%F %T')] Done. Output: data/web/train/${EXPERIMENT_NAME}/"
