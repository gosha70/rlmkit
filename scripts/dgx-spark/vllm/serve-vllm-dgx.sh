#!/usr/bin/env bash
# serve-vllm-dgx.sh — Start the vLLM OpenAI-compatible server on DGX Spark.
#
# DGX Spark uses Unified Memory Architecture (UMA): GPU and CPU share the same
# 128 GB pool.  vLLM's default memory-profiling step can fail here because free
# memory fluctuates during startup (desktop apps, UMA measurement instability),
# triggering an assertion even when the model fits.  This script works around
# that by:
#   1. Flushing the OS page cache before starting (reduces UMA noise).
#   2. Using conservative GPU memory utilization (default 0.3 instead of 0.9).
#   3. Capping max-model-len to reduce KV cache size.
#   4. Optionally bypassing auto-profiling entirely via --kv-cache-memory-bytes.
#
# Usage:
#   bash serve-vllm-dgx.sh [MODEL] [HOST] [PORT] [GPU_MEM_UTIL] [MAX_MODEL_LEN] [KV_CACHE_BYTES]
#
# Positional args (all optional):
#   MODEL           HuggingFace model ID        default: Qwen/Qwen2.5-7B-Instruct
#   HOST            bind address                default: 0.0.0.0
#   PORT            port                        default: 8000
#   GPU_MEM_UTIL    fraction of memory for vLLM default: 0.3  (ignored when KV_CACHE_BYTES is set)
#   MAX_MODEL_LEN   max context tokens          default: 8192
#   KV_CACHE_BYTES  explicit KV cache size      default: (empty — use GPU_MEM_UTIL instead)
#                   e.g. 8G, 10G, 16G
#
# Retry strategy if startup fails:
#   1. Conservative auto (default):
#        bash serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct
#   2. Explicit KV cache (bypasses profiling entirely):
#        bash serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct 0.0.0.0 8000 "" 8192 8G
#   3. If still flaky, close desktop apps or connect via SSH-only before running.
#
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"
GPU_MEM_UTIL="${4:-0.3}"
MAX_MODEL_LEN="${5:-8192}"
KV_CACHE_BYTES="${6:-}"

# ---------------------------------------------------------------------------
# Pre-flight: kill any leftover vLLM process and flush the OS page cache.
# The cache flush reduces UMA memory-measurement noise that causes the
# profiling assertion on DGX Spark.
# ---------------------------------------------------------------------------
echo "[serve-vllm-dgx] Stopping any running vLLM process..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 1

echo "[serve-vllm-dgx] Flushing OS page cache (reduces UMA noise)..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || \
  echo "[serve-vllm-dgx] WARNING: cache flush skipped (no sudo or not Linux)"

echo "[serve-vllm-dgx] Memory state before start:"
free -h | grep -E "^(Mem|Swap)"

# ---------------------------------------------------------------------------
# Build the vLLM argument list.
# If KV_CACHE_BYTES is set, use --kv-cache-memory-bytes (bypasses profiling).
# Otherwise fall back to --gpu-memory-utilization (conservative auto mode).
# ---------------------------------------------------------------------------
VLLM_ARGS=(
  --model          "${MODEL}"
  --enforce-eager
  --max-model-len  "${MAX_MODEL_LEN}"
  --host           "${HOST}"
  --port           "${PORT}"
)

if [[ -n "${KV_CACHE_BYTES}" ]]; then
  echo "[serve-vllm-dgx] KV cache mode: --kv-cache-memory-bytes ${KV_CACHE_BYTES}"
  VLLM_ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_BYTES}")
else
  echo "[serve-vllm-dgx] Auto KV cache mode: --gpu-memory-utilization ${GPU_MEM_UTIL}"
  VLLM_ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
fi

echo "[serve-vllm-dgx] Starting: model=${MODEL} host=${HOST} port=${PORT} max-model-len=${MAX_MODEL_LEN}"
echo ""

cd ~/dgx-spark-vllm/vllm
source ~/dgx-spark-vllm/vllm_env/bin/activate

python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
