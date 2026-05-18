#!/usr/bin/env bash
# start-qwen3-coder-next.sh — Start vLLM serving Qwen3-Coder-Next-NVFP4 on DGX Spark.
#
# Companion to docs/hosts/dgx-spark-vllm.md (the operator manual).
# Every flag below is load-bearing — see §3 and §7 of that doc for
# what breaks if you change one. Verified 2026-05-17 on DGX Spark
# (GB10 Grace-Blackwell, sm_121a, 128 GB unified memory).
#
# Usage:
#
#   bash start-qwen3-coder-next.sh
#
# Overridable via env vars (defaults match the verified config):
#
#   VLLM_VENV               default: ~/dgx-spark-vllm/vllm_env
#   VLLM_MODEL_PATH         default: ~/dgx-spark-vllm/models/Qwen3-Coder-Next-NVFP4
#   VLLM_SERVED_MODEL_NAME  default: RedHatAI/Qwen3-Coder-Next-NVFP4
#   VLLM_MAX_MODEL_LEN      default: 131072
#   VLLM_GPU_UTIL           default: 0.72
#   VLLM_PORT               default: 8000
#
# This script does NOT install vLLM or build flashinfer — those steps
# live in docs/hosts/dgx-spark.md §4. This script assumes both are
# already in place at $VLLM_VENV.

set -euo pipefail

VLLM_VENV="${VLLM_VENV:-$HOME/dgx-spark-vllm/vllm_env}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$HOME/dgx-spark-vllm/models/Qwen3-Coder-Next-NVFP4}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-RedHatAI/Qwen3-Coder-Next-NVFP4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.72}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Sanity-check the venv exists before we touch the system.
if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
  echo "error: vLLM venv not found at $VLLM_VENV" >&2
  echo "       set VLLM_VENV to the right path, or follow" >&2
  echo "       docs/hosts/dgx-spark.md §4 to build vLLM from source." >&2
  exit 1
fi

echo "==> killing any stale vLLM / EngineCore workers"
pkill -f "vllm.entrypoints.openai.api_server" || true
pkill -f "EngineCore" || true

echo "==> flushing OS page cache (reduces UMA noise during vLLM startup probe)"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
  echo "    (skipped — sudo unavailable; not fatal, just less deterministic)"

echo "==> activating venv: $VLLM_VENV"
# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

# Disable flashinfer's experimental FP4 MoE path; the standard FP4
# kernel is what's verified.
export VLLM_USE_FLASHINFER_MOE_FP4=0

# Cap concurrent nvcc/cicc/cc1plus during flashinfer JIT (Blocker #1
# in docs/hosts/dgx-spark-vllm.md §7). Drop both to 1 if 2 still OOMs.
export MAX_JOBS=2
export NINJA_JOBS=2

echo "==> starting vLLM"
echo "    model:                $VLLM_MODEL_PATH"
echo "    served-model-name:    $VLLM_SERVED_MODEL_NAME"
echo "    max-model-len:        $VLLM_MAX_MODEL_LEN"
echo "    gpu-memory-util:      $VLLM_GPU_UTIL"
echo "    port:                 $VLLM_PORT"
echo

exec python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_PATH" \
  --served-model-name "$VLLM_SERVED_MODEL_NAME" \
  --enforce-eager \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host 0.0.0.0 --port "$VLLM_PORT"
