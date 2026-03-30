#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"

cd ~/dgx-spark-vllm/vllm
source ~/dgx-spark-vllm/vllm_env/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --enforce-eager \
  --host "${HOST}" \
  --port "${PORT}"
