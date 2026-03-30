#!/usr/bin/env bash
set -euo pipefail

VLLM_HOST="${1:-127.0.0.1}"
VLLM_PORT="${2:-8000}"
CHAT_MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
# BASE_MODEL defaults to CHAT_MODEL so a single-model server always passes.
# Override with arg 4 only when explicitly serving a separate base model.
BASE_MODEL="${4:-${CHAT_MODEL}}"

echo "==> Models endpoint"
curl -f "http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
echo
echo
echo "==> Completions test (${BASE_MODEL})"
curl -f "http://${VLLM_HOST}:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${BASE_MODEL}\", \"prompt\": \"Say hello in one sentence.\", \"max_tokens\": 16}"
echo
echo
echo "==> Chat-completions test (${CHAT_MODEL})"
curl -f "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${CHAT_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}], \"max_tokens\": 32}"
echo
