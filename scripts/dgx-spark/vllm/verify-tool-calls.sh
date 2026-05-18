#!/usr/bin/env bash
# verify-tool-calls.sh — 90-second smoke test for the DGX Spark + vLLM
# + Qwen3-Coder-Next-NVFP4 deployment.
#
# Companion to docs/hosts/dgx-spark-vllm.md §6. The three checks below
# are byte-for-byte the curls documented there:
#
#   1. /v1/models returns model id + max_model_len >= 65536
#   2. single-turn tool call extracts into structured tool_calls[]
#      (catches Blocker #2 — wrong / missing reasoning parser)
#   3. two-turn multi-turn completes via /v1/chat/completions
#      (catches Blocker #4 if a LiteLLM proxy is between client and
#       vLLM; raw curl bypasses that route)
#
# Usage:
#
#   bash verify-tool-calls.sh                              # against localhost:8000
#   VLLM_BASE=http://<dgx-spark-ip>:8000 bash verify-tool-calls.sh
#   VLLM_MODEL=RedHatAI/Qwen3-Coder-Next-NVFP4 bash verify-tool-calls.sh
#
# Exits 0 if all three pass, non-zero on any failure.

set -euo pipefail

VLLM_BASE="${VLLM_BASE:-http://localhost:8000}"
VLLM_MODEL="${VLLM_MODEL:-RedHatAI/Qwen3-Coder-Next-NVFP4}"

if ! command -v jq >/dev/null; then
  echo "error: jq is required for the smoke test (response parsing)" >&2
  exit 2
fi

pass() { printf '  PASS: %s\n' "$1"; }
fail() { printf '  FAIL: %s\n' "$1"; }

failures=0

# ---------------------------------------------------------------------
# Step 1 — model loaded, context large enough
# ---------------------------------------------------------------------
echo "Step 1: /v1/models ($VLLM_BASE)"
models_body=$(curl -s --max-time 10 "$VLLM_BASE/v1/models" || true)
if [[ -z "$models_body" ]]; then
  fail "no response from $VLLM_BASE/v1/models — is vLLM up?"
  failures=$((failures + 1))
else
  model_id=$(echo "$models_body" | jq -r '.data[0].id // empty')
  max_len=$(echo "$models_body" | jq -r '.data[0].max_model_len // 0')
  if [[ -z "$model_id" ]]; then
    fail "response had no .data[0].id — server returned: $(echo "$models_body" | head -c 200)"
    failures=$((failures + 1))
  elif (( max_len < 65536 )); then
    fail "max_model_len=$max_len < 65536 — will reject claude-code-class agent envelopes (Blocker #3 in docs/hosts/dgx-spark-vllm.md §7)"
    failures=$((failures + 1))
  else
    pass "model=$model_id, max_model_len=$max_len"
  fi
fi
echo

# ---------------------------------------------------------------------
# Step 2 — single-turn tool call is extracted into tool_calls[]
# ---------------------------------------------------------------------
echo "Step 2: single-turn tool call"
step2_body=$(curl -s --max-time 60 "$VLLM_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "model": "$VLLM_MODEL",
  "messages": [{"role":"user","content":"Use the provided tool to create foo.py with a function bar() that returns 42."}],
  "tools": [{
    "type":"function",
    "function": {
      "name":"edit_file",
      "description":"Edit or create a file.",
      "parameters": {
        "type":"object",
        "properties": {
          "path":{"type":"string"},
          "content":{"type":"string"}
        },
        "required":["path","content"]
      }
    }
  }],
  "tool_choice":"auto",
  "temperature": 0,
  "max_tokens": 300
}
JSON
)" || true)

tool_call_name=$(echo "$step2_body" | jq -r '.choices[0].message.tool_calls[0].function.name // empty' 2>/dev/null || echo "")
if [[ "$tool_call_name" == "edit_file" ]]; then
  pass "tool_calls[0].function.name = edit_file"
else
  fail "tool_calls[] missing or wrong shape — Blocker #2 (see docs/hosts/dgx-spark-vllm.md §7)"
  echo "  response (truncated):"
  echo "$step2_body" | jq '.choices[0].message | {content, tool_calls, reasoning}' 2>/dev/null | head -20 | sed 's/^/    /'
  failures=$((failures + 1))
fi
echo

# ---------------------------------------------------------------------
# Step 3 — two-turn continuation
# ---------------------------------------------------------------------
echo "Step 3: two-turn continuation"
turn1_body=$(curl -s --max-time 30 "$VLLM_BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "model": "$VLLM_MODEL",
  "messages": [{"role":"user","content":"What is 2+2? Reply with just the number."}],
  "max_tokens": 10,
  "temperature": 0
}
JSON
)" || true)

turn1_content=$(echo "$turn1_body" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
if [[ -z "$turn1_content" ]]; then
  fail "turn 1 returned no content — response: $(echo "$turn1_body" | head -c 200)"
  failures=$((failures + 1))
else
  # Escape the prior assistant content for safe inline embedding in the next JSON body.
  assistant_escaped=$(printf '%s' "$turn1_content" | jq -Rs .)

  turn2_body=$(curl -s --max-time 30 "$VLLM_BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<JSON
{
  "model": "$VLLM_MODEL",
  "messages": [
    {"role":"user","content":"What is 2+2?"},
    {"role":"assistant","content": $assistant_escaped},
    {"role":"user","content":"And 3+3?"}
  ],
  "max_tokens": 10,
  "temperature": 0
}
JSON
)" || true)

  turn2_content=$(echo "$turn2_body" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
  if [[ -z "$turn2_content" ]]; then
    fail "turn 2 returned no content — likely Blocker #4 (LiteLLM Responses-API misroute) if going through a proxy; raw curl should not hit this. Response: $(echo "$turn2_body" | head -c 200)"
    failures=$((failures + 1))
  else
    pass "turn 1 = $(printf '%s' "$turn1_content" | head -c 40 | tr -d '\n'); turn 2 = $(printf '%s' "$turn2_content" | head -c 40 | tr -d '\n')"
  fi
fi
echo

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
if (( failures == 0 )); then
  echo "All 3 smoke-test checks passed."
  exit 0
else
  echo "$failures of 3 smoke-test checks failed — see docs/hosts/dgx-spark-vllm.md §7."
  exit 1
fi
