# DGX Spark + vLLM: Qwen3-Coder-Next-NVFP4 operator manual

> **Verified 2026-05-17 on DGX Spark with vLLM 0.6+ and flashinfer 0.6.6, sm_121a target.** Model: `RedHatAI/Qwen3-Coder-Next-NVFP4`. End-to-end against the `aider-polyglot` benchmark, full multi-turn agentic tool loops. Every flag in §3 is load-bearing — removing or changing any of them re-introduces one of the four blockers in §7.

This doc is the Qwen3-Coder-Next-NVFP4 companion to [`dgx-spark.md`](dgx-spark.md) (general DGX Spark setup) and [`vllm.md`](vllm.md) (general vLLM). Read those first if you have not already.

## 1. When to use this doc

You want a coding-grade model self-hosted on a DGX Spark, accessible via an OpenAI-compatible API with tool calling enabled, callable from RLMKit or any agent client (claude-code, Cline, raw OpenAI SDK). The verified setup runs `RedHatAI/Qwen3-Coder-Next-NVFP4` and matches Sonnet-class behaviour on multi-turn agentic tool loops (see `aider-polyglot` notes at the end).

If you want a simpler local setup (smaller models, no tool calling, Ollama instead of vLLM), use [`dgx-spark.md`](dgx-spark.md) §3. If you want generic vLLM with no Spark-specific guidance, use [`vllm.md`](vllm.md).

## 2. Hardware assumptions

- DGX Spark (GB10 Grace-Blackwell, sm_121a), 128 GB unified memory.
- ARM64 Linux.
- NVFP4 quantization requires Blackwell **sm_120+** (Spark is sm_121a, supported).
- vLLM and flashinfer built from source for sm_121a — see [`dgx-spark.md`](dgx-spark.md) §4 for the build procedure. The wheel paths there are the assumed starting state of this doc.

## 3. Working configuration (verified)

Paste this verbatim. Every flag is load-bearing — see §7 for what breaks if you remove one.

```bash
pkill -f "vllm.entrypoints.openai.api_server" || true
pkill -f "EngineCore" || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null

source ~/dgx-spark-vllm/vllm_env/bin/activate
export VLLM_USE_FLASHINFER_MOE_FP4=0
export MAX_JOBS=2
export NINJA_JOBS=2

python -m vllm.entrypoints.openai.api_server \
  --model ~/dgx-spark-vllm/models/Qwen3-Coder-Next-NVFP4 \
  --served-model-name RedHatAI/Qwen3-Coder-Next-NVFP4 \
  --enforce-eager \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.72 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

Flag-by-flag rationale:

- **`pkill -f ...`** — kill any stale vLLM workers so the new boot isn't fighting for ports or KV cache.
- **`drop_caches`** — flush the OS page cache; reduces UMA accounting noise during vLLM's startup memory probe.
- **`VLLM_USE_FLASHINFER_MOE_FP4=0`** — disable flashinfer's experimental FP4 MoE path. The verified setup uses the standard FP4 kernel.
- **`MAX_JOBS=2`, `NINJA_JOBS=2`** — cap concurrent nvcc/cicc/cc1plus processes during flashinfer JIT (Blocker #1).
- **`--enforce-eager`** — skip CUDA graph capture. Trades throughput for startup stability on Spark.
- **`--max-model-len 131072`** — ceiling on `prompt_tokens + max_tokens`. Sized for claude-code-class clients (Blocker #3); see §5 to tune.
- **`--max-num-seqs 1`, `--max-num-batched-tokens 8192`** — single-stream serving with a modest batch ceiling. Keeps KV cache pressure predictable.
- **`--gpu-memory-utilization 0.72`** — empirically the headroom that fits 44 GB weights + 128K KV cache on 128 GB unified memory.
- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder`** — Qwen3-Coder-Next emits `<tool_call>` XML blocks that vLLM's `qwen3_coder` parser extracts into structured `tool_calls[]`. **Do not** also set `--reasoning-parser qwen3` — see Blocker #2.
- **`--trust-remote-code`** — Qwen3-Coder uses custom `modeling_*.py`; required by Hugging Face's loader.
- **`--host 0.0.0.0 --port 8000`** — bind on the LAN. Pair with a VPN, SSH tunnel, or trusted-LAN topology per [`hosts/README.md` §5](README.md#5-security--network-boundaries).

A copy-paste runnable wrapper of this command lives at [`scripts/dgx-spark/vllm/start-qwen3-coder-next.sh`](../../scripts/dgx-spark/vllm/start-qwen3-coder-next.sh) (overridable via `VLLM_MODEL_PATH`, `VLLM_SERVED_MODEL_NAME`, `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_UTIL`, `VLLM_PORT`, `VLLM_VENV`, `VLLM_TOOL_CALL_PARSER`, `VLLM_MAX_NUM_BATCHED_TOKENS`, and `VLLM_REASONING_PARSER` — the last defaults to unset, because the verified config above passes no `--reasoning-parser`; see Blocker #2).

## 4. First-boot expectations

The first boot at a given `(model, max-model-len, sm-target)` tuple is **slow**. flashinfer JIT-compiles ~90 kernels for sm_121a; on Spark this takes 10–15 minutes. The boot will look stuck on `Loading model weights` for several minutes, then on a sequence of `Compiling kernel ...` lines.

Subsequent boots at the same `(model, max-model-len)` are near-instant — the compiled kernels are cached at `~/.cache/flashinfer/<ver>/121a/`. Bumping `--max-model-len` invalidates the relevant tile-size cache and re-triggers compilation for the new value; budget 5–15 minutes on the first boot at each new `max-model-len`.

If a first boot is killed by an OOM mid-JIT (Blocker #1), the partial cache under `~/.cache/flashinfer/<ver>/121a/cached_ops/fused_moe_120/` survives. The next attempt resumes from where the previous one crashed — do not wipe the cache to "start fresh."

## 5. Memory tuning table

Sourced from verified runs on a 128 GB DGX Spark with NVFP4 Qwen3-Coder-Next (~44 GB weights):

| `max-model-len` | `gpu-memory-utilization` | Fits | Notes |
|---|---|---|---|
| 8192   | 0.50 | ✓ | Smoke-test only; rejects any agent client with ≥ 1K prompt + 32K output |
| 16384  | 0.55 | ✓ | Marginal for thin clients; rejects claude-code-class envelopes |
| 32768  | 0.55 | ✓ | Floor for raw OpenAI SDK clients; insufficient for claude-code |
| 65536  | 0.65 | ✓ | Tight for claude-code (~38K envelope + 32K output = 70K > 65K — will reject) |
| 131072 | 0.72 | ✓ | **Verified** working ceiling for claude-code clients with 2× headroom |
| 262144 | —    | not tested | Qwen3-Coder-Next supports 256K natively but Spark memory caps it |

Bumping `--max-model-len` triggers flashinfer recompilation for new sequence-length tile sizes. Budget 5–15 minutes on first boot at each new value; subsequent boots at the same value are instant.

## 6. Verification smoke test (90 seconds)

Run these three curls in order. They confirm the model is loaded, the tool-call parser is wired correctly, and multi-turn requests don't trip a route bug. The same checks are scripted at [`scripts/dgx-spark/vllm/verify-tool-calls.sh`](../../scripts/dgx-spark/vllm/verify-tool-calls.sh).

### Step 1: model is loaded and context is large enough

```bash
curl -s http://localhost:8000/v1/models | jq '{
  model: .data[0].id,
  max_model_len: .data[0].max_model_len
}'
```

- **PASS:** `max_model_len >= 65536` (and ideally ≥ 131072 for claude-code clients).
- **FAIL:** 404 (server not up), or `max_model_len < 33000` (will reject any request with default agent envelopes).

### Step 2: single-turn tool call is extracted into structured `tool_calls[]`

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"RedHatAI/Qwen3-Coder-Next-NVFP4",
    "messages":[{"role":"user","content":"Use the provided tool to create foo.py with a function bar() that returns 42."}],
    "tools":[{"type":"function","function":{"name":"edit_file","description":"Edit or create a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
    "tool_choice":"auto",
    "temperature":0,
    "max_tokens":300
  }' | jq '.choices[0].message | {content, tool_calls, reasoning}'
```

- **PASS:** `tool_calls` is a non-empty array containing a `function.name == "edit_file"` entry; `content` is null or a short preamble; `reasoning` is null.
- **FAIL:** `tool_calls: []` with the XML buried in `content` or `reasoning`. This means the tool-call parser is missing (Blocker #2) or the reasoning parser is wrongly enabled and intercepting the call.

### Step 3: two-turn continuation completes without route error

```bash
# Turn 1
RESP1=$(curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RedHatAI/Qwen3-Coder-Next-NVFP4","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":10,"temperature":0}')
echo "$RESP1" | jq '.choices[0].message.content'

# Turn 2 — continuation with prior assistant message in history
ASSISTANT=$(echo "$RESP1" | jq -r '.choices[0].message.content')
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"RedHatAI/Qwen3-Coder-Next-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"},{\"role\":\"assistant\",\"content\":\"$ASSISTANT\"},{\"role\":\"user\",\"content\":\"And 3+3?\"}],\"max_tokens\":10,\"temperature\":0}" \
  | jq '.choices[0].message.content'
```

- **PASS:** both turns return HTTP 200 with non-null `content`. Tail the vLLM access log; both requests show `POST /v1/chat/completions HTTP/1.1 200`.
- **FAIL:** turn 2 returns 400 with `212 validation errors` — Blocker #4. (This only manifests when the client goes through a LiteLLM proxy routed via `/v1/responses`; raw curl against vLLM directly will not hit it, so this step's role is mostly to confirm `/v1/chat/completions` works for multi-turn.)

## 7. Troubleshooting matrix

Every blocker below was hit during the 2026-05-17 debug session. The symptom strings are verbatim — operators grep for exact error text.

| # | Symptom (verbatim) | Cause | Fix | Detection command |
|---|---|---|---|---|
| 1 | `ninja: build stopped: subcommand failed` followed by `Engine core initialization failed`, exit code 137 on one of the nvcc/cicc/cc1plus subprocesses during first boot | flashinfer JIT compile for sm_121a runs ~8 concurrent nvcc processes (~1–2 GB each); combined with the 44 GB model weights, this OOM-kills one of them and aborts the whole build | Set `MAX_JOBS=2 NINJA_JOBS=2` in the environment before starting vLLM. Drop to `MAX_JOBS=1` if 2 still OOMs. The partial flashinfer cache at `~/.cache/flashinfer/<ver>/121a/cached_ops/fused_moe_120/` is preserved across runs, so the next attempt resumes from where the previous one crashed | `watch -n 2 'ps -o pid,pcpu,pmem,cmd -ax \| grep -E "nvcc\|cicc\|cc1plus" \| grep -v grep \| wc -l'` during boot — should stay ≤ 2 |
| 2 | vLLM startup fails with `Reasoning parser 'qwen3_coder' not found. Available parsers: ... qwen3 ...`, OR the model responds with `tool_calls: []` and the tool-call XML buried inside `reasoning` field on a tool-use request | The `qwen3_coder` reasoning parser does not exist in this vLLM build. The `qwen3` reasoning parser does exist, but it consumes the `<tool_call>` XML blocks before the tool-call parser sees them, producing empty `tool_calls[]` | Omit `--reasoning-parser` entirely for Qwen3-Coder-Next. The model does not have a separate thinking-mode output to extract; everything it emits is either content or a `<tool_call>` block | Run the second verification curl in §6 and inspect `.choices[0].message`. PASS = `tool_calls: [{...}]` populated. FAIL = `tool_calls: []` with XML elsewhere in the response |
| 3 | All requests return HTTP 400 in <1 second with `This model's maximum context length is N tokens. However, you requested M output tokens and your prompt contains K characters` where M + estimated_prompt_tokens > N | vLLM enforces `prompt_tokens + max_tokens <= max_model_len` strictly. Agent clients like claude-code ship a large request envelope (system prompt + tool schemas + skills metadata, often ~30–40K tokens) on every call. With default `max_tokens=32000`, total context floor is ~70K | Raise `--max-model-len` to at least `measured_envelope + max_output_tokens + safety_margin`. For claude-code as the client, this is ~131072. For thinner clients (raw OpenAI SDK, single-shot completions), 32768 may suffice. The verified config uses 131072 | `curl -s http://<host>:8000/v1/models \| jq '.data[0].max_model_len'` — compare against expected agent client's worst-case `(prompt + max_tokens)` |
| 4 | A multi-turn agentic conversation crashes mid-loop with HTTP 400 + `OpenAIException` + `212 validation errors: {'type': 'string_type', 'loc': ('body', 'input', 'str'), 'msg': 'Input should be a valid string'}` and the offending payload contains `{'type': 'input_text', 'text': ...}` and `{'type': 'message', 'role': ..., 'content': [...]}` blocks | LiteLLM ≥ 1.50 auto-detects vLLM's `/v1/responses` (OpenAI Responses API) endpoint and routes the generic `openai/` provider through it. vLLM's Responses API implementation rejects LiteLLM's multi-turn input-array shape. The model itself is fine; the route is wrong | In the LiteLLM proxy config, change the `model` field from `openai/<name>` to `hosted_vllm/<name>`. The `hosted_vllm/` provider is LiteLLM's purpose-built vLLM adapter and pins to `/v1/chat/completions` deterministically. Fallback: pin `litellm<1.50` | After the fix, tail the vLLM access log during a multi-turn request: every line must read `POST /v1/chat/completions HTTP/1.1 200`. Any `POST /v1/responses` line means the route fix didn't take effect |

## 8. Using this server from RLMKit (via LiteLLM)

The model string passed into LiteLLM determines which vLLM endpoint LiteLLM hits. With LiteLLM ≥ 1.50, the `hosted_vllm/` provider prefix pins to `/v1/chat/completions`; the generic `openai/` prefix is auto-detected as the OpenAI Responses API and routed via `/v1/responses`, which vLLM rejects on multi-turn (Blocker #4).

### Recommended pattern: `provider="litellm"` + explicit `hosted_vllm/` model string

Use the LiteLLM passthrough path and pass the fully-qualified model string yourself:

```python
from rlmkit import interact

r = interact(
    content, query, mode="rlm",
    provider="litellm",                                          # passthrough — no provider-prefix rewriting
    model="hosted_vllm/RedHatAI/Qwen3-Coder-Next-NVFP4",         # pins to /v1/chat/completions
    api_base="http://<dgx-spark-ip>:8000/v1",
)
```

`provider="litellm"` is not in RLMKit's per-provider prefix table, so the model string passes through verbatim and LiteLLM sees the `hosted_vllm/` prefix it needs.

### Wrong pattern — recreates Blocker #4

```python
# Wrong — RLMKit's built-in vllm provider prepends "openai/" to the
# model string (see "Known limitation" below), and LiteLLM >= 1.50
# then auto-routes openai/* through /v1/responses on a vLLM upstream.
r = interact(
    content, query,
    provider="vllm", model="RedHatAI/Qwen3-Coder-Next-NVFP4",
    api_base="http://<dgx-spark-ip>:8000/v1",
)
```

### Known limitation: RLMKit's built-in `vllm` provider still hardcodes `openai/`

RLMKit's built-in `vllm` backend — both `provider="vllm"` in `interact()` and the **vllm** entry in RLM Studio's LLM Provider form — currently hardcodes the LiteLLM prefix to `openai/` at three call sites:

- `src/rlmkit/api.py` — `_PROVIDER_PREFIXES["vllm"] = "openai/"`
- `src/rlmkit/server/routes/providers.py` — per-backend prefix table used by the Studio Provider form
- `src/rlmkit/server/dependencies.py` — the equivalent table consumed by the request dispatch path

An operator who points the built-in `vllm` provider at this Spark setup will therefore still hit Blocker #4 once LiteLLM is ≥ 1.50, even though every flag in §3 is correct.

**Workarounds today:**
- **Python API:** use the `provider="litellm"` pattern above.
- **RLM Studio:** there is no in-UI workaround; the form's vllm backend goes through the affected code path. Use the Python API for Spark-hosted vLLM until the fix lands, or pin `litellm<1.50`.

**Fix (out of scope for this docs change):** flip the three table values from `"openai/"` to `"hosted_vllm/"` and add a regression test that asserts a vLLM multi-turn request lands on `/v1/chat/completions`. Tracked as a follow-up.

## References

- vLLM tool-calling docs — <https://docs.vllm.ai/en/stable/features/tool_calling/>
- vLLM Claude Code integration — <https://docs.vllm.ai/en/stable/serving/integrations/claude_code/>
- LiteLLM `hosted_vllm` provider — <https://docs.litellm.ai/docs/providers/vllm>
- Anchor benchmark (separate repo `code-copilot-team`): aider-polyglot `python/bowling`, n=3 each — Sonnet 3/3 pass @ 141 ± 39 s; Qwen3-Coder-Next-NVFP4 (vLLM) 3/3 pass @ 473 ± 449 s. Both backends completed full multi-turn agentic tool loops with the configuration above.
