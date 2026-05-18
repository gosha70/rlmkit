# DGX Spark Setup Scripts

Shell scripts for setting up the NVIDIA DGX Spark as a self-hosted inference backend for RLMKit.

## Two paths

| Path | Directory | API | RLMKit usage |
|------|-----------|-----|--------------|
| **Ollama** | `ollama/` | `http://<spark-ip>:11434` | `provider="ollama", model="<model>", api_base=...` |
| **vLLM** | `vllm/` | `http://<spark-ip>:8000/v1` (OpenAI-compatible) | `provider="litellm", model="hosted_vllm/<model>", api_base=...` (LiteLLM ≥ 1.50 routes the generic `openai/` prefix through `/v1/responses`, which vLLM rejects on multi-turn — see `docs/hosts/dgx-spark-vllm.md` §7 Blocker #4) |

## Quick start

### Ollama path (simpler)

```bash
# On the DGX Spark machine:
bash ollama/setup-ollama-dgx.sh
ollama pull gpt-oss:20b

# From your dev machine:
curl http://<spark-ip>:11434/api/tags
```

```python
from rlmkit import interact
r = interact(content, query, mode="rlm",
             provider="ollama", model="gpt-oss:20b",
             api_base="http://<spark-ip>:11434")
```

### vLLM path (OpenAI-compatible, better for large models)

```bash
# On the DGX Spark machine:
bash vllm/setup-vllm-dgx.sh
bash vllm/serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct

# Verify from your dev machine:
bash vllm/verify-vllm-api-dgx.sh <spark-ip> 8000 Qwen/Qwen2.5-7B-Instruct
```

```python
from rlmkit import interact
# hosted_vllm/ (not openai/) — LiteLLM ≥ 1.50 routes openai/<...> through
# vLLM's /v1/responses, which rejects multi-turn input arrays. The
# hosted_vllm/ provider pins to /v1/chat/completions. See
# docs/hosts/dgx-spark-vllm.md §7 Blocker #4.
r = interact(content, query, mode="rlm",
             provider="litellm", model="hosted_vllm/Qwen2.5-7B-Instruct",
             api_base="http://<spark-ip>:8000/v1")
```

## When to use which

- **Ollama**: easier setup, good for models already in `ollama pull` format, direct RLMKit `provider="ollama"` support.
- **vLLM**: OpenAI-compatible API, better throughput for large models, required for HuggingFace models not packaged for Ollama.

## ⚠ Run only one backend at a time

DGX Spark has **128 GB unified memory** shared between the ARM CPU and the Blackwell GPU.
Ollama and vLLM each claim a large slice of this pool when a model is loaded.

**Running both simultaneously causes:**
- The second backend to spill model layers onto the CPU (→ 10–50× slower inference)
- Requests to hang at the HTTP layer because the model never gets enough GPU compute
- Apparent timeouts that are actually just extremely slow CPU-side generation

**Before starting Ollama, stop vLLM:**
```bash
# Kill the vLLM server process
pkill -f "vllm.entrypoints.openai.api_server" || true
# Confirm memory is free
free -h          # expect >100 GB available
ollama ps        # confirm no models loaded
```

**Before starting vLLM, stop Ollama models:**
```bash
ollama stop <model-name>     # or: sudo systemctl stop ollama
free -h
```

**Quick memory check** (DGX Dashboard at `http://localhost:11000/`):
- System Memory gauge should read < 20 GB used before loading any model
- GPU Utilization should jump to > 80% once inference starts — if it stays near 0%, memory is likely exhausted and layers have fallen back to CPU

See each subdirectory's README for full script details.
