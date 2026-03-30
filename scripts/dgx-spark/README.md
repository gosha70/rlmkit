# DGX Spark Setup Scripts

Shell scripts for setting up the NVIDIA DGX Spark as a self-hosted inference backend for RLMKit.

## Two paths

| Path | Directory | API | RLMKit usage |
|------|-----------|-----|--------------|
| **Ollama** | `ollama/` | `http://<spark-ip>:11434` | `provider="ollama", model="<model>", api_base=...` |
| **vLLM** | `vllm/` | `http://<spark-ip>:8000/v1` (OpenAI-compatible) | `provider="litellm", model="openai/<model>", api_base=...` |

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
r = interact(content, query, mode="rlm",
             provider="litellm", model="openai/Qwen2.5-7B-Instruct",
             api_base="http://<spark-ip>:8000/v1")
```

## When to use which

- **Ollama**: easier setup, good for models already in `ollama pull` format, direct RLMKit `provider="ollama"` support.
- **vLLM**: OpenAI-compatible API, better throughput for large models, required for HuggingFace models not packaged for Ollama.

See each subdirectory's README for full script details.
