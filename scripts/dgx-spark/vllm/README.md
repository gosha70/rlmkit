# DGX Spark vLLM Scripts Bundle

This bundle contains shell scripts for setting up and using **vLLM on NVIDIA DGX Spark**.

## Files

### `setup-vllm-dgx.sh`
Installs the system dependencies, creates a Python virtual environment, installs CUDA-enabled PyTorch, clones vLLM, and builds it from source.

This script follows the working path from the chat, including:
- CUDA 13 environment
- GCC 12
- `TORCH_CUDA_ARCH_LIST="12.1a"`
- editable vLLM install

Run:

```bash
bash setup-vllm-dgx.sh
```

Use it when you want:
- a local source-built vLLM setup
- OpenAI-compatible serving on DGX Spark
- a reusable dev environment for vLLM

### `test-vllm-dgx.sh`
Creates and runs a small smoke test using `facebook/opt-125m` in eager mode.

It verifies:
- CUDA is visible
- vLLM imports correctly
- model loading works
- a real generation call succeeds

Run:

```bash
bash test-vllm-dgx.sh
```

### `serve-vllm-dgx.sh`
Starts the OpenAI-compatible vLLM server with DGX Spark memory workarounds.

DGX Spark uses Unified Memory Architecture (UMA). vLLM's default memory-profiling
step can trigger an assertion during startup when free memory fluctuates. This script
works around that by flushing the OS page cache before launch and using conservative
memory defaults.

Usage:

```bash
bash serve-vllm-dgx.sh [MODEL] [HOST] [PORT] [GPU_MEM_UTIL] [MAX_MODEL_LEN] [KV_CACHE_BYTES]
```

Defaults:
- model: `Qwen/Qwen2.5-7B-Instruct`
- host: `0.0.0.0`
- port: `8000`
- gpu_mem_util: `0.3` (conservative; ignored when `KV_CACHE_BYTES` is set)
- max_model_len: `8192`
- kv_cache_bytes: *(empty — use `GPU_MEM_UTIL` auto mode)*

Examples:

```bash
# Default (conservative auto mode — try this first)
bash serve-vllm-dgx.sh

# Explicit model
bash serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct

# Explicit KV cache — bypasses profiling assertion entirely
bash serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct 0.0.0.0 8000 "" 8192 8G

# Larger KV cache once 8G is confirmed working
bash serve-vllm-dgx.sh Qwen/Qwen2.5-7B-Instruct 0.0.0.0 8000 "" 8192 16G
```

**Retry order if startup fails with a memory-profiling assertion:**
1. Conservative auto (default): `bash serve-vllm-dgx.sh`
2. Explicit KV cache (bypasses profiling): `bash serve-vllm-dgx.sh <model> 0.0.0.0 8000 "" 8192 8G`
3. If still flaky, close desktop apps or connect via SSH only before running.

### `verify-vllm-api-dgx.sh`
Verifies the running vLLM server by calling:
- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`

Usage:

```bash
bash verify-vllm-api-dgx.sh [HOST] [PORT] [CHAT_MODEL] [BASE_MODEL]
```

Default example:

```bash
bash verify-vllm-api-dgx.sh 192.168.1.23 8000 Qwen/Qwen2.5-7B-Instruct facebook/opt-125m
```

### `clean-vllm-build-dgx.sh`
Cleans repo-local build artifacts and purges the pip cache.

Use it when you want:
- to retry a vLLM source build
- to remove stale CMake/Ninja artifacts
- to force a fresh build attempt

Run:

```bash
bash clean-vllm-build-dgx.sh
```

## Recommended usage order

1. Run `setup-vllm-dgx.sh`
2. Run `test-vllm-dgx.sh`
3. Start server with `serve-vllm-dgx.sh`
4. Verify with `verify-vllm-api-dgx.sh`

## Important usage rule

### Base model -> completions endpoint

Use base models like `facebook/opt-125m` with:

```text
/v1/completions
```

### Instruct/chat model -> chat completions endpoint

Use instruct/chat models like `Qwen/Qwen2.5-7B-Instruct` with:

```text
/v1/chat/completions
```

## Typical verification commands

```bash
curl http://127.0.0.1:8000/v1/models
curl http://<dgx-spark-ip>:8000/v1/models
```

## Recommendation

Use vLLM when you want:
- an OpenAI-compatible API
- server-style inference
- instruct/chat model serving

Use **host Ollama** instead when you want:
- simpler RLM Studio integration
- direct Ollama-based local serving
