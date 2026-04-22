# vLLM

High-throughput inference server. Linux only — upstream does not
support Windows or macOS for production deployments.

- **Best for:** high-throughput GPU serving with an
  OpenAI-compatible API; larger unquantized models.
- **You'll need:** Linux, Python 3.9+, a CUDA-capable GPU, and
  enough VRAM for your chosen model (Spark tuning details below).
- **Known-good config:** base URL `http://localhost:8000/v1`;
  model id matches whatever you pass to `--model`.
- **Most common failure:** `--gpu-memory-utilization` default is
  too aggressive and startup aborts. See §6.

## 1. Install

Requires Python 3.9+ and a CUDA-capable GPU.

```bash
pip install vllm
```

## 2. Start the OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000
```

Server listens on `http://localhost:8000/v1` with an OpenAI-compatible
API surface.

## 3. Select a model

vLLM loads a single model per server process. The model id passed on
`--model` is what you configure in RLM Studio; it can be a Hugging Face
repo (`meta-llama/Llama-3.1-8B-Instruct`) or a local path.

## 4. Add to RLM Studio

See [hosts/README.md §3](README.md#3-what-rlm-studio-needs) for the general field shape. vLLM-specific values:

| Field    | Value                                 |
|----------|---------------------------------------|
| Backend  | `vllm`                                |
| Model    | Same id passed to `--model`           |
| Base URL | `http://localhost:8000/v1`            |

No API key is required. RLM Studio's Settings intentionally doesn't expose an API-key field for local backends — prefer a trusted network boundary over `vllm --api-key`. See [hosts/README.md §5](README.md#5-security--network-boundaries) for details.

## 5. Test connection

Click **Test Connection** in Settings.

## 6. Common errors

- **CUDA out of memory** — reduce `--max-model-len`, use `--dtype float16`,
  or pick a smaller model.
- **Model not found** — make sure the Hugging Face repo id is correct
  and accessible (gated models need a HF token exported as
  `HUGGING_FACE_HUB_TOKEN`).
- **Slow first request** — vLLM compiles CUDA kernels lazily; the first
  request may take tens of seconds.
