# DGX Spark

Nvidia DGX Spark compact workstation. Self-hosted LLM serving on
Grace Blackwell hardware. Two supported paths below:

- **Ollama** on port `11434` — easiest fit for quantized local models.
- **vLLM** on port `8000` — OpenAI-compatible serving, higher
  throughput, but tighter memory tuning on Spark's unified memory.

RLM Studio treats both as separate LLM Provider entries; pick the one
you actually started. The "Open in Settings" button from this guide
pre-fills the vLLM shape because that's the OpenAI-compatible path.

## 1. Sanity-check the machine

```bash
hostnamectl
cat /etc/os-release
uname -a
nvidia-smi
free -h
df -h
ip addr
```

`nvidia-smi` must show the GPU and a sensible driver version before
you proceed. Everything downstream assumes it does.

## 2. Install and start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

By default Ollama only listens on `127.0.0.1`. To reach it from a
laptop running RLM Studio, configure it to listen on all interfaces
and store models in a host-owned directory:

```bash
sudo mkdir -p /var/lib/ollama/models
sudo chown -R ollama:ollama /var/lib/ollama
sudo systemctl edit ollama.service
```

In the override editor, add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/var/lib/ollama/models"
```

Save, then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama
sudo ss -lntp | grep 11434
```

## 3. Pull a model

```bash
ollama pull llama3.2
ollama pull gpt-oss:20b
ollama pull qwen2.5:14b
ollama list
```

On Spark, quantized models (GGUF) fit comfortably up to the 14B–20B
class. See §7 for the Ollama vs vLLM sizing rule of thumb.

## 4. Install and start vLLM (alternative path)

Only needed if you want the OpenAI-compatible serving surface. Skip to
§5 if Ollama is enough for your workload.

System packages and build toolchain:

```bash
sudo apt-get update
sudo apt-get install -y \
  gcc-12 g++-12 build-essential cmake ninja-build git \
  python3.12-dev python3-dev python3-venv python3-full \
  libnuma-dev numactl
```

Venv and CUDA-enabled PyTorch:

```bash
mkdir -p ~/dgx-spark-vllm && cd ~/dgx-spark-vllm
python3 -m venv vllm_env
source vllm_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir \
  'torch==2.10.0+cu130' \
  'torchvision==0.25.0+cu130' \
  'torchaudio==2.10.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
```

Clone and build vLLM from source (wheel builds are flaky on Spark):

```bash
cd ~/dgx-spark-vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="12.1a"
export MAX_JOBS=8

pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

Start the OpenAI-compatible server (safe starting point — 7B instruct
model with conservative memory budget):

```bash
source ~/dgx-spark-vllm/vllm_env/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --enforce-eager \
  --gpu-memory-utilization 0.4 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

See §7 for why these flags matter on Spark's unified memory.

## 5. Add to RLM Studio

### Ollama path

| Field    | Value                            |
|----------|----------------------------------|
| Backend  | `ollama`                         |
| Base URL | `http://<dgx-spark-ip>:11434`    |
| Model    | e.g. `qwen2.5:14b`, `llama3.2`   |

### vLLM path

| Field    | Value                            |
|----------|----------------------------------|
| Backend  | `vllm`                           |
| Base URL | `http://<dgx-spark-ip>:8000/v1`  |
| Model    | Same id you passed to `--model`  |

No API key is required by default. RLM Studio's Settings does not
currently expose an API-key field for local backends (vLLM counts as
local), so if you need to secure your Spark server, run it behind a
trusted network boundary — VPN, SSH tunnel, or listen only on
`127.0.0.1` — rather than via vLLM's own `--api-key`.

## 6. Test connection

Click **Test Connection** in Settings. From a shell you can also
check directly:

```bash
curl http://<dgx-spark-ip>:11434/api/version   # Ollama
curl http://<dgx-spark-ip>:8000/v1/models       # vLLM
```

## 7. Common errors

### Ollama

- **`11434` already in use** — something else is bound. Inspect with
  `sudo ss -lntp | grep 11434` and `ps aux | grep ollama`. A
  previous Docker-based Ollama or the integrated Open WebUI image
  are common culprits.
- **Reachable locally but not from laptop** — you missed the
  `OLLAMA_HOST=0.0.0.0:11434` override. Repeat §2.
- **Wrong model cache path** — if Ollama writes to
  `/var/lib/docker/volumes/...` unexpectedly, an old container is
  still running. Stop it, then use `/var/lib/ollama/models` as §2
  sets up.

### vLLM — memory is the tricky part

Spark uses unified system memory. vLLM's CUDA-side memory checks
and Spark's system-memory dashboard do not always agree, so vLLM can
refuse to start even when the dashboard looks fine.

**Default gotcha.** Without `--gpu-memory-utilization`, vLLM reserves
about 0.9 of what it sees. That is usually too aggressive on Spark.
Start at 0.4 for 7B-class models, 0.7 for 32B-class.

**Three failure patterns you will see:**

1. **Startup reservation check fails.** Free memory at startup is
   below the 0.9 default, or memory accounting shifts during
   startup profiling. Usually not a true OOM — lower
   `--gpu-memory-utilization` or pin the KV cache (below).
2. **Model loads, then no KV cache room left.** Seen with
   `Qwen/Qwen3-32B` at 0.4. Weights fit, cache does not. Raise
   utilization *and* reduce `--max-model-len`.
3. **Real OOM during weight load.** Seen with
   `Qwen/Qwen2.5-72B-Instruct`. In single-GPU BF16 without
   quantization, 70B-class is too large for Spark. Drop to a
   smaller model, or use Ollama with a quantized variant.

**Practical ranking for this Spark setup:**

- **Fits comfortably:** 7B–14B instruct models.
- **Possible with tuning:** some 30B–32B models (raise utilization,
  lower context, optionally pin KV cache).
- **Not recommended here:** 70B–72B BF16 without quantization.

**When auto-profiling is unstable, pin the KV cache explicitly:**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --enforce-eager \
  --gpu-memory-utilization 0.7 \
  --kv-cache-memory-bytes 4G \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

Try larger KV values (`6G`, `8G`) once `4G` starts cleanly.

**Reducing `--max-model-len` trades off context budget.** The
constraint `input_tokens + output_tokens <= max_model_len` is a hard
ceiling. If you drop to 4096 to fit memory, keep request
`max_tokens` small (256–512 is usually right for RLM loops, history
replays, and retrieval-augmented prompts).

**Between restarts, kill stale servers:**

```bash
pkill -f vllm.entrypoints.openai.api_server || true
nvidia-smi
free -h
```

### vLLM — build and request shape

- **`externally-managed-environment`** — you skipped the venv. Go
  back to §4.
- **`numa.h: No such file or directory`** — `sudo apt-get install
  libnuma-dev numactl`.
- **`NVFP4 / .e2m1x2 fails for sm_121`** — rebuild with
  `TORCH_CUDA_ARCH_LIST="12.1a"` in the environment.
- **Imports fail from some directories only** — run from the vLLM
  repo root.
- **Base model with chat endpoint returns garbage.** Base models like
  `facebook/opt-125m` belong on `/v1/completions`. Instruct/chat
  models like `Qwen/Qwen2.5-7B-Instruct` belong on
  `/v1/chat/completions`. Mixing the two is a common source of
  empty or nonsense responses.

## 8. External references

- NVIDIA DGX Spark portal — <https://build.nvidia.com/spark>
- DGX Spark User Guide (PDF) — <https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf>
- DGX Spark first boot — <https://docs.nvidia.com/dgx/dgx-spark/first-boot.html>
- NVIDIA Spark playbooks — <https://github.com/NVIDIA/dgx-spark-playbooks>
- NVIDIA vLLM on Spark — <https://build.nvidia.com/spark/vllm/instructions>
- NVIDIA vLLM Spark troubleshooting — <https://build.nvidia.com/spark/vllm/troubleshooting>
- Ollama FAQ — <https://docs.ollama.com/faq>
- vLLM GPU install — <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/>
- vLLM engine args — <https://docs.vllm.ai/en/stable/configuration/engine_args/>
