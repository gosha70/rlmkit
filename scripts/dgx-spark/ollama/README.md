# DGX Spark Scripts Bundle

This bundle contains shell scripts for the **recommended Ollama + Open WebUI setup** on NVIDIA DGX Spark.

## Files

### `setup-ollama-dgx.sh`
Installs **host Ollama** on DGX Spark, configures it to listen on `0.0.0.0:11434`, and uses `/var/lib/ollama/models` as the model store.

Use it when you want:
- host Ollama as the main model backend
- RLMKit or document-assistant to connect directly to DGX Spark
- one clean Ollama service instead of multiple embedded stores

Run:

```bash
bash setup-ollama-dgx.sh
```

Then pull models:

```bash
ollama pull llama3.2
ollama pull gpt-oss:20b
ollama pull qwen2.5:14b
ollama list
```

### `setup-openwebui-dgx.sh`
Runs **Open WebUI** in the recommended mode, pointing it to **host Ollama** using `OLLAMA_BASE_URL`.

Use it when you want:
- a browser UI on top of host Ollama
- optional human-facing chat UI
- to avoid a second embedded Ollama store

Run:

```bash
bash setup-openwebui-dgx.sh <dgx-spark-ip>
```

Example:

```bash
bash setup-openwebui-dgx.sh 192.168.1.23
```

Then open:

```text
http://<dgx-spark-ip>:8080
```

### `setup-openwebui-integrated-dgx.sh`
Runs the **integrated Open WebUI + embedded Ollama** container.

Use it when you want:
- a self-contained demo
- quick testing without separately managing host Ollama integration

Run:

```bash
bash setup-openwebui-integrated-dgx.sh
```

Then pull models inside the container:

```bash
docker exec -it open-webui ollama pull gpt-oss:20b
docker exec -it open-webui ollama list
```

### `cleanup-openwebui-dgx.sh`
Stops and removes the Open WebUI container while keeping volumes. Also prints optional commands for deeper cleanup.

Use it when you want:
- to remove the running container
- to preserve volumes for rollback
- to clean up old Open WebUI state later

Run:

```bash
bash cleanup-openwebui-dgx.sh
```

## Recommended usage order

### Best long-term setup
1. Run `setup-ollama-dgx.sh`
2. Pull models into host Ollama
3. Run `setup-openwebui-dgx.sh <dgx-spark-ip>` if you want a UI

### Self-contained demo
1. Run `setup-openwebui-integrated-dgx.sh`
2. Pull models inside the container

## Verification commands

Check Ollama:

```bash
curl http://127.0.0.1:11434/api/version
curl http://<dgx-spark-ip>:11434/api/tags
```

Check Open WebUI:

```bash
docker ps
docker logs --tail=50 open-webui
curl http://127.0.0.1:8080
```

## Unified memory — check before running large models

DGX Spark uses **unified memory**: system RAM and GPU VRAM are the same 128 GB pool.
If memory is nearly full, Ollama cannot load a model and inference requests will hang
indefinitely at the HTTP layer (no timeout fires because the TCP connection succeeds
but the model never starts generating).

**Check before running:**

```bash
# Memory overview
free -h

# What models are currently loaded (consuming memory)
ollama ps

# Dashboard at http://localhost:11000/ — check System Memory gauge
```

**Free up memory if needed:**

```bash
# Stop a specific loaded model
ollama stop <model-name>

# Or restart Ollama to unload everything
sudo systemctl restart ollama

# Verify free memory after
free -h
ollama ps   # should show nothing loaded
```

**Rule of thumb for model memory requirements:**

| Model size | FP16 | INT4/Q4 |
|-----------|------|---------|
| 7B  | ~14 GB | ~5 GB  |
| 14B | ~28 GB | ~10 GB |
| 20B | ~40 GB | ~14 GB |
| 32B | ~64 GB | ~22 GB |

Leave at least 10-15 GB headroom for the OS and Ollama overhead.

## Recommendation

For apps like **RLMKit** and **document-assistant**, prefer:

- **host Ollama** as the source of truth
- Open WebUI only as an optional UI layer
