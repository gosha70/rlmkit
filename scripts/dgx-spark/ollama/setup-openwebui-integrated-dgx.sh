#!/usr/bin/env bash
set -euo pipefail

echo "==> Recreate integrated Open WebUI + Ollama container"
docker rm -f open-webui 2>/dev/null || true

docker run -d \
  --name open-webui \
  --restart unless-stopped \
  --gpus=all \
  -p 8080:8080 \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  ghcr.io/open-webui/open-webui:ollama

echo "==> Container status"
docker ps --filter name=open-webui
echo
echo "==> Recent logs"
docker logs --tail=50 open-webui || true

cat <<'EOM'

To pull a model into the integrated container:
  docker exec -it open-webui ollama pull gpt-oss:20b
  docker exec -it open-webui ollama list

Recommendation:
  Use this mode only for a self-contained demo.
  For apps and RLM Studio, prefer host Ollama + OLLAMA_BASE_URL.
EOM
