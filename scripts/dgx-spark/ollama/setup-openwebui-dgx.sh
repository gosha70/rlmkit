#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST_IP="${1:-127.0.0.1}"

echo "==> Recreate Open WebUI against host Ollama: ${OLLAMA_HOST_IP}:11434"
docker rm -f open-webui 2>/dev/null || true

docker run -d \
  --name open-webui \
  --restart unless-stopped \
  -p 8080:8080 \
  -e OLLAMA_BASE_URL="http://${OLLAMA_HOST_IP}:11434" \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

echo "==> Container status"
docker ps --filter name=open-webui
echo
echo "==> Recent logs"
docker logs --tail=50 open-webui || true

cat <<EOM

Open WebUI is configured to use host Ollama at:
  http://${OLLAMA_HOST_IP}:11434

Verify:
  curl http://127.0.0.1:8080
  curl http://<dgx-spark-ip>:8080
EOM
