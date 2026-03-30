#!/usr/bin/env bash
set -euo pipefail

echo "==> Stop and remove Open WebUI container (keep volumes)"
docker stop open-webui 2>/dev/null || true
docker rm open-webui 2>/dev/null || true

echo "==> Remaining related volumes"
docker volume ls | grep open-webui || true

cat <<'EOM'

To remove only old integrated model data later:
  docker run --rm -it \
    -v open-webui-ollama:/root/.ollama \
    ollama/ollama:latest \
    ollama rm gpt-oss:20b

To remove all Open WebUI state:
  docker volume rm open-webui
  docker volume rm open-webui-ollama
EOM
