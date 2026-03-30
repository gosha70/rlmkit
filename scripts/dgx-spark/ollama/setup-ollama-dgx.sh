#!/usr/bin/env bash
set -euo pipefail

echo "==> Install Ollama"
curl -fsSL https://ollama.com/install.sh | sh

echo "==> Create host model directory"
sudo mkdir -p /var/lib/ollama/models
sudo chown -R ollama:ollama /var/lib/ollama

echo "==> Create systemd override"
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/var/lib/ollama/models"
EOF

echo "==> Reload and restart service"
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama

echo "==> Verify service"
sudo systemctl status ollama --no-pager || true
sudo ss -lntp | grep 11434 || true
curl -fsS http://127.0.0.1:11434/api/version || true

cat <<'EOM'

Next steps:
  ollama pull llama3.2
  ollama pull gpt-oss:20b
  ollama pull qwen2.5:14b
  ollama list

Remote verification:
  curl http://<dgx-spark-ip>:11434/
  curl http://<dgx-spark-ip>:11434/api/version
  curl http://<dgx-spark-ip>:11434/api/tags
EOM
