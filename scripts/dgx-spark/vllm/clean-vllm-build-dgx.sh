#!/usr/bin/env bash
set -euo pipefail

cd ~/dgx-spark-vllm/vllm 2>/dev/null || {
  echo "Repo not found at ~/dgx-spark-vllm/vllm"
  exit 0
}

echo "==> Remove local build artifacts"
rm -rf build/ .deps/ _skbuild/ *.egg-info
find . -name CMakeCache.txt -delete
find . -name CMakeFiles -type d -prune -exec rm -rf {} +
find . -name '*.so' -delete
find . -name '*.o' -delete
find . -name '*.a' -delete

echo "==> Clear pip cache"
source ~/dgx-spark-vllm/vllm_env/bin/activate 2>/dev/null || true
pip cache purge || true

echo "Cleanup complete."
