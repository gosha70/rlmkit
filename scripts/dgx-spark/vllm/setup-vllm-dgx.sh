#!/usr/bin/env bash
set -euo pipefail

echo "==> Install system dependencies"
sudo apt-get update
sudo apt-get install -y \
  gcc-12 g++-12 build-essential cmake ninja-build git \
  python3.12-dev python3-dev python3-venv python3-full \
  libnuma-dev numactl

echo "==> Create working directory and venv"
mkdir -p ~/dgx-spark-vllm
cd ~/dgx-spark-vllm
python3 -m venv vllm_env
source vllm_env/bin/activate

echo "==> Upgrade packaging tools"
pip install --upgrade pip setuptools wheel

echo "==> Install CUDA-enabled PyTorch"
pip install --no-cache-dir \
  'torch==2.10.0+cu130' \
  'torchvision==0.25.0+cu130' \
  'torchaudio==2.10.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130

echo "==> Clone vLLM"
if [ ! -d ~/dgx-spark-vllm/vllm ]; then
  git clone https://github.com/vllm-project/vllm.git ~/dgx-spark-vllm/vllm
fi
cd ~/dgx-spark-vllm/vllm

echo "==> Configure build environment"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="12.1a"
export MAX_JOBS=8

echo "==> Install build requirements and build vLLM"
pip install -r requirements/build.txt
pip install --no-build-isolation -e .

echo "==> Verify torch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda built:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

cat <<'EOM'

Setup complete.

Recommended next steps:
  cd ~/dgx-spark-vllm/vllm
  source ~/dgx-spark-vllm/vllm_env/bin/activate
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000

Then verify:
  curl http://127.0.0.1:8000/v1/models
  curl http://<dgx-spark-ip>:8000/v1/models
EOM
