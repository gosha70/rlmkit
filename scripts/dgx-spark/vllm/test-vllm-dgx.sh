#!/usr/bin/env bash
set -euo pipefail

cd ~/dgx-spark-vllm/vllm
source ~/dgx-spark-vllm/vllm_env/bin/activate

cat > test_vllm_install.py <<'PY'
from vllm import LLM, SamplingParams
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = llm.generate(["Hello from DGX Spark"], params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
PY

python test_vllm_install.py
