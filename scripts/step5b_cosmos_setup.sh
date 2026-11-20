#!/usr/bin/env bash
# One-time GPU host setup for Step 5b: Cosmos 3 Nano transfer through NVIDIA's Diffusers recipe
# (cosmos-framework .agents/skills/cosmos3-inference-transfer, references/diffusers.md § 1-3).
# Run under nohup; writes SETUP_DONE as the last line on success. Resumable via /root/.np_stage_*.
# Expects the HF token at /root/.cache/huggingface/token (mode 600).
set -euxo pipefail

DIFFUSERS_COMMIT=${DIFFUSERS_COMMIT:-4f788320e9fa974d4bfe0cf5aabc3bd213c39af2}  # main on 2026-11-19, has Cosmos3OmniModularPipeline
VENV=/root/.venv-cosmos3-diffusers
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
export HF_HOME=/root/.cache/huggingface
export HF_HUB_DISABLE_XET=1   # as in the recipe's environment cell

done_stage() { [ -f "/root/.np_stage_$1" ]; }
mark_stage() { touch "/root/.np_stage_$1"; echo "STAGE_DONE $1"; }

if ! done_stage apt; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # libxcb1 libgl1 libglib2.0-0: required before the first import (cosmos_guardrail imports cv2)
  apt-get install -y git ffmpeg libxcb1 libgl1 libglib2.0-0 curl
  uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  mark_stage apt
fi

if ! done_stage venv; then
  # Driver CUDA 13.x -> cu130 backend (recipe § 2).
  uv venv "$VENV" --python 3.13 --seed --managed-python --allow-existing
  uv pip install --python "$VENV/bin/python" --torch-backend=cu130 \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@$DIFFUSERS_COMMIT" \
    accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers
  uv cache clean
  mark_stage venv
fi

if ! done_stage model; then
  "$VENV/bin/hf" download nvidia/Cosmos3-Nano
  mark_stage model
fi

"$VENV/bin/python" - <<'PY'
import torch, diffusers
from diffusers import Cosmos3OmniModularPipeline  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__)
PY
mkdir -p /root/cosmos_results /root/cosmos_inputs
# The recipe's shared negative caption, from the pinned cosmos-framework commit.
curl -LsSf -o /root/cosmos_inputs/negative_prompt.json \
  https://raw.githubusercontent.com/NVIDIA/cosmos-framework/82d6f4e7c08b04f4cfa7c6de103d9be2e30af213/.agents/skills/cosmos3-inference-transfer/assets/negative_prompt.json
"$VENV/bin/python" -c "import json; json.load(open('/root/cosmos_inputs/negative_prompt.json'))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/cosmos_results/gpu.txt
"$VENV/bin/python" -c "import diffusers, torch; print(diffusers.__version__, torch.__version__)" > /root/cosmos_results/versions.txt
echo SETUP_DONE
