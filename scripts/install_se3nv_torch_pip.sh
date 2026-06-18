#!/bin/bash
# Install PyTorch 1.9.1+cu111 via pip (works when conda solve fails on py3.9).
# Must be run with SE3nv already activated.

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "SE3nv" ]]; then
    echo "ERROR: activate SE3nv first: conda activate SE3nv"
    exit 1
fi

echo "Installing PyTorch 1.9.1+cu111 via pip..."
pip install --upgrade pip
pip install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

python - <<'PY'
import torch
v = torch.__version__
if not v.startswith("1.9") and not v.startswith("1.10"):
    raise SystemExit(f"ERROR: expected torch 1.9.x, got {v}")
print("torch", v, "cuda", torch.cuda.is_available())
PY
