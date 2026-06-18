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

# torch 1.9 + from_numpy breaks with numpy 1.26+ (TypeError: expected np.ndarray).
# scipy 1.12+ expects numpy 2.x ABI — pin both for DGL import stability.
pip install "numpy==1.23.5" "scipy==1.10.1"

python - <<'PY'
import torch
v = torch.__version__
if not v.startswith("1.9") and not v.startswith("1.10"):
    raise SystemExit(f"ERROR: expected torch 1.9.x, got {v}")
import numpy as np
x = np.zeros((2, 3), dtype=np.float32)
_ = torch.from_numpy(x)
print("torch", v, "cuda", torch.cuda.is_available(), "numpy", np.__version__)
PY
