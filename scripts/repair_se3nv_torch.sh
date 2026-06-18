#!/bin/bash
# Downgrade SE3nv to PyTorch 1.9 + DGL 1.0 for RFdiffusion inference.
#
# Run on a GPU compute node when SE3nv has torch 2.x (breaks RFdiffusion).
# Usage: bash scripts/repair_se3nv_torch.sh

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "SE3nv"; then
    echo "ERROR: SE3nv env not found. Run: bash scripts/install_rfdiffusion_longleaf_gpu.sh"
    exit 1
fi

conda activate SE3nv
module load cuda 2>/dev/null || true

CURRENT="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
echo "=== SE3nv PyTorch repair ==="
echo "Current torch: $CURRENT"

if python - <<'PY'
import torch
major = int(torch.__version__.split(".")[0])
minor = int(torch.__version__.split(".")[1].split("+")[0])
raise SystemExit(0 if major == 1 and minor <= 10 else 1)
PY
then
    echo "Torch version already compatible with RFdiffusion (1.9–1.10)."
    bash "$(dirname "$0")/repair_se3nv_dgl.sh"
    exit 0
fi

echo "Removing incompatible torch/dgl..."
pip uninstall -y torch torchvision torchaudio dgl dglgo 2>/dev/null || true

# Conda solve often fails on py3.9 (pillow/jpeg conflicts); pip cu111 wheels are reliable.
bash "$(dirname "$0")/install_se3nv_torch_pip.sh"

bash "$(dirname "$0")/repair_se3nv_dgl.sh"

echo "=== SE3nv PyTorch repair complete ==="
