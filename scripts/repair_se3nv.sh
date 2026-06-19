#!/bin/bash
# Repair SE3nv for RFdiffusion (torch 1.9, numpy/scipy, cudatoolkit, CUDA DGL).
#
# Run on a GPU compute node:
#   conda activate SE3nv   # optional; script activates SE3nv
#   bash scripts/repair_se3nv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

echo "=== SE3nv repair ==="
echo "Python: $(python -V)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"

pip install --upgrade pip

pip uninstall -y torch torchvision torchaudio dgl dglgo 2>/dev/null || true

bash "$SCRIPT_DIR/install_se3nv_torch_pip.sh"

pip install "torchdata==0.9.0" --no-deps

echo "Installing cudatoolkit 11.1 (DGL links against conda CUDA runtime)..."
conda install -y --override-channels -c conda-forge cudatoolkit=11.1.1

source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

pip uninstall -y dgl dglgo 2>/dev/null || true

dgl_installed=0
if conda install -y -c dglteam -c conda-forge "dgl-cuda11.1=0.9.1post1"; then
    echo "Installed dgl-cuda11.1 0.9.1post1 (conda dglteam)"
    dgl_installed=1
elif pip install --no-cache-dir "dgl==1.0.0" -f https://data.dgl.ai/wheels/cu111/repo.html; then
    echo "Installed dgl 1.0.0 (pip cu111 repo)"
    dgl_installed=1
elif pip install --no-cache-dir "dgl==1.1.2" -f https://data.dgl.ai/wheels/torch-1.9/cu111/repo.html; then
    echo "Installed dgl 1.1.2 (pip torch-1.9 cu111 repo)"
    dgl_installed=1
fi

if [[ $dgl_installed -eq 0 ]]; then
    echo "ERROR: Could not install a compatible CUDA DGL package."
    exit 1
fi

python - <<'PY'
import torch
import dgl
import rfdiffusion
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("rfdiffusion OK")
PY

python "$SCRIPT_DIR/verify_dgl_cuda.py"

echo "=== SE3nv repair complete ==="
