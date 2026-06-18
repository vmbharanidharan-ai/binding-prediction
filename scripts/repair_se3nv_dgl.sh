#!/bin/bash
# Repair DGL/torchdata pins in an existing SE3nv env (no full reinstall).
#
# Fixes:
#   - ModuleNotFoundError: No module named 'torchdata.datapipes'
#   - DGLError: Operator Range does not support cuda device
#
# Run on a GPU compute node after: conda activate SE3nv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

echo "=== SE3nv DGL repair ==="
echo "Python: $(python -V)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"

pip install --upgrade pip

# torchdata 0.9.x still ships datapipes (required by DGL / RFdiffusion).
# Do not --force-reinstall: that can pull torch 2.x and break RFdiffusion.
pip install "torchdata==0.9.0" --no-deps
pip install "numpy==1.23.5" "scipy==1.10.1"

echo "Installing cudatoolkit 11.1 (DGL links against conda CUDA runtime, not PyTorch bundled libs)..."
conda install -y --override-channels -c conda-forge cudatoolkit=11.1.1

source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

pip uninstall -y dgl dglgo 2>/dev/null || true

install_dgl_pip() {
    echo "Trying pip: $*"
    pip install --no-cache-dir "$@"
}

dgl_installed=0
# Official RFdiffusion stack uses conda dgl-cuda11.1; prefer when available.
if conda install -y -c dglteam -c conda-forge "dgl-cuda11.1=0.9.1post1"; then
    echo "Installed dgl-cuda11.1 0.9.1post1 (conda dglteam)"
    dgl_installed=1
elif install_dgl_pip "dgl==1.0.0" -f https://data.dgl.ai/wheels/cu111/repo.html; then
    echo "Installed dgl 1.0.0 (pip cu111 repo)"
    dgl_installed=1
elif install_dgl_pip "dgl==1.1.2" -f https://data.dgl.ai/wheels/torch-1.9/cu111/repo.html; then
    echo "Installed dgl 1.1.2 (pip torch-1.9 cu111 repo)"
    dgl_installed=1
fi

if [[ $dgl_installed -eq 0 ]]; then
    echo "ERROR: Could not install a compatible CUDA DGL package."
    echo "Try a full reinstall: bash scripts/install_rfdiffusion_longleaf_gpu.sh"
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

echo "=== SE3nv DGL repair complete ==="
