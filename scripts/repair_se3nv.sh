#!/bin/bash
# Repair SE3nv for RFdiffusion (torch 1.9, numpy/scipy, cudatoolkit, CUDA DGL).
#
# Run on a GPU compute node:
#   bash scripts/repair_se3nv.sh
#
# To also install the rfdiffusion Python package + weights, run on a GPU node:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   export RFDIFFUSION_ROOT=$PROJECT_ROOT/RFdiffusion
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/install_rfdiffusion_longleaf_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$PROJECT_ROOT/RFdiffusion}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

echo "=== SE3nv repair ==="
echo "Python: $(python -V)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"

pip install --upgrade pip

pip uninstall -y torch torchvision torchaudio dgl dglgo 2>/dev/null || true
conda remove -y dgl-cuda11.1 2>/dev/null || true

bash "$SCRIPT_DIR/install_se3nv_torch_pip.sh"

pip install "torchdata==0.9.0" --no-deps

echo "Installing cudatoolkit 11.1 (DGL links against conda CUDA runtime)..."
conda install -y --override-channels -c conda-forge cudatoolkit=11.1.1

source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

# Use pip DGL only — conda dgl-cuda11.1 pulls numpy 2.x and breaks torch 1.9.
if ! pip install --no-cache-dir "dgl==1.0.0" -f https://data.dgl.ai/wheels/cu111/repo.html; then
    pip install --no-cache-dir "dgl==1.1.2" -f https://data.dgl.ai/wheels/torch-1.9/cu111/repo.html
fi

# Re-pin after any conda activity (conda dgl may have upgraded numpy/scipy).
pip install "numpy==1.23.5" "scipy==1.10.1"

if ! python -c "import rfdiffusion" 2>/dev/null; then
    echo ""
    echo "rfdiffusion module not installed — installing from $RFDIFFUSION_ROOT ..."
    if [[ ! -d "$RFDIFFUSION_ROOT/env/SE3Transformer" ]]; then
        echo "ERROR: RFdiffusion repo not found at $RFDIFFUSION_ROOT"
        echo "Run on login node: bash scripts/setup_rfdiffusion_longleaf.sh"
        exit 1
    fi
    cd "$RFDIFFUSION_ROOT/env/SE3Transformer"
    pip install --no-cache-dir -r requirements.txt
    python setup.py install
    cd "$RFDIFFUSION_ROOT"
    pip install -e .
fi

python - <<'PY'
import numpy as np
import scipy
import torch
import dgl
import rfdiffusion
print("numpy", np.__version__, "scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("rfdiffusion OK")
PY

python "$SCRIPT_DIR/verify_dgl_cuda.py"

echo "=== SE3nv repair complete ==="
