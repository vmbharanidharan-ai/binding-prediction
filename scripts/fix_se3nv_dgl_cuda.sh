#!/bin/bash
# Diagnose and repair DGL CUDA on Longleaf (Operator Range does not support cuda device).
#
# Run on a GPU node:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/fix_se3nv_dgl_cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$PROJECT_ROOT/RFdiffusion}"

echo "=========================================="
echo "SE3nv DGL CUDA diagnostic"
echo "=========================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "RFDIFFUSION_ROOT: $RFDIFFUSION_ROOT"
echo ""

if ! nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    echo "ERROR: No GPU visible — request a GPU node with srun first."
    exit 1
fi
echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

# shellcheck source=scripts/dgl_cuda_libpath.sh
source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

echo "=== Environment ==="
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo ""

echo "=== CUDA libs in conda env ==="
for lib in libcudart.so libcudart.so.11.0 libnvrtc.so; do
    found="$(find "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64" -name "${lib}*" 2>/dev/null | head -1 || true)"
    if [[ -n "$found" ]]; then
        echo "  OK $found"
    else
        echo "  MISSING $lib"
    fi
done
echo ""

echo "=== Package versions ==="
python - <<'PY'
import dgl
import numpy as np
import scipy
import torch

print(f"torch:  {torch.__version__} (cuda: {torch.cuda.is_available()})")
print(f"numpy:  {np.__version__}")
print(f"scipy:  {scipy.__version__}")
print(f"dgl:    {dgl.__version__}")
print(f"dgl at: {dgl.__file__}")
PY
echo ""

echo "=== DGL CUDA test (g.edges on GPU) ==="
set +e
python "$SCRIPT_DIR/verify_dgl_cuda.py"
test_rc=$?
set -e
echo ""

if [[ $test_rc -eq 0 ]]; then
    echo "=========================================="
    echo "DGL CUDA OK — no repair needed"
    echo "=========================================="
    echo "Next: bash scripts/verify_se3nv_rfdiffusion.sh"
    exit 0
fi

echo "=========================================="
echo "DGL CUDA FAILED — running repair_se3nv.sh"
echo "=========================================="
bash "$SCRIPT_DIR/repair_se3nv.sh"

source "$SCRIPT_DIR/dgl_cuda_libpath.sh"
python "$SCRIPT_DIR/verify_dgl_cuda.py"

echo ""
echo "=========================================="
echo "Repair complete — re-run Step 3 preflight:"
echo "  bash scripts/verify_se3nv_rfdiffusion.sh"
echo "=========================================="
