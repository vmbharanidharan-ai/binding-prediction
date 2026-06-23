#!/bin/bash
# RFdiffusion / SE3nv runtime for Longleaf GPU jobs.

source "$(conda info --base)/etc/profile.d/conda.sh"

# alphafoldenv or other venvs prepended to PATH must not override SE3nv.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || unset VIRTUAL_ENV
fi

conda activate "${RFDIFFUSION_ENV:-SE3nv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

if ! ls "${CONDA_PREFIX}/lib"/libcudart.so* "${CONDA_PREFIX}/lib64"/libcudart.so* 2>/dev/null | head -1 | grep -q .; then
    echo "WARN: libcudart not found — run: conda install -c conda-forge cudatoolkit=11.1.1"
fi

export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"

# Show which DGL binary is used (CPU-only wheels cause 'Range does not support cuda').
if python -c "import dgl; print('DGL', dgl.__version__, 'at', dgl.__file__)" 2>/dev/null; then
    :
fi
