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

# DGL links against conda cudatoolkit — fail early if missing.
if [[ ! -f "${CONDA_PREFIX}/lib/libcudart.so" && ! -f "${CONDA_PREFIX}/lib/libcudart.so.11.0" ]]; then
    echo "WARN: cudatoolkit not found in SE3nv — run: conda install -c conda-forge cudatoolkit=11.1.1"
fi

export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"

# Show which DGL binary is used (CPU-only wheels cause 'Range does not support cuda').
if python -c "import dgl; print('DGL', dgl.__version__, 'at', dgl.__file__)" 2>/dev/null; then
    :
fi
