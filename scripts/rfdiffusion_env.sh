#!/bin/bash
# RFdiffusion / SE3nv runtime for Longleaf GPU jobs.

source "$(conda info --base)/etc/profile.d/conda.sh"

# alphafoldenv or other venvs prepended to PATH must not override SE3nv.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || unset VIRTUAL_ENV
fi

conda activate "${RFDIFFUSION_ENV:-SE3nv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DGL requires conda cudatoolkit libs. Do NOT module-load Longleaf CUDA 12 — it breaks cu111 DGL.
source "$SCRIPT_DIR/dgl_cuda_libpath.sh"

export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
