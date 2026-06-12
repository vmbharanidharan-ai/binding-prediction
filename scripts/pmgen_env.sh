#!/bin/bash
# PMGen runtime env for Longleaf GPU jobs.
# JAX 0.2.x pip wheels bundle CUDA 11 libs; module cuda pollutes LD_LIBRARY_PATH → SIGABRT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PMGEN_ENV:-PMGen}"

# Driver visible to nvidia-smi / JAX; do not point JAX at module CUDA libs.
module load cuda 2>/dev/null || true
unset LD_LIBRARY_PATH
