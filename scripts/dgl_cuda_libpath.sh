#!/bin/bash
# Conda cudatoolkit libs required by pip/conda DGL (PyTorch cu111 wheels bundle CUDA; DGL does not).
# Source after: conda activate SE3nv
#
# Replace (do not append) so Longleaf module cuda from parent SLURM jobs cannot override cu111 libs.

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
fi
