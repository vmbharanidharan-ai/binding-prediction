#!/bin/bash
# Conda cudatoolkit libs required by pip DGL (PyTorch cu111 wheels bundle CUDA; DGL does not).
# Source after: conda activate SE3nv
#
# DGL libdgl.so needs libcudart from conda cudatoolkit at runtime on Longleaf.

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    _dgl_libs=()
    [[ -d "${CONDA_PREFIX}/lib" ]] && _dgl_libs+=("${CONDA_PREFIX}/lib")
    [[ -d "${CONDA_PREFIX}/lib64" ]] && _dgl_libs+=("${CONDA_PREFIX}/lib64")
    if [[ ${#_dgl_libs[@]} -gt 0 ]]; then
        export LD_LIBRARY_PATH="$(
            IFS=:
            echo "${_dgl_libs[*]}"
        )"
    fi
fi
