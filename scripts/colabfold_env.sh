#!/bin/bash
# ColabFold / alphafoldenv runtime for Longleaf GPU jobs.
# Use the venv where jax-cuda12-plugin was installed (not neo_binder).

export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-${PROJECT_ROOT}/alphafoldenv}"

module load cuda 2>/dev/null || true

if [[ -f "${ALPHAFOLD_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ALPHAFOLD_ENV}/bin/activate"
fi

export PATH="${ALPHAFOLD_ENV}/bin:${PATH}"

# Prefer JAX CUDA plugin wheels over module CUDA libs when needed.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
