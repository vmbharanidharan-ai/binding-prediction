#!/bin/bash
# Activate official PMGen conda env on Longleaf GPU jobs.
# Mirrors: conda activate PMGen  (from https://github.com/soedinglab/PMGen)
#
# Longleaf-only addition: unset LD_LIBRARY_PATH so pip jaxlib (cuda11) is not
# overridden by `module load cuda` (prevents SIGABRT on jax.devices()).

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PMGEN_ENV:-PMGen}"

module load cuda 2>/dev/null || true
unset LD_LIBRARY_PATH

export PMGEN_ROOT="${PMGEN_ROOT:-${PROJECT_ROOT}/PMGen}"
