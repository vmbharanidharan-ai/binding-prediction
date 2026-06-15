#!/bin/bash
# ColabFold / alphafoldenv runtime for Longleaf GPU jobs.
# Sourced by run_colabfold.py and verify_colabfold_gpu.sh.

export PROJECT_ROOT="${PROJECT_ROOT:-}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-${PROJECT_ROOT}/alphafoldenv}"
export COLABFOLD_BIN="${COLABFOLD_BIN:-${ALPHAFOLD_ENV}/bin/colabfold_batch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/colabfold_work_paths.sh
source "${SCRIPT_DIR}/colabfold_work_paths.sh"

module load cuda 2>/dev/null || true

if [[ -f "${ALPHAFOLD_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ALPHAFOLD_ENV}/bin/activate"
fi

export PATH="${ALPHAFOLD_ENV}/bin:${PATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
