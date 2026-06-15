#!/bin/bash
# Keep ColabFold weights and cache on /work — not $HOME (Longleaf home quota).
#
# Source from slurm/common.sh, scripts/colabfold_env.sh, and prefetch scripts.
# Requires PROJECT_ROOT to be set.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    return 0 2>/dev/null || true
fi

export COLABFOLD_DATA_DIR="${COLABFOLD_DATA_DIR:-${COLABFOLD_PARAMS_DIR:-${PROJECT_ROOT}/colabfold_params}}"
export COLABFOLD_PARAMS_DIR="${COLABFOLD_PARAMS_DIR:-${COLABFOLD_DATA_DIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"

mkdir -p "${COLABFOLD_DATA_DIR}" "${XDG_CACHE_HOME}/colabfold" "${XDG_CACHE_HOME}"

export COLABFOLD_DATA_DIR COLABFOLD_PARAMS_DIR XDG_CACHE_HOME
