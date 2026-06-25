#!/bin/bash
# Load semi-permanent project paths from $PROJECT_ROOT/.env (written by init_project.sh).
#
# Sourced by slurm/common_paths.sh and submit_step.sh when binding-prediction lives at
#   $PROJECT_ROOT/binding-prediction/

_neo_repo_root() {
    if [[ -n "${BASH_SOURCE[1]:-}" ]]; then
        cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd
        return
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/slurm" ]]; then
        echo "${SLURM_SUBMIT_DIR}"
        return
    fi
    pwd
}

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    _repo="$(_neo_repo_root)"
    if [[ -f "${_repo}/../.env" ]]; then
        # shellcheck source=/dev/null
        source "${_repo}/../.env"
    fi
fi
