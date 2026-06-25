#!/bin/bash
# Shared path exports for all pipeline SLURM jobs (no conda activation).

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/project_env.sh
source "${_SLURM_DIR}/../scripts/project_env.sh"

export PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -n "$PROJECT_ROOT" && -z "${NEO_BINDER_WORK_ROOT:-}" ]]; then
    export NEO_BINDER_WORK_ROOT="$PROJECT_ROOT/work"
fi
export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
export INPUT_TSV="${INPUT_TSV:-data/step5_input.tsv}"
export PMGEN_ROOT="${PMGEN_ROOT:-${PROJECT_ROOT}/PMGen}"
export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
export RFDIFFUSION_CONTAINER="${RFDIFFUSION_CONTAINER:-${PROJECT_ROOT}/rfdiffusion.sif}"
export PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-${PROJECT_ROOT}/ProteinMPNN}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-${PROJECT_ROOT}/alphafoldenv}"
export COLABFOLD_BIN="${COLABFOLD_BIN:-${ALPHAFOLD_ENV}/bin/colabfold_batch}"

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/colabfold_work_paths.sh
source "${_SLURM_DIR}/../scripts/colabfold_work_paths.sh"

mkdir -p logs "$NEO_BINDER_WORK_ROOT"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
