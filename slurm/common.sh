#!/bin/bash
# Shared setup for all neo binder pipeline SLURM jobs.

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

export PROJECT_ROOT="${PROJECT_ROOT:-}"
export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
export INPUT_TSV="${INPUT_TSV:-data/step5_input.tsv}"
export PMGEN_ROOT="${PMGEN_ROOT:-${PROJECT_ROOT}/PMGen}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-${PROJECT_ROOT}/alphafoldenv}"
if [[ -d "${ALPHAFOLD_ENV}/bin" ]]; then
    export PATH="${ALPHAFOLD_ENV}/bin:$PATH"
fi
mkdir -p logs "$NEO_BINDER_WORK_ROOT"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Auto-build IMGT HLA index on first run (no manual FASTA download needed)
if [[ ! -f hla/hla_index.pkl ]]; then
    echo "HLA index not found — downloading IMGT and building index..."
    python hla/setup_hla.py
fi

echo "Host:     $(hostname)"
echo "Work dir: $NEO_BINDER_WORK_ROOT"
echo "Input:    $INPUT_TSV"
echo "Started:  $(date)"
echo ""
