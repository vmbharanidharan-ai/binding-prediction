#!/bin/bash
# Shared setup for all neo binder pipeline SLURM jobs.

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
mkdir -p logs "$NEO_BINDER_WORK_ROOT"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "Host:     $(hostname)"
echo "Work dir: $NEO_BINDER_WORK_ROOT"
echo "Started:  $(date)"
echo ""
