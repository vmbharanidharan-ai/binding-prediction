#!/bin/bash
# Shared setup for pipeline orchestration SLURM jobs (neo_binder Python).
#
# Step-specific runtimes (ColabFold, RFdiffusion, ProteinMPNN) activate their own
# envs inside subprocess wrappers — do NOT prepend alphafoldenv to PATH here.

set -euo pipefail

# shellcheck source=slurm/common_paths.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

# Auto-build IMGT HLA index on first run (no manual FASTA download needed)
if [[ ! -f hla/hla_index.pkl ]]; then
    echo "HLA index not found — downloading IMGT and building index..."
    python hla/setup_hla.py
fi

echo "Host:     $(hostname)"
echo "Work dir: $NEO_BINDER_WORK_ROOT"
echo "Pipeline: $(which python) (neo_binder)"
echo "ColabFold data: ${COLABFOLD_DATA_DIR:-<unset>}"
echo "XDG cache:      ${XDG_CACHE_HOME:-<unset>}"
echo "Input:    $INPUT_TSV"
echo "Started:  $(date)"
echo ""
