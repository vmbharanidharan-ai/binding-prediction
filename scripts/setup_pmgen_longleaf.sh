#!/bin/bash
# PMGen install helper for Longleaf.
#
# Phase 1 (login node): clone PMGen repo
# Phase 2 (GPU node via sbatch): conda env create — requires CUDA during install
#
# Usage:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/setup_pmgen_longleaf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
PMGEN_ROOT="${PMGEN_ROOT:-$PROJECT_ROOT/PMGen}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PMGEN_ROOT=$PMGEN_ROOT"
echo "REPO_ROOT=$REPO_ROOT"

# --- Phase 1: clone only (safe on login node) ---
if [[ ! -d "$PMGEN_ROOT/.git" ]]; then
    git clone https://github.com/soedinglab/PMGen.git "$PMGEN_ROOT"
else
    echo "PMGen repo already exists at $PMGEN_ROOT"
fi

mkdir -p "$REPO_ROOT/logs"

echo ""
echo "PMGen conda env MUST be created on a GPU node (login nodes have no CUDA)."
echo "Submitting install job to SLURM..."
echo ""

cd "$REPO_ROOT"
export PROJECT_ROOT PMGEN_ROOT
unset SBATCH_QOS SLURM_QOS 2>/dev/null || true

JOB_ID=$(sbatch --export=ALL slurm/setup_pmgen_install.sbatch | awk '{print $NF}')
echo "Submitted install job: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO_ROOT/logs/pmgen_install_${JOB_ID}.out"
echo ""
echo "When install finishes, verify:"
echo "  conda activate PMGen"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "Then submit Step 1:"
echo "  export NEO_BINDER_WORK_ROOT=\$PROJECT_ROOT/work"
echo "  source data/generated/current_pair.env"
echo "  ./slurm/submit_step.sh 1"
