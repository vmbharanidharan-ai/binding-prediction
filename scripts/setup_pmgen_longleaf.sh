#!/bin/bash
# PMGen install helper for Longleaf.
#
# Official release: https://github.com/soedinglab/PMGen
#   git clone ... && cd PMGen && bash -l install.sh --no_modeller && conda activate PMGen
#
# On Longleaf, official install.sh often fails (PMGen.yml channel priority / no CUDA on login).
# This helper clones PMGen on the login node, then runs a GPU sbatch install that produces
# the same outcome (PMGen env, PANDORA, AFfine weights, ProteinMPNN).
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
PMGEN_INSTALL_MODE="${PMGEN_INSTALL_MODE:-longleaf}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PMGEN_ROOT=$PMGEN_ROOT"
echo "REPO_ROOT=$REPO_ROOT"
echo "PMGEN_INSTALL_MODE=$PMGEN_INSTALL_MODE (longleaf | official)"

if [[ ! -d "$PMGEN_ROOT/.git" ]]; then
    git clone https://github.com/soedinglab/PMGen.git "$PMGEN_ROOT"
else
    echo "PMGen repo already exists at $PMGEN_ROOT"
fi

mkdir -p "$REPO_ROOT/logs"

echo ""
echo "PMGen must be installed on a GPU node (official install.sh requires CUDA)."
echo "Submitting install job to SLURM..."
echo ""

cd "$REPO_ROOT"
export PROJECT_ROOT PMGEN_ROOT PMGEN_INSTALL_MODE
unset SBATCH_QOS SLURM_QOS 2>/dev/null || true

JOB_ID=$(sbatch --export=ALL slurm/setup_pmgen_install.sbatch | awk '{print $NF}')
echo "Submitted install job: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO_ROOT/logs/pmgen_install_${JOB_ID}.out"
echo ""
echo "Verify (login):  bash scripts/verify_pmgen_login.sh"
echo "Verify (GPU):    sbatch slurm/verify_pmgen.sbatch"
echo ""
echo "Official manual test (on GPU node after install):"
echo "  conda activate PMGen && cd $PMGEN_ROOT"
echo "  python run_PMGen.py --mode wrapper --run single --df input.tsv --output_dir out/ --initial_guess"
echo ""
echo "Submit Step 1:"
echo "  export NEO_BINDER_WORK_ROOT=\$PROJECT_ROOT/work"
echo "  source data/generated/current_pair.env"
echo "  ./slurm/submit_step.sh 1"
