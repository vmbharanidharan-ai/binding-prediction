#!/bin/bash
# RFdiffusion install helper for Longleaf.
#
# Phase 1 (login): clone RFdiffusion
# Phase 2 (GPU node via sbatch): create SE3nv env + weights
#
# Usage:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/setup_rfdiffusion_longleaf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$PROJECT_ROOT/RFdiffusion}"
RFDIFFUSION_WEIGHTS="${RFDIFFUSION_WEIGHTS:-$PROJECT_ROOT/rfdiffusion_weights}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RFDIFFUSION_ROOT=$RFDIFFUSION_ROOT"
echo "RFDIFFUSION_WEIGHTS=$RFDIFFUSION_WEIGHTS"
echo "REPO_ROOT=$REPO_ROOT"

if [[ ! -d "$RFDIFFUSION_ROOT/.git" ]]; then
    git clone https://github.com/RosettaCommons/RFdiffusion.git "$RFDIFFUSION_ROOT"
else
    echo "RFdiffusion repo already exists at $RFDIFFUSION_ROOT"
fi

mkdir -p "$REPO_ROOT/logs"

echo ""
echo "SE3nv env should be created on a GPU node (DGL + CUDA)."
echo "Submitting install job to SLURM..."
echo ""

cd "$REPO_ROOT"
export PROJECT_ROOT RFDIFFUSION_ROOT RFDIFFUSION_WEIGHTS
unset SBATCH_QOS SLURM_QOS 2>/dev/null || true

JOB_ID=$(sbatch --export=ALL slurm/setup_rfdiffusion_install.sbatch | awk '{print $NF}')
echo "Submitted install job: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO_ROOT/logs/rfdiffusion_install_${JOB_ID}.out"
echo ""
echo "When install finishes, verify:"
echo "  source scripts/rfdiffusion_env.sh"
echo "  python -c \"import torch, dgl; print(torch.cuda.is_available(), dgl.__version__)\""
echo ""
echo "Then patch config (once):"
echo "  export RFDIFFUSION_ROOT=$RFDIFFUSION_ROOT"
echo "  SE3PY=\$(conda run -n SE3nv which python)"
echo "  sed -i \"s|rfdiffusion_cmd:.*|rfdiffusion_cmd: \\\"\$SE3PY \$RFDIFFUSION_ROOT/scripts/run_inference.py\\\"|\" config/config.yaml"
