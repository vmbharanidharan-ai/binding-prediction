#!/bin/bash
#SBATCH --job-name=rfdiff_install
#SBATCH --output=logs/rfdiffusion_install_%j.out
#SBATCH --error=logs/rfdiffusion_install_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-/work/users/$USER/minibinder_prediction}"
export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$PROJECT_ROOT/RFdiffusion}"

echo "=== RFdiffusion install on GPU node ==="
echo "Host: $(hostname)"
echo "RFDIFFUSION_ROOT: $RFDIFFUSION_ROOT"
echo "Started: $(date)"

if [[ ! -d "$RFDIFFUSION_ROOT" ]]; then
    echo "ERROR: RFdiffusion not cloned at $RFDIFFUSION_ROOT"
    echo "Run on login node first: bash scripts/setup_rfdiffusion_longleaf.sh"
    exit 1
fi

set +e
bash "$REPO_ROOT/scripts/install_rfdiffusion_longleaf_gpu.sh"
INSTALL_RC=$?
set -e

if [[ $INSTALL_RC -ne 0 ]]; then
    echo "ERROR: install_rfdiffusion_longleaf_gpu.sh failed with exit code $INSTALL_RC"
    exit $INSTALL_RC
fi

source "$REPO_ROOT/scripts/rfdiffusion_env.sh"
bash "$REPO_ROOT/scripts/verify_se3nv_rfdiffusion.sh"

echo "=== RFdiffusion install complete ==="
echo "Finished: $(date)"
