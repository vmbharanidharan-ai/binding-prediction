#!/bin/bash
#SBATCH --job-name=pmgen_install
#SBATCH --output=logs/pmgen_install_%j.out
#SBATCH --error=logs/pmgen_install_%j.err
#SBATCH --time=04:00:00
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
export PMGEN_ROOT="${PMGEN_ROOT:-$PROJECT_ROOT/PMGen}"
export PMGEN_INSTALL_MODE="${PMGEN_INSTALL_MODE:-longleaf}"

echo "=== PMGen install on GPU node ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"
echo "Install mode: $PMGEN_INSTALL_MODE"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Started: $(date)"

if [[ ! -d "$PMGEN_ROOT" ]]; then
    echo "ERROR: PMGen not cloned at $PMGEN_ROOT"
    exit 1
fi

cd "$PMGEN_ROOT"

INSTALL_RC=0
if [[ "$PMGEN_INSTALL_MODE" == "official" ]]; then
    set +e
    bash "$REPO_ROOT/scripts/install_pmgen_official_gpu.sh"
    INSTALL_RC=$?
    set -e
    if [[ $INSTALL_RC -ne 0 ]]; then
        echo "Official install.sh failed (exit $INSTALL_RC) — falling back to Longleaf-equivalent install..."
        bash "$REPO_ROOT/scripts/install_pmgen_longleaf_gpu.sh"
    fi
else
    bash "$REPO_ROOT/scripts/install_pmgen_longleaf_gpu.sh"
fi

source "$REPO_ROOT/scripts/pmgen_env.sh"
python -c "import PANDORA, jax, torch; print('PANDORA OK'); print('jax', jax.devices()); print('torch cuda', torch.cuda.is_available())"

echo "=== PMGen install complete ==="
echo "Finished: $(date)"
