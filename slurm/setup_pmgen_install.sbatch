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

# PMGen requires CUDA visible during conda env create (tensorflow GPU builds).
# Run from binding-prediction repo root:
#   export PROJECT_ROOT=...
#   sbatch slurm/setup_pmgen_install.sbatch

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-/work/users/$USER/minibinder_prediction}"
export PMGEN_ROOT="${PMGEN_ROOT:-$PROJECT_ROOT/PMGen}"

echo "=== PMGen install on GPU node ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Started: $(date)"

if [[ ! -d "$PMGEN_ROOT" ]]; then
    echo "ERROR: PMGen not cloned at $PMGEN_ROOT"
    echo "Run on login node first: git clone https://github.com/soedinglab/PMGen.git $PMGEN_ROOT"
    exit 1
fi

cd "$PMGEN_ROOT"

# Skip full reinstall if PMGen env + AFfine weights already present (resume-safe).
if conda env list | awk '{print $1}' | grep -qx "PMGen" && [[ -d "$PMGEN_ROOT/AFfine/af_params" ]]; then
    echo "PMGen env and AFfine weights found — running install script in resume/verify mode only."
fi

set +e
bash "$REPO_ROOT/scripts/install_pmgen_longleaf_gpu.sh"
INSTALL_RC=$?
set -e

if [[ $INSTALL_RC -ne 0 ]]; then
    echo "ERROR: install_pmgen_longleaf_gpu.sh failed with exit code $INSTALL_RC"
    echo "Check logs/pmgen_install_${SLURM_JOB_ID}.out for details."
    exit $INSTALL_RC
fi

source "$REPO_ROOT/scripts/pmgen_env.sh"
python -c "import jax, torch; print('jax', jax.__version__, jax.devices()); print('torch cuda', torch.cuda.is_available())"

echo "=== PMGen install complete ==="
echo "Finished: $(date)"
