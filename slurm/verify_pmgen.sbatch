#!/bin/bash
#SBATCH --job-name=pmgen_verify
#SBATCH --output=logs/pmgen_verify_%j.out
#SBATCH --error=logs/pmgen_verify_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-/work/users/$USER/minibinder_prediction}"
export PMGEN_ROOT="${PMGEN_ROOT:-$PROJECT_ROOT/PMGen}"

echo "=== PMGen GPU verify ==="
echo "Host: $(hostname)"
bash "$REPO_ROOT/scripts/verify_pmgen_login.sh"
source "$REPO_ROOT/scripts/pmgen_env.sh"
python -c "import jax, torch, PANDORA; print('jax', jax.devices()); print('torch cuda', torch.cuda.is_available()); print('PANDORA OK')"
echo "=== PMGen GPU verify complete ==="
