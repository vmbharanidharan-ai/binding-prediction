#!/bin/bash
#SBATCH --job-name=pandora_fetch
#SBATCH --output=logs/pandora_fetch_%j.out
#SBATCH --error=logs/pandora_fetch_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Download official pre-built PANDORA database to /work (needs network).
#
# Submit from binding-prediction repo:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   bash scripts/setup_pandora_database.sh

set -euo pipefail

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-/work/users/$USER/minibinder_prediction}"

bash "$REPO_ROOT/scripts/fetch_pandora_database.sh"
