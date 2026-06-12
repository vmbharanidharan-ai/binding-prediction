#!/bin/bash
#SBATCH --job-name=pandora_build
#SBATCH --output=logs/pandora_build_%j.out
#SBATCH --error=logs/pandora_build_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=general

# Fallback: build PANDORA database from IMGT (30–60+ min). Use only if fetch fails.

set -euo pipefail

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-/work/users/$USER/minibinder_prediction}"

bash "$REPO_ROOT/scripts/build_pandora_database.sh"
