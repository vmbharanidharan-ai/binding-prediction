#!/bin/bash
#SBATCH --job-name=neo_step2
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --cpus-per-task=4

# Neo Binder Pipeline — Step 2: Structural scoring + clustering
# CPU-only step (no GPU required)

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Step 2: Structure Scoring ==="

python step2_structure_scoring/interface_metrics.py \
    --input "$NEO_BINDER_WORK_ROOT/step2_scored/parsed_structures.tsv" \
    --output "$NEO_BINDER_WORK_ROOT/step2_scored/interface_metrics.tsv"

python step2_structure_scoring/cluster_structures.py \
    --input "$NEO_BINDER_WORK_ROOT/step2_scored/interface_metrics.tsv" \
    --output "$NEO_BINDER_WORK_ROOT/step2_scored/clustered_structures.tsv"

python step2_structure_scoring/rank_structures.py \
    --input "$NEO_BINDER_WORK_ROOT/step2_scored/clustered_structures.tsv" \
    --output "$NEO_BINDER_WORK_ROOT/step2_scored/ranked_structures.tsv"

echo "=== Step 2 complete ==="
