#!/bin/bash
#SBATCH --job-name=neo_step2
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --cpus-per-task=4

# Step 2: Structural scoring + clustering (CPU)
# Output: work/step2_scored/ranked_structures.tsv

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

echo "=== Step 2: Structure Scoring ==="
python run_pipeline.py --step step2 --input "$INPUT_TSV"
echo "=== Step 2 complete ==="
