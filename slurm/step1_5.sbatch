#!/bin/bash
#SBATCH --job-name=neo_step1_5
#SBATCH --output=logs/step1_5_%j.out
#SBATCH --error=logs/step1_5_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=general

# Step 1.5 (optional): Truncate peptide–HLA complexes to HLA groove for RFdiffusion
# Input:  work/step2_scored/parsed_structures.tsv (from Step 1)
# Output: work/step1_5_truncated/  (Step 1 outputs unchanged)

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

echo "=== Step 1.5: Truncate HLA groove ==="
python run_pipeline.py --step step1_5
echo "=== Step 1.5 complete ==="
