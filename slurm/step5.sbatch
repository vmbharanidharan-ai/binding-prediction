#!/bin/bash
#SBATCH --job-name=neo_step5
#SBATCH --output=logs/step5_%j.out
#SBATCH --error=logs/step5_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --cpus-per-task=4

# Step 5: ML ranking (CPU)
# Requires: embeddings (step0), step2, step4 outputs
# Output: work/step5_ranked/final_rankings.tsv

source "$(dirname "$0")/common.sh"

echo "=== Step 5: ML Ranking ==="
python run_pipeline.py --step step5 --input data/step5_input.tsv
echo "=== Step 5 complete ==="
