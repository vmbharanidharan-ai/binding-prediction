#!/bin/bash
#SBATCH --job-name=neo_step0
#SBATCH --output=logs/step0_%j.out
#SBATCH --error=logs/step0_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# Step 0: ESM-2 embeddings (run before step5; can parallelize with step1)

source "$(dirname "$0")/common.sh"

echo "=== Step 0: Embeddings ==="
python run_pipeline.py --step embeddings --input data/step5_input.tsv
echo "=== Step 0 complete ==="
