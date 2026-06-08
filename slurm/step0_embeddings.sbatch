#!/bin/bash
#SBATCH --job-name=neo_step0
#SBATCH --output=logs/step0_%j.out
#SBATCH --error=logs/step0_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-gpu,l40-gpu,v100-gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=4

# Step 0: ESM-2 embeddings (run before step5; can parallelize with step1)

source "$(dirname "$0")/common.sh"

echo "=== Step 0: Embeddings ==="
python run_pipeline.py --step embeddings --input "$INPUT_TSV"
echo "=== Step 0 complete ==="
