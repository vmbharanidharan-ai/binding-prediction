#!/bin/bash
#SBATCH --job-name=neo_step4
#SBATCH --output=logs/step4_%j.out
#SBATCH --error=logs/step4_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Step 4: Binder validation (ColabFold multimer, GPU)
# Output: work/step4_validated/binder_scores.tsv

source "$(dirname "$0")/common.sh"

echo "=== Step 4: Binder Validation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python run_pipeline.py --step step4 --input "$INPUT_TSV"
echo "=== Step 4 complete ==="
