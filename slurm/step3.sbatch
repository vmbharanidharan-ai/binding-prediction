#!/bin/bash
#SBATCH --job-name=neo_step3
#SBATCH --output=logs/step3_%j.out
#SBATCH --error=logs/step3_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=8

# Step 3: RFdiffusion binder design (GPU)
# Output: work/step3_binders/binder_designs.tsv

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

echo "=== Step 3: RFdiffusion ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python run_pipeline.py --step step3 --input "$INPUT_TSV"
echo "=== Step 3 complete ==="
