#!/bin/bash
#SBATCH --job-name=neo_step3_5
#SBATCH --output=logs/step3_5_%j.out
#SBATCH --error=logs/step3_5_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=4

# Step 3.5: ProteinMPNN sequence design on RFdiffusion backbones
# Input:  work/step3_binders/binder_designs.tsv
# Output: work/step3_5_sequences/designed_binders.tsv

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

echo "=== Step 3.5: ProteinMPNN ==="
echo "PROTEINMPNN_ROOT: ${PROTEINMPNN_ROOT:-${PROJECT_ROOT}/ProteinMPNN}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python run_pipeline.py --step step3_5
echo "=== Step 3.5 complete ==="
