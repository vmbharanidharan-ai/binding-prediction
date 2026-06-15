#!/bin/bash
#SBATCH --job-name=neo_step1
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=8

# Step 1: Peptide–HLA structure generation (ColabFold)
# Output: work/step1_structures/ and work/step2_scored/parsed_structures.tsv

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

echo "=== Step 1: Structure Generation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python run_pipeline.py --step step1 --input "$INPUT_TSV"
echo "=== Step 1 complete ==="
