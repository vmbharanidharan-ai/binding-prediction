#!/bin/bash
#SBATCH --job-name=neo_step3
#SBATCH --output=logs/step3_%j.out
#SBATCH --error=logs/step3_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta-gpu,a100-gpu,gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=8

# Step 3: RFdiffusion binder design (GPU, Apptainer container)
# Output: work/step3_binders/binder_designs.tsv
#
# Orchestration: neo_binder (run_pipeline.py)
# RFdiffusion inference: Apptainer (scripts/run_rfdiffusion_container.sh)

# shellcheck source=slurm/common_paths.sh
source "${SLURM_SUBMIT_DIR}/slurm/common_paths.sh"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

echo "=========================================="
echo "Step 3: RFdiffusion Binder Design"
echo "=========================================="
echo "Executor:      neo_binder (Python)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Pipeline dir:  $(pwd)"
echo "Pipeline env:  $(which python)"
echo ""

export RFDIFFUSION_CONTAINER="${RFDIFFUSION_CONTAINER:-${PROJECT_ROOT}/rfdiffusion.sif}"
echo "Container:     $RFDIFFUSION_CONTAINER"
if [[ ! -f "$RFDIFFUSION_CONTAINER" ]]; then
    echo "ERROR: Container not found at $RFDIFFUSION_CONTAINER"
    echo "Build with: sbatch slurm/build_rfdiffusion_container.sbatch"
    exit 1
fi
echo "Container OK"
echo ""

bash "${SLURM_SUBMIT_DIR}/scripts/verify_rfdiffusion_container.sh"

echo "Input:         $INPUT_TSV"
python run_pipeline.py --step step3 --input "$INPUT_TSV"
echo "=== Step 3 complete ==="
