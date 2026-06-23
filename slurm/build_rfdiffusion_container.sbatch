#!/bin/bash
#SBATCH --job-name=rfdiff_build
#SBATCH --output=logs/rfdiffusion_container_build_%j.out
#SBATCH --error=logs/rfdiffusion_container_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=4

# One-time RFdiffusion Apptainer image build on a GPU node.

# shellcheck source=slurm/common_paths.sh
source "${SLURM_SUBMIT_DIR}/slurm/common_paths.sh"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

bash "${SLURM_SUBMIT_DIR}/scripts/build_rfdiffusion_container.sh"
bash "${SLURM_SUBMIT_DIR}/scripts/verify_rfdiffusion_container.sh"
