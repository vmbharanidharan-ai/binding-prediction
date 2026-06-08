#!/bin/bash
#SBATCH --job-name=neo_step1
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Neo Binder Pipeline — Step 1: Peptide–HLA structure generation
# UNC Longleaf HPC

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
mkdir -p logs "$NEO_BINDER_WORK_ROOT"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Step 1: Structure Generation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Work root: $NEO_BINDER_WORK_ROOT"

python step1_structure_generation/generate_inputs.py \
    --input data/step5_input.tsv \
    --output-dir "$NEO_BINDER_WORK_ROOT/inputs/step1"

python step1_structure_generation/run_colabfold.py \
    --manifest "$NEO_BINDER_WORK_ROOT/inputs/step1/input_manifest.tsv" \
    --output-dir "$NEO_BINDER_WORK_ROOT/step1_structures"

python step1_structure_generation/parse_outputs.py \
    --structure-dir "$NEO_BINDER_WORK_ROOT/step1_structures" \
    --manifest "$NEO_BINDER_WORK_ROOT/inputs/step1/input_manifest.tsv" \
    --output "$NEO_BINDER_WORK_ROOT/step2_scored/parsed_structures.tsv"

echo "=== Step 1 complete ==="
