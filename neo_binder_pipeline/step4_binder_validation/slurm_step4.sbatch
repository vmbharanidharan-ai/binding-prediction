#!/bin/bash
#SBATCH --job-name=neo_step4
#SBATCH --output=logs/step4_%j.out
#SBATCH --error=logs/step4_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Neo Binder Pipeline — Step 4: Binder validation (AlphaFold-Multimer)

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Step 4: Binder Validation ==="

python step4_binder_validation/build_complexes.py \
    --binders "$NEO_BINDER_WORK_ROOT/step3_binders/binder_designs.tsv" \
    --contigs "$NEO_BINDER_WORK_ROOT/step3_binders/contigs/contig_manifest.tsv" \
    --output-dir "$NEO_BINDER_WORK_ROOT/step4_validated/complexes"

python step4_binder_validation/run_colabfold_multimer.py \
    --manifest "$NEO_BINDER_WORK_ROOT/step4_validated/complexes/complex_manifest.tsv" \
    --output-dir "$NEO_BINDER_WORK_ROOT/step4_validated/multimer"

python step4_binder_validation/binder_scoring.py \
    --multimer-dir "$NEO_BINDER_WORK_ROOT/step4_validated/multimer" \
    --manifest "$NEO_BINDER_WORK_ROOT/step4_validated/complexes/complex_manifest.tsv" \
    --output "$NEO_BINDER_WORK_ROOT/step4_validated/binder_scores.tsv"

echo "=== Step 4 complete ==="
