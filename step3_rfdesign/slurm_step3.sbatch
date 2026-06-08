#!/bin/bash
#SBATCH --job-name=neo_step3
#SBATCH --output=logs/step3_%j.out
#SBATCH --error=logs/step3_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Neo Binder Pipeline — Step 3: RFdiffusion binder design
# Requires GPU + RFdiffusion installation

set -euo pipefail

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neo_binder

# RFdiffusion may need a separate env — uncomment if needed:
# conda activate SE3nv

export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-/work/users/$USER/neo_binder}"
export RFDIFFUSION_WEIGHTS="${RFDIFFUSION_WEIGHTS:-/work/users/$USER/rfdiffusion_weights}"
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Step 3: RFdiffusion Binder Design ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python step3_rfdesign/build_contig_map.py \
    --input "$NEO_BINDER_WORK_ROOT/step2_scored/ranked_structures.tsv" \
    --output-dir "$NEO_BINDER_WORK_ROOT/step3_binders/contigs"

python step3_rfdesign/run_rfdiffusion.py \
    --manifest "$NEO_BINDER_WORK_ROOT/step3_binders/contigs/contig_manifest.tsv" \
    --output-dir "$NEO_BINDER_WORK_ROOT/step3_binders"

echo "=== Step 3 complete ==="
