#!/bin/bash
# One-time RFdiffusion setup: clone weights repo + build Apptainer container.
#
# Usage:
#   export PROJECT_ROOT=/work/users/$USER/your_dataset/minibinder_prediction
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/setup_rfdiffusion_longleaf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"

echo "=== RFdiffusion setup (Apptainer) ==="
echo "PROJECT_ROOT:    $PROJECT_ROOT"
echo "RFDIFFUSION_ROOT: $RFDIFFUSION_ROOT"
echo ""

if [[ ! -d "$RFDIFFUSION_ROOT/.git" ]]; then
    echo "Cloning RFdiffusion..."
    git clone https://github.com/RosettaCommons/RFdiffusion.git "$RFDIFFUSION_ROOT"
else
    echo "RFdiffusion repo present: $RFDIFFUSION_ROOT"
fi

if [[ ! -d "${RFDIFFUSION_ROOT}/models" ]] || [[ -z "$(ls -A "${RFDIFFUSION_ROOT}/models"/*.pt 2>/dev/null)" ]]; then
    echo ""
    echo "Download model weights into ${RFDIFFUSION_ROOT}/models/"
    echo "  See: https://github.com/RosettaCommons/RFdiffusion#model-weights"
    echo "  Example: bash scripts/download_rfdiffusion_weights.sh"
fi

CONTAINER="${PROJECT_ROOT}/rfdiffusion.sif"
if [[ -f "$CONTAINER" ]]; then
    echo ""
    echo "Container already built: $CONTAINER"
    ls -lh "$CONTAINER"
else
    echo ""
    echo "Container not found. Build on a GPU node:"
    echo "  sbatch slurm/build_rfdiffusion_container.sbatch"
    echo "  # or interactively:"
    echo "  srun --partition=gpu --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash"
    echo "  bash scripts/build_rfdiffusion_container.sh"
fi

echo ""
echo "After build, verify on GPU:"
echo "  bash scripts/verify_rfdiffusion_container.sh"
echo ""
echo "=== Setup instructions complete ==="
