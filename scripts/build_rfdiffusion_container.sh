#!/bin/bash
# Build RFdiffusion Apptainer image on a Longleaf GPU node (~40 min one-time).
#
# Usage (interactive GPU):
#   srun --partition=gpu --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
#   export PROJECT_ROOT=/work/users/$USER/your_dataset/minibinder_prediction
#   bash scripts/build_rfdiffusion_container.sh
#
# Or submit: sbatch slurm/build_rfdiffusion_container.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEF_FILE="$REPO_ROOT/containers/rfdiffusion.def"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
OUTPUT_SIF="${RFDIFFUSION_CONTAINER:-${PROJECT_ROOT}/rfdiffusion.sif}"
BUILD_DIR="${BUILD_DIR:-/tmp/rfdiffusion-build-$$}"

if [[ ! -f "$DEF_FILE" ]]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

RUNNER=""
if command -v apptainer &>/dev/null; then
    RUNNER=apptainer
elif command -v singularity &>/dev/null; then
    RUNNER=singularity
else
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    command -v apptainer &>/dev/null && RUNNER=apptainer
    command -v singularity &>/dev/null && RUNNER="${RUNNER:-singularity}"
fi
if [[ -z "$RUNNER" ]]; then
    echo "ERROR: apptainer/singularity not found"
    exit 1
fi

mkdir -p "$BUILD_DIR"
TMP_SIF="$BUILD_DIR/rfdiffusion.sif"

echo "=== Build RFdiffusion container ==="
echo "Runner:   $RUNNER"
echo "Def:      $DEF_FILE"
echo "Output:   $OUTPUT_SIF"
echo "Build in: $BUILD_DIR"
echo ""

"$RUNNER" build --nv "$TMP_SIF" "$DEF_FILE"

cp "$TMP_SIF" "$OUTPUT_SIF"
chmod 644 "$OUTPUT_SIF"
rm -rf "$BUILD_DIR"

echo ""
echo "=== Build complete ==="
ls -lh "$OUTPUT_SIF"
echo ""
echo "Next: bash scripts/verify_rfdiffusion_container.sh"
