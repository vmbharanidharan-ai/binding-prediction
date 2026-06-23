#!/bin/bash
# Print which Python each pipeline env resolves to (run on login or GPU node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Pipeline environment isolation check ==="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo ""

check_conda() {
    local name="$1"
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        echo "--- $name ---"
        conda run -n "$name" python -c "import sys; print(sys.executable)"
        conda run -n "$name" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null \
            || echo "(no torch — expected for neo_binder)"
    else
        echo "--- $name --- MISSING"
    fi
    echo ""
}

check_conda neo_binder
check_conda proteinmpnn

if [[ -f "$PROJECT_ROOT/alphafoldenv/bin/python" ]]; then
    echo "--- alphafoldenv (venv) ---"
    "$PROJECT_ROOT/alphafoldenv/bin/python" -c "import sys; print(sys.executable)"
    "$PROJECT_ROOT/alphafoldenv/bin/python" -c "import jax; print('jax', jax.__version__)" 2>/dev/null \
        || echo "(jax not installed)"
else
    echo "--- alphafoldenv --- MISSING"
fi
echo ""

CONTAINER="${RFDIFFUSION_CONTAINER:-${PROJECT_ROOT}/rfdiffusion.sif}"
echo "--- RFdiffusion container ---"
if [[ -f "$CONTAINER" ]]; then
    ls -lh "$CONTAINER"
    if command -v apptainer &>/dev/null || command -v singularity &>/dev/null; then
        module load apptainer 2>/dev/null || true
        RUNNER="$(command -v apptainer || command -v singularity)"
        "$RUNNER" exec "$CONTAINER" python -c "import rfdiffusion; print('rfdiffusion', rfdiffusion.__file__)"
    else
        echo "(apptainer not loaded — skip import test)"
    fi
else
    echo "MISSING: $CONTAINER"
    echo "Build: sbatch slurm/build_rfdiffusion_container.sbatch"
fi

echo ""
echo "=== Done ==="
