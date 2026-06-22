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
check_conda SE3nv
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

echo "--- After sourcing rfdiffusion_env.sh (SE3nv runtime) ---"
export PROJECT_ROOT
# shellcheck source=scripts/rfdiffusion_env.sh
source "$SCRIPT_DIR/rfdiffusion_env.sh"
echo "which python: $(which python)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
python -c "import dgl; print('dgl', dgl.__version__, dgl.__file__)"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python "$SCRIPT_DIR/verify_dgl_cuda.py"
else
    echo "(skip DGL CUDA test — no GPU on this node)"
fi

echo "=== Done ==="
