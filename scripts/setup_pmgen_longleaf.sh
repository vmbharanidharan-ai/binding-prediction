#!/bin/bash
# One-time PMGen install for Longleaf (Step 1 structure backend).
#
# Usage:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   bash scripts/setup_pmgen_longleaf.sh
#
# Requires: conda/mamba, git, GPU node optional for install (params download is CPU)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PMGEN_ROOT="${PMGEN_ROOT:-$PROJECT_ROOT/PMGen}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PMGEN_ROOT=$PMGEN_ROOT"

if [[ ! -d "$PMGEN_ROOT/.git" ]]; then
    git clone https://github.com/soedinglab/PMGen.git "$PMGEN_ROOT"
fi

cd "$PMGEN_ROOT"
echo "Installing PMGen (GPU, no Modeller — uses --initial_guess mode)."
bash -l install.sh --no_modeller

conda activate PMGen
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo ""
echo "PMGen installed. Add to your Longleaf session:"
echo "  export PROJECT_ROOT=$PROJECT_ROOT"
echo "  export PMGEN_ROOT=$PMGEN_ROOT"
echo "  conda activate neo_binder"
echo ""
echo "config/config.yaml should have: step1.backend: pmgen"
