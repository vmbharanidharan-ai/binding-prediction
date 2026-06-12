#!/bin/bash
# Run upstream PMGen install.sh on a GPU node (official release path).
#
# Equivalent to the README:
#   git clone https://github.com/soedinglab/PMGen.git && cd PMGen
#   bash -l install.sh --no_modeller
#   conda activate PMGen
#
# Use on Longleaf when PMGen.yml solves on your cluster. If env create fails
# (strict channel priority / no CUDA on login), fall back to:
#   bash scripts/install_pmgen_longleaf_gpu.sh

set -euo pipefail

PMGEN_ROOT="${PMGEN_ROOT:-$(pwd)}"
cd "$PMGEN_ROOT"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Official PMGen install.sh (GPU node) ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"

if [[ ! -f install.sh ]]; then
    echo "ERROR: install.sh not found — clone https://github.com/soedinglab/PMGen.git"
    exit 1
fi

# Non-interactive: skip Modeller (structure prediction with --initial_guess does not need it)
export KEY_MODELLER="${KEY_MODELLER:-NOMODELLERKEY}"

bash -l install.sh --no_modeller

conda activate PMGen
python -c "import PANDORA; print('PANDORA OK')"
test -f run_PMGen.py && echo "run_PMGen.py OK"

echo "=== Official PMGen install complete ==="
