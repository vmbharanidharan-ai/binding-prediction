#!/bin/bash
# Fix ColabFold crash: AttributeError: module 'jax' has no attribute 'linear_util'
#
# Cause: JAX >= 0.4.24 removed jax.linear_util; old dm-haiku in alphafoldenv needs >= 0.0.12.
# Run on Longleaf login node (or GPU node) after activating alphafoldenv.
#
# Usage:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/repair_colabfold_jax_haiku.sh

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"

if [[ ! -f "${ALPHAFOLD_ENV}/bin/activate" ]]; then
    echo "ERROR: alphafoldenv not found at ${ALPHAFOLD_ENV}"
    exit 1
fi

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"
module load cuda 2>/dev/null || true

echo "=== Before repair ==="
pip show jax jaxlib dm-haiku chex 2>/dev/null | grep -E '^Name|^Version' || true

echo ""
echo "=== Upgrading dm-haiku (and chex) for JAX >= 0.4.24 ==="
pip install --upgrade 'dm-haiku>=0.0.12' 'chex>=0.1.86'

echo ""
echo "=== Verify imports ==="
python -c "
import haiku
import jax
print('dm-haiku:', haiku.__version__)
print('jax:', jax.__version__)
print('devices:', jax.devices())
print('haiku + jax OK')
"

echo ""
echo "=== Done. Re-run Step 1 after clearing failed colabfold_status.tsv ==="
