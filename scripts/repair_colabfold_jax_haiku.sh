#!/bin/bash
# Repair alphafoldenv JAX + haiku for ColabFold on Longleaf (Python 3.9, CUDA 12).
#
# Working stack: jax==0.4.27, jaxlib+cuda12, dm-haiku==0.0.12, chex==0.1.90
#
# Usage:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/repair_colabfold_jax_haiku.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/colabfold_versions.env
source "${SCRIPT_DIR}/colabfold_versions.env"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"

if [[ ! -f "${ALPHAFOLD_ENV}/bin/activate" ]]; then
    echo "ERROR: alphafoldenv not found at ${ALPHAFOLD_ENV}"
    echo "Run: bash scripts/setup_colabfold_longleaf.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"

echo "Python: $(python -V)"
echo "Target: jax==${COLABFOLD_JAX_VERSION}, dm-haiku==${COLABFOLD_HAIKU_VERSION}"
echo ""
echo "=== Before ==="
pip show jax jaxlib dm-haiku chex 2>/dev/null | grep -E '^Name|^Version' || true

echo ""
echo "=== Pin haiku + chex (Python 3.9 compatible) ==="
pip install "dm-haiku==${COLABFOLD_HAIKU_VERSION}" "chex==${COLABFOLD_CHEX_VERSION}"

echo ""
echo "=== Pin JAX ${COLABFOLD_JAX_VERSION} with CUDA 12 wheels ==="
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true
pip install -U \
    "jax[cuda12_pip]==${COLABFOLD_JAX_VERSION}" \
    -f "${JAX_CUDA_RELEASES_URL}"

echo ""
echo "=== After ==="
pip show jax jaxlib dm-haiku chex | grep -E '^Name|^Version'

echo ""
echo "=== Verify imports ==="
python -c "
import haiku
import jax
from colabfold.alphafold import models
print('dm-haiku:', haiku.__version__)
print('jax:', jax.__version__)
print('colabfold models import OK')
"

echo ""
echo "Done. GPU check: srun ... bash scripts/verify_colabfold_gpu.sh"
