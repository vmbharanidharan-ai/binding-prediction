#!/bin/bash
# Fix ColabFold crash: AttributeError: module 'jax' has no attribute 'linear_util'
#
# ColabFold 1.5.5 pins dm-haiku==0.0.10 (Python 3.9). Newer haiku (0.0.14+) needs Py3.10+.
# JAX >= 0.4.24 removed jax.linear_util, which haiku 0.0.10 still uses.
#
# Fix: keep dm-haiku==0.0.10 and pin JAX/jaxlib 0.4.23 with CUDA 12 wheels.
#
# Usage (Longleaf login node):
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/repair_colabfold_jax_haiku.sh

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"
JAX_VERSION="${JAX_VERSION:-0.4.23}"
JAX_CUDA_INDEX="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

if [[ ! -f "${ALPHAFOLD_ENV}/bin/activate" ]]; then
    echo "ERROR: alphafoldenv not found at ${ALPHAFOLD_ENV}"
    exit 1
fi

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"

echo "Python: $(python -V)"
echo ""
echo "=== Before repair ==="
pip show jax jaxlib dm-haiku 2>/dev/null | grep -E '^Name|^Version' || true

echo ""
echo "=== Pin dm-haiku for ColabFold 1.5.5 (Python 3.9) ==="
pip install 'dm-haiku==0.0.10'

echo ""
echo "=== Pin JAX ${JAX_VERSION} with CUDA 12 (compatible with haiku 0.0.10) ==="
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true
pip install -U \
    "jax[cuda12_pip]==${JAX_VERSION}" \
    -f "${JAX_CUDA_INDEX}"

echo ""
echo "=== After repair ==="
pip show jax jaxlib dm-haiku | grep -E '^Name|^Version'

echo ""
echo "=== Verify imports (CPU ok on login node) ==="
python -c "
import haiku
import jax
print('dm-haiku:', haiku.__version__)
print('jax:', jax.__version__)
print('haiku import OK')
"

echo ""
echo "=== GPU check (optional; run on GPU node) ==="
echo "  srun --partition=volta-gpu --qos=gpu_access --gres=gpu:1 --mem=8G --time=00:05:00 \\"
echo "    bash scripts/verify_colabfold_gpu.sh"
echo ""
echo "Done. Clear colabfold_status.tsv and resubmit Step 1."
