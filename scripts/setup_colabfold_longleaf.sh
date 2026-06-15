#!/bin/bash
# Create or repair alphafoldenv for ColabFold Step 1/4 on UNC Longleaf.
#
# Usage (login node):
#   export PROJECT_ROOT=/work/users/<onyen>/.../minibinder_prediction
#   bash scripts/setup_colabfold_longleaf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/colabfold_versions.env
source "${SCRIPT_DIR}/colabfold_versions.env"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"
export COLABFOLD_DATA_DIR="${COLABFOLD_DATA_DIR:-$PROJECT_ROOT/colabfold_params}"

echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ALPHAFOLD_ENV:     $ALPHAFOLD_ENV"
echo "COLABFOLD_DATA_DIR: $COLABFOLD_DATA_DIR"
mkdir -p "$COLABFOLD_DATA_DIR"

if [[ ! -d "${ALPHAFOLD_ENV}" ]]; then
    echo "Creating alphafoldenv venv..."
    python3 -m venv "$ALPHAFOLD_ENV"
fi

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"
pip install --upgrade pip wheel

if ! python -c "import colabfold" 2>/dev/null; then
    echo "Installing ColabFold..."
    pip install "colabfold[alphafold]==1.5.5"
fi

bash "${SCRIPT_DIR}/repair_colabfold_jax_haiku.sh"

echo ""
echo "=== Download AlphaFold weights to /work (recommended once on login node) ==="
bash "${SCRIPT_DIR}/prefetch_colabfold_weights.sh"
echo "Export before sbatch:"
echo "  export PROJECT_ROOT=$PROJECT_ROOT"
echo "  export ALPHAFOLD_ENV=$ALPHAFOLD_ENV"
echo "  export COLABFOLD_BIN=$ALPHAFOLD_ENV/bin/colabfold_batch"
echo "  export COLABFOLD_DATA_DIR=$COLABFOLD_DATA_DIR"
