#!/bin/bash
# Quick GPU check for ColabFold alphafoldenv (run on a GPU node via srun or sbatch).
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"

# shellcheck disable=SC1091
source "$(dirname "$0")/colabfold_env.sh"

echo "Python: $(which python)"
echo "colabfold_batch: $(which colabfold_batch)"
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import jaxlib; print('jaxlib:', jaxlib.__version__, jaxlib.__file__)"
