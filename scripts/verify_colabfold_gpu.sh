#!/bin/bash
# Quick GPU check for ColabFold alphafoldenv (run on a GPU node via srun or sbatch).
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"

# shellcheck disable=SC1091
source "$(dirname "$0")/colabfold_env.sh"

echo "Python:          $(which python)"
echo "colabfold_batch: $(which colabfold_batch)"
echo "COLABFOLD_DATA:  ${COLABFOLD_DATA_DIR:-<default cache>}"

python -c "
import haiku
import jax
from colabfold.alphafold import models
print('haiku:', haiku.__version__, '| jax:', jax.__version__)
print('jax devices:', jax.devices())
print('colabfold model loader OK')
"
