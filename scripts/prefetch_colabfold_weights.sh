#!/bin/bash
# Download ColabFold AlphaFold weights to /work (run once on login node with network).
#
# Multimer v3 weights (~3.8 GB) must exist before GPU jobs; compute nodes may
# fail or time out if weights are downloaded during sbatch.
#
# Usage:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/prefetch_colabfold_weights.sh

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"
export COLABFOLD_DATA_DIR="${COLABFOLD_DATA_DIR:-$PROJECT_ROOT/colabfold_data}"

mkdir -p "$COLABFOLD_DATA_DIR"

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"

echo "Downloading ColabFold weights to: $COLABFOLD_DATA_DIR"
python - <<'PY'
import os
from pathlib import Path

from colabfold.download import download_alphafold_params

data_dir = Path(os.environ["COLABFOLD_DATA_DIR"])
for model in ("alphafold2_ptm", "alphafold2_multimer_v3"):
    print(f"=== {model} ===")
    download_alphafold_params(model, data_dir)
print("Done.")
PY

echo ""
echo "Weights ready. Export for sbatch:"
echo "  export COLABFOLD_DATA_DIR=$COLABFOLD_DATA_DIR"
