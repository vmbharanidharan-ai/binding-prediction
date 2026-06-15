#!/bin/bash
# Download ColabFold AlphaFold weights to /work (run once on login node with network).
#
# Multimer v3 weights (~3.8 GB) must exist before GPU jobs. Uses COLABFOLD_DATA_DIR
# on /work, not ~/.cache/colabfold (home quota).
#
# Usage:
#   export PROJECT_ROOT=/work/users/.../minibinder_prediction
#   bash scripts/prefetch_colabfold_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-$PROJECT_ROOT/alphafoldenv}"

# shellcheck source=scripts/colabfold_work_paths.sh
source "${SCRIPT_DIR}/colabfold_work_paths.sh"

# shellcheck disable=SC1091
source "${ALPHAFOLD_ENV}/bin/activate"

echo "Downloading ColabFold weights to: $COLABFOLD_DATA_DIR"
echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"
python - <<'PY'
import os
from pathlib import Path

from colabfold.download import download_alphafold_params

data_dir = Path(os.environ["COLABFOLD_DATA_DIR"])
for model in ("alphafold2_ptm", "alphafold2_multimer_v3"):
    print(f"=== {model} ===")
    download_alphafold_params(model, data_dir)
print("Done.")
print("Params directory:", data_dir / "params")
PY

echo ""
echo "Weights ready. These are set automatically in slurm/common.sh:"
echo "  COLABFOLD_DATA_DIR=$COLABFOLD_DATA_DIR"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
