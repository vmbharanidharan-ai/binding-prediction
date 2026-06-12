#!/bin/bash
# Login-node PMGen sanity checks (no GPU required).
# CSB-PANDORA pip package imports as: import PANDORA  (uppercase, not pandora)

set -euo pipefail

PMGEN_ROOT="${PMGEN_ROOT:-${PROJECT_ROOT:-}/PMGen}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PMGen

echo "=== PMGen login-node verify ==="
echo "PMGEN_ROOT: $PMGEN_ROOT"

python -c "import PANDORA, torch; print('PANDORA OK'); print('torch', torch.__version__)"
command -v pandora-fetch >/dev/null && echo "pandora-fetch CLI OK" || echo "WARN: pandora-fetch not on PATH"

test -d "$PMGEN_ROOT/AFfine/af_params" && echo "AFfine weights OK" || echo "WARN: AFfine weights missing"
test -f "$PMGEN_ROOT/run_PMGen.py" && echo "run_PMGen.py OK" || echo "WARN: run_PMGen.py not found"

echo "=== Login verify complete (GPU/JAX not tested here) ==="
