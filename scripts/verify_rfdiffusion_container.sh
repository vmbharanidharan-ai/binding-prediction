#!/bin/bash
# Verify RFdiffusion Apptainer container on a GPU node (run before Step 3).
#
# Usage:
#   export PROJECT_ROOT=/work/users/$USER/your_dataset/minibinder_prediction
#   bash scripts/verify_rfdiffusion_container.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
CONTAINER="${RFDIFFUSION_CONTAINER:-${PROJECT_ROOT}/rfdiffusion.sif}"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "Build with: bash scripts/build_rfdiffusion_container.sh"
    exit 1
fi

export RFDIFFUSION_CONTAINER="$CONTAINER"
export PROJECT_ROOT

echo "=== RFdiffusion container preflight ==="
echo "Container: $CONTAINER"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
echo ""

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

RUNNER=""
if command -v apptainer &>/dev/null; then
    RUNNER=apptainer
elif command -v singularity &>/dev/null; then
    RUNNER=singularity
else
    echo "ERROR: apptainer/singularity not found"
    exit 1
fi

"$RUNNER" exec --nv --bind /work:/work --bind /nas:/nas "$CONTAINER" python - <<'PYEOF'
import torch
import dgl
import rfdiffusion

print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")
print(f"dgl {dgl.__version__}")
print(f"rfdiffusion {rfdiffusion.__file__}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

g = dgl.graph(([0, 1], [1, 2])).to("cuda")
src, dst = g.edges()
print(f"DGL CUDA graph.edges() OK ({len(src)} edges)")

major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(torch.cuda.current_device())
print(f"GPU {name} compute capability {major}.{minor}")
if major > 8 or (major == 8 and minor >= 9):
    raise SystemExit(
        f"GPU {name} (sm_{major}{minor}) incompatible with torch 1.9.1+cu111; "
        "use volta-gpu (V100) or a100-gpu for Step 3"
    )

from e3nn import o3

x = torch.randn(16, 3, device="cuda")
_ = o3.spherical_harmonics(2, x, normalize=True)
print("e3nn spherical_harmonics NVRTC OK")
PYEOF

RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
if [[ -d "${RFDIFFUSION_ROOT}/models" ]]; then
    shopt -s nullglob
    weights=("${RFDIFFUSION_ROOT}/models"/*.pt)
    if [[ ${#weights[@]} -eq 0 ]]; then
        echo "WARN: No *.pt weights in ${RFDIFFUSION_ROOT}/models"
    else
        echo "Weights:  ${#weights[@]} checkpoint(s) in ${RFDIFFUSION_ROOT}/models"
    fi
else
    echo "WARN: Weights dir missing: ${RFDIFFUSION_ROOT}/models"
fi

echo ""
echo "=== Container preflight passed ==="
