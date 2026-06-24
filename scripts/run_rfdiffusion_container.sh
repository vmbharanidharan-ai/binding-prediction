#!/bin/bash
# Run RFdiffusion inference inside an Apptainer/Singularity container.
# Called by step3_rfdesign/run_rfdiffusion.py (neo_binder orchestration).
#
# Usage:
#   bash run_rfdiffusion_container.sh inference.input_pdb=... contigmap.contigs=[...] ...
#
# Environment:
#   RFDIFFUSION_CONTAINER  — path to .sif (default: $PROJECT_ROOT/rfdiffusion.sif)
#   RFDIFFUSION_ROOT       — host RFdiffusion clone (weights bind-mounted)
#   PROJECT_ROOT           — dataset project root on Longleaf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER="${RFDIFFUSION_CONTAINER:-}"
if [[ -z "$CONTAINER" ]]; then
    if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/rfdiffusion.sif" ]]; then
        CONTAINER="${PROJECT_ROOT}/rfdiffusion.sif"
    elif [[ -f "$REPO_ROOT/../rfdiffusion.sif" ]]; then
        CONTAINER="$(cd "$REPO_ROOT/.." && pwd)/rfdiffusion.sif"
    else
        echo "ERROR: Container not found. Set RFDIFFUSION_CONTAINER or build rfdiffusion.sif"
        echo "  bash scripts/build_rfdiffusion_container.sh"
        exit 1
    fi
fi

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hydra_override> [hydra_override ...]"
    exit 1
fi

# Apptainer on Longleaf; fall back to singularity.
RUNNER=""
if command -v apptainer &>/dev/null; then
    RUNNER=apptainer
elif command -v singularity &>/dev/null; then
    RUNNER=singularity
else
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    if command -v apptainer &>/dev/null; then
        RUNNER=apptainer
    elif command -v singularity &>/dev/null; then
        RUNNER=singularity
    else
        echo "ERROR: apptainer or singularity not found (try: module load apptainer)"
        exit 1
    fi
fi

RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT:-}/RFdiffusion}"
INFERENCE_SCRIPT="${RFDIFFUSION_INFERENCE_SCRIPT:-/opt/rfdiffusion/scripts/run_inference.py}"

BIND_ARGS=(
    --bind /work:/work
    --bind /nas:/nas
)
if [[ -d /proj ]]; then
    BIND_ARGS+=(--bind /proj:/proj)
fi
if [[ -d /scratch ]]; then
    BIND_ARGS+=(--bind /scratch:/scratch)
fi
if [[ -n "${RFDIFFUSION_ROOT}" && -d "${RFDIFFUSION_ROOT}/models" ]]; then
    BIND_ARGS+=(--bind "${RFDIFFUSION_ROOT}/models:/opt/rfdiffusion/models:ro")
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
SCHEDULES_DIR="${RFDIFFUSION_SCHEDULES_DIR:-${PROJECT_ROOT:-/tmp}/work/.rfdiffusion_schedules}"

echo "=========================================="
echo "RFdiffusion Inference (Apptainer)"
echo "=========================================="
echo "Container:  $CONTAINER"
echo "Runner:     $RUNNER"
echo "Script:     $INFERENCE_SCRIPT"
echo "GPU:        $GPU_NAME"
echo "Weights:    ${RFDIFFUSION_ROOT}/models → /opt/rfdiffusion/models"
echo "Schedules:  ${SCHEDULES_DIR}"
echo ""

echo "Pre-flight: DGL CUDA test..."
if ! "$RUNNER" exec --nv "${BIND_ARGS[@]}" "$CONTAINER" python - <<'PYEOF'
import torch
import dgl

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available in container")
g = dgl.graph(([0], [1])).to("cuda")
src, dst = g.edges()
if len(src) < 1:
    raise RuntimeError("DGL CUDA graph.edges() failed")
PYEOF
then
    echo "ERROR: Container DGL CUDA pre-flight failed"
    exit 1
fi
echo "DGL CUDA OK"
echo ""

echo "Pre-flight: GPU architecture + e3nn NVRTC (torch 1.9.1+cu111)..."
if ! "$RUNNER" exec --nv "${BIND_ARGS[@]}" "$CONTAINER" python - <<'PYEOF'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available in container")

major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(torch.cuda.current_device())
print(f"GPU {name} compute capability {major}.{minor}")

# torch 1.9.1+cu111 / CUDA 11.1 NVRTC cannot JIT for Ada (L40S sm_89) or Hopper+.
if major > 8 or (major == 8 and minor >= 9):
    raise RuntimeError(
        f"GPU {name} (sm_{major}{minor}) is incompatible with RFdiffusion container "
        f"(torch 1.9.1+cu111). Resubmit Step 3 on volta-gpu (V100) or a100-gpu only."
    )

from e3nn import o3

x = torch.randn(16, 3, device="cuda")
_ = o3.spherical_harmonics(2, x, normalize=True)
print("e3nn spherical_harmonics NVRTC OK")
PYEOF
then
    echo "ERROR: GPU architecture pre-flight failed (see message above)"
    exit 1
fi
echo "GPU architecture OK"
echo ""

# Hydra defaults write to outputs/ under --pwd (/opt/rfdiffusion), which is read-only.
HYDRA_RUN_DIR="${RFDIFFUSION_HYDRA_DIR:-/tmp/rfdiffusion_hydra}"
mkdir -p "$HYDRA_RUN_DIR"

# RFdiffusion caches IGSO3 schedules here; /opt/rfdiffusion/schedules is read-only in the image.
mkdir -p "$SCHEDULES_DIR"

DEFAULT_HYDRA_ARGS=(
    "hydra.run.dir=${HYDRA_RUN_DIR}"
    "hydra.job.chdir=false"
    "inference.schedule_directory_path=${SCHEDULES_DIR}"
)

echo "Running inference..."
set +e
"$RUNNER" exec --nv --env HYDRA_FULL_ERROR=1 "${BIND_ARGS[@]}" --pwd /opt/rfdiffusion "$CONTAINER" \
    python "$INFERENCE_SCRIPT" "${DEFAULT_HYDRA_ARGS[@]}" "$@"
INFER_RC=$?
set -e
if [[ $INFER_RC -ne 0 ]]; then
    echo ""
    echo "ERROR: RFdiffusion inference failed (exit code $INFER_RC)"
    echo "Hydra overrides: $*"
    exit "$INFER_RC"
fi
