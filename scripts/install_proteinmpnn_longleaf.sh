#!/bin/bash
# Install ProteinMPNN conda env + verify weights (login or GPU node).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-$PROJECT_ROOT/ProteinMPNN}"
PROTEINMPNN_ENV="${PROTEINMPNN_ENV:-proteinmpnn}"

echo "=== ProteinMPNN install ==="
echo "Host: $(hostname)"
echo "PROTEINMPNN_ROOT: $PROTEINMPNN_ROOT"
echo "PROTEINMPNN_ENV: $PROTEINMPNN_ENV"

if [[ ! -f "$PROTEINMPNN_ROOT/protein_mpnn_run.py" ]]; then
    echo "ERROR: Clone ProteinMPNN first: bash scripts/setup_proteinmpnn_longleaf.sh"
    exit 1
fi

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$PROTEINMPNN_ENV"; then
    echo "Creating conda env $PROTEINMPNN_ENV (python 3.10)..."
    conda create -n "$PROTEINMPNN_ENV" python=3.10 pip -y --override-channels -c conda-forge
fi

conda activate "$PROTEINMPNN_ENV"

echo "Installing PyTorch + NumPy..."
if ! python -c "import torch" &>/dev/null; then
    conda install -y --override-channels \
        -c pytorch -c nvidia \
        pytorch pytorch-cuda=11.8 numpy \
        || pip install torch numpy
fi

echo "Checking model weights..."
WEIGHT_DIRS=(
    "$PROTEINMPNN_ROOT/vanilla_model_weights"
    "$PROTEINMPNN_ROOT/soluble_model_weights"
    "$PROTEINMPNN_ROOT/ca_model_weights"
)
for dir in "${WEIGHT_DIRS[@]}"; do
    mkdir -p "$dir"
done

download_weight() {
    local url="$1"
    local dest="$2"
    if [[ ! -f "$dest" ]]; then
        echo "Downloading $(basename "$dest")..."
        wget -q -O "$dest" "$url" || curl -L -o "$dest" "$url"
    fi
}

BASE="https://files.ipd.uw.edu/pub/ProteinMPNN/vanilla_model_weights"
download_weight "$BASE/v_48_002.pt" "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_002.pt"
download_weight "$BASE/v_48_010.pt" "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_010.pt"
download_weight "$BASE/v_48_020.pt" "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_020.pt"
download_weight "$BASE/v_48_030.pt" "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_030.pt"

SOL="https://files.ipd.uw.edu/pub/ProteinMPNN/soluble_model_weights"
download_weight "$SOL/v_48_010.pt" "$PROTEINMPNN_ROOT/soluble_model_weights/v_48_010.pt"
download_weight "$SOL/v_48_020.pt" "$PROTEINMPNN_ROOT/soluble_model_weights/v_48_020.pt"

CA="https://files.ipd.uw.edu/pub/ProteinMPNN/ca_model_weights"
download_weight "$CA/v_48_002.pt" "$PROTEINMPNN_ROOT/ca_model_weights/v_48_002.pt"
download_weight "$CA/v_48_010.pt" "$PROTEINMPNN_ROOT/ca_model_weights/v_48_010.pt"
download_weight "$CA/v_48_020.pt" "$PROTEINMPNN_ROOT/ca_model_weights/v_48_020.pt"

python <<'PY'
import sys
try:
    import torch
    import numpy
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("numpy", numpy.__version__)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
PY

test -f "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_020.pt"

echo "=== ProteinMPNN install complete ==="
echo "Verify on GPU: bash scripts/verify_proteinmpnn_gpu.sh"
