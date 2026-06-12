#!/bin/bash
# Longleaf-compatible RFdiffusion / SE3nv install (run on a GPU compute node).

set -euo pipefail

RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$(pwd)}"
cd "$RFDIFFUSION_ROOT"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Longleaf RFdiffusion install ==="
echo "Host: $(hostname)"
echo "RFDIFFUSION_ROOT: $RFDIFFUSION_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

RESUME=0
if conda env list | awk '{print $1}' | grep -qx "SE3nv" \
    && [[ -d models ]] && ls models/*.pt &>/dev/null \
    && conda run -n SE3nv python -c "import rfdiffusion" &>/dev/null; then
    echo "SE3nv + weights + rfdiffusion module found — resume/verify mode."
    RESUME=1
    conda activate SE3nv
elif conda env list | awk '{print $1}' | grep -qx "SE3nv"; then
    echo "Removing partial SE3nv env..."
    conda env remove -n SE3nv -y || true
fi

if [[ $RESUME -eq 0 ]]; then
    echo "Step 1/6: Creating SE3nv env (python 3.9)..."
    conda create -n SE3nv python=3.9 pip -y --override-channels -c conda-forge
    conda activate SE3nv

    echo "Step 2/6: Installing PyTorch 1.9 + CUDA 11.1 (avoid SE3nv.yml solve failures)..."
    if ! conda install -y --override-channels \
        -c pytorch -c conda-forge \
        pytorch=1.9.1 torchvision=0.10.1 torchaudio=0.9.1 cudatoolkit=11.1; then
        echo "WARN: conda pytorch install failed — trying pip cu111 wheels..."
        pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
            -f https://download.pytorch.org/whl/torch_stable.html
    fi

    echo "Step 3/6: Installing DGL (cu111 pip wheel) + hydra..."
    pip install --upgrade pip
    pip install "dgl==1.0.0+cu111" -f https://data.dgl.ai/wheels/cu111/repo.html
    pip install hydra-core pyrsistent torchdata==0.9.0 "numpy<2"

    echo "Step 4/6: Installing NVIDIA SE3Transformer..."
    cd env/SE3Transformer
    pip install --no-cache-dir -r requirements.txt
    python setup.py install
    cd "$RFDIFFUSION_ROOT"

    echo "Step 5/6: Installing RFdiffusion package..."
    pip install -e .
fi

echo "Step 6/6: Downloading model weights (if missing)..."
mkdir -p models
download_weight() {
    local url="$1"
    local dest="$2"
    if [[ ! -f "$dest" ]]; then
        wget -O "$dest" "$url"
    fi
}
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt" \
    models/Base_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt" \
    models/Complex_base_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt" \
    models/Complex_Fold_base_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt" \
    models/InpaintSeq_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt" \
    models/InpaintSeq_Fold_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt" \
    models/ActiveSite_ckpt.pt
download_weight "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt" \
    models/Base_epoch8_ckpt.pt

_SAVED_LD="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH
set +e
python <<'PY'
import traceback
try:
    import torch
    import dgl
    import rfdiffusion
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("dgl", dgl.__version__)
    print("rfdiffusion OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
PY
VERIFY_RC=$?
set -e
export LD_LIBRARY_PATH="$_SAVED_LD"

if [[ $VERIFY_RC -ne 0 ]]; then
    echo "WARN: verify exited $VERIFY_RC — try on GPU: source scripts/rfdiffusion_env.sh"
    echo "      If DGL import failed: pip install dgl==1.0.0+cu111 -f https://data.dgl.ai/wheels/cu111/repo.html"
fi

echo "=== Longleaf RFdiffusion install complete ==="
