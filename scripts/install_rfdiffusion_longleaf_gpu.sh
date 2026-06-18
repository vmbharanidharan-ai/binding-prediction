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
TORCH_OK=0
if conda env list | awk '{print $1}' | grep -qx "SE3nv"; then
    if conda run -n SE3nv python - <<'PY' 2>/dev/null
import torch
major = int(torch.__version__.split(".")[0])
minor = int(torch.__version__.split(".")[1].split("+")[0])
raise SystemExit(0 if major == 1 and minor <= 10 else 1)
PY
    then
        TORCH_OK=1
    fi
fi

if [[ $TORCH_OK -eq 1 ]] \
    && [[ -d models ]] && ls models/*.pt &>/dev/null \
    && conda run -n SE3nv python -c "import rfdiffusion" &>/dev/null; then
    echo "SE3nv + torch 1.x + weights + rfdiffusion module found — resume/verify mode."
    RESUME=1
    conda activate SE3nv
elif conda env list | awk '{print $1}' | grep -qx "SE3nv"; then
    if [[ $TORCH_OK -eq 0 ]]; then
        echo "SE3nv has incompatible torch (need 1.9.x). Run: bash scripts/repair_se3nv_torch.sh"
        echo "Attempting in-place PyTorch repair now..."
        conda activate SE3nv
        pip uninstall -y torch torchvision torchaudio dgl dglgo 2>/dev/null || true
        bash "$(dirname "$0")/install_se3nv_torch_pip.sh"
        bash "$(dirname "$0")/repair_se3nv_dgl.sh"
        RESUME=1
        conda activate SE3nv
    else
        echo "Removing partial SE3nv env..."
        conda env remove -n SE3nv -y || true
    fi
fi

if [[ $RESUME -eq 0 ]]; then
    echo "Step 1/6: Creating SE3nv env (python 3.9)..."
    conda create -n SE3nv python=3.9 pip -y --override-channels -c conda-forge
    conda activate SE3nv

    echo "Step 2/6: Installing PyTorch 1.9 + CUDA 11.1 (pip cu111; conda solve fails on py3.9)..."
    bash "$(dirname "$0")/install_se3nv_torch_pip.sh"

    echo "Step 3/6: cudatoolkit + DGL (cu111) + hydra..."
    conda install -y --override-channels -c conda-forge cudatoolkit=11.1.1
    if ! conda install -y -c dglteam -c conda-forge "dgl-cuda11.1=0.9.1post1"; then
        pip install --no-cache-dir "dgl==1.0.0" -f https://data.dgl.ai/wheels/cu111/repo.html
    fi
    pip install hydra-core pyrsistent "torchdata==0.9.0" --no-deps
    pip install "numpy==1.23.5" "scipy==1.10.1"

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
source "$(dirname "$0")/dgl_cuda_libpath.sh"
set +e
python <<'PY'
import traceback
try:
    import torch
    import dgl
    import rfdiffusion
    major = int(torch.__version__.split(".")[0])
    if major >= 2:
        raise RuntimeError(
            f"torch {torch.__version__} is incompatible with RFdiffusion; need 1.9.x. "
            "Run: bash scripts/repair_se3nv_torch.sh"
        )
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("dgl", dgl.__version__)
    print("rfdiffusion OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
PY
VERIFY_RC=$?
if [[ $VERIFY_RC -eq 0 ]]; then
    python "$(dirname "$0")/verify_dgl_cuda.py" || VERIFY_RC=$?
fi
set -e
export LD_LIBRARY_PATH="$_SAVED_LD"

if [[ $VERIFY_RC -ne 0 ]]; then
    echo "WARN: verify exited $VERIFY_RC — try on GPU: source scripts/rfdiffusion_env.sh"
    echo "      If DGL import failed: bash scripts/repair_se3nv_dgl.sh"
fi

echo "=== Longleaf RFdiffusion install complete ==="
