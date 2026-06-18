#!/bin/bash
# Repair DGL/torchdata pins in an existing SE3nv env (no full reinstall).
#
# Fixes: ModuleNotFoundError: No module named 'torchdata.datapipes'
# Run on a GPU compute node after: conda activate SE3nv

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

echo "=== SE3nv DGL repair ==="
echo "Python: $(python -V)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"

pip install --upgrade pip

# torchdata 0.9.x still ships datapipes (required by DGL / RFdiffusion)
pip install "torchdata==0.9.0" "numpy<2" --force-reinstall

pip uninstall -y dgl dglgo 2>/dev/null || true

install_dgl() {
    echo "Trying: $*"
    pip install "$@"
}

# +cu111 suffix wheels are often unavailable; install 1.0.0 from the cu111 repo.
if install_dgl "dgl==1.0.0" -f https://data.dgl.ai/wheels/cu111/repo.html; then
    echo "Installed dgl 1.0.0 (cu111 repo)"
elif install_dgl "dgl==1.0.1" -f https://data.dgl.ai/wheels/cu111/repo.html; then
    echo "Installed dgl 1.0.1 (cu111 repo)"
elif install_dgl "dgl==1.1.2" -f https://data.dgl.ai/wheels/torch-1.9/cu111/repo.html; then
    echo "Installed dgl 1.1.2 (torch-1.9 cu111 repo)"
else
    echo "ERROR: Could not install a compatible DGL wheel."
    echo "Try a full reinstall: bash scripts/install_rfdiffusion_longleaf_gpu.sh"
    exit 1
fi

_SAVED_LD="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH
python <<'PY'
import torch
import dgl
import rfdiffusion
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("rfdiffusion OK")
PY
export LD_LIBRARY_PATH="$_SAVED_LD"

echo "=== SE3nv repair complete ==="
