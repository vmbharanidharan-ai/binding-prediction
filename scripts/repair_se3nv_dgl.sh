#!/bin/bash
# Repair DGL/torchdata pins in an existing SE3nv env (no full reinstall).

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv

pip install "dgl==1.0.0+cu111" -f https://data.dgl.ai/wheels/cu111/repo.html
pip install torchdata==0.9.0 "numpy<2"

_SAVED_LD="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH
python <<'PY'
import torch, dgl, rfdiffusion
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("rfdiffusion OK")
PY
export LD_LIBRARY_PATH="$_SAVED_LD"

echo "SE3nv repair complete."
