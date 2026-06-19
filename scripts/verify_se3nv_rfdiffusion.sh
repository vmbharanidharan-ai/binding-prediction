#!/bin/bash
# Preflight SE3nv for RFdiffusion (run on a GPU node before Step 3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/rfdiffusion_env.sh"

echo "=== SE3nv RFdiffusion preflight ==="
echo "Python: $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"

python - <<'PY'
import sys

import numpy as np
import torch

v = torch.__version__
major = int(v.split(".")[0])
if major >= 2:
    print(f"ERROR: torch {v} incompatible (need 1.9.x). Run: bash scripts/repair_se3nv.sh")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False (need GPU node)")
    sys.exit(1)

np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
if np_ver >= (1, 26):
    print(f"ERROR: numpy {np.__version__} breaks torch.from_numpy with torch 1.9")
    print("Fix: pip install numpy==1.23.5")
    sys.exit(1)

try:
    _ = torch.from_numpy(np.zeros((2, 3), dtype=np.float32))
except TypeError as exc:
    print(f"ERROR: torch.from_numpy failed: {exc}")
    print("Fix: pip uninstall -y numpy scipy && pip install numpy==1.23.5 scipy==1.10.1")
    sys.exit(1)

try:
    import scipy
    import dgl
    import rfdiffusion  # noqa: F401
except ImportError as exc:
    print(f"ERROR: scipy/dgl import failed: {exc}")
    print("Fix: pip uninstall -y numpy scipy && pip install numpy==1.23.5 scipy==1.10.1")
    sys.exit(1)

dev = torch.device("cuda:0")
u = torch.tensor([0, 1], device=dev)
dst = torch.tensor([1, 2], device=dev)
g = dgl.graph((u, dst)).to(dev)
_ = g.num_edges()

print(f"OK: torch {torch.__version__}, numpy {np.__version__}, dgl {dgl.__version__}, rfdiffusion import")
PY

echo "=== SE3nv preflight passed ==="
