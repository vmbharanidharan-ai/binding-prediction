#!/bin/bash
# Install PyTorch after neo_binder conda env — uses Longleaf module cuda, not conda pytorch-cuda
set -euo pipefail
module load cuda 2>/dev/null || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
