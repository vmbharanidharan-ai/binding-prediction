#!/bin/bash
# Verify RFdiffusion install (run on a GPU compute node).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-$PROJECT_ROOT/RFdiffusion}"

source "$REPO_ROOT/scripts/rfdiffusion_env.sh"

echo "=== RFdiffusion verify ==="
echo "Host: $(hostname)"
echo "RFDIFFUSION_ROOT: $RFDIFFUSION_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

python -c "import torch, dgl, rfdiffusion; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('dgl', dgl.__version__)"

for w in Base_ckpt.pt Complex_base_ckpt.pt; do
    test -f "$RFDIFFUSION_ROOT/models/$w" || {
        echo "MISSING weight: $RFDIFFUSION_ROOT/models/$w"
        exit 1
    }
done

echo "RFdiffusion imports + key weights OK"
echo "=== RFdiffusion verify complete ==="
