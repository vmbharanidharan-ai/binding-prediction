#!/bin/bash
# ProteinMPNN setup for Longleaf (login node).
#
# Usage:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/setup_proteinmpnn_longleaf.sh
#   bash scripts/install_proteinmpnn_longleaf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-$PROJECT_ROOT/ProteinMPNN}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROTEINMPNN_ROOT=$PROTEINMPNN_ROOT"

if [[ ! -d "$PROTEINMPNN_ROOT/.git" ]]; then
    git clone https://github.com/dauparas/ProteinMPNN.git "$PROTEINMPNN_ROOT"
else
    echo "ProteinMPNN repo already exists at $PROTEINMPNN_ROOT"
fi

export PROJECT_ROOT PROTEINMPNN_ROOT
bash "$REPO_ROOT/scripts/install_proteinmpnn_longleaf.sh"

echo ""
echo "Done. Step 3.5 will use:"
echo "  PROTEINMPNN_ROOT=$PROTEINMPNN_ROOT"
echo "  conda env: proteinmpnn"
echo ""
echo "Quick verify (GPU node recommended for cuda check):"
echo "  bash scripts/verify_proteinmpnn_gpu.sh"
