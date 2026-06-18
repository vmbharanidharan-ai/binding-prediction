#!/bin/bash
# Verify ProteinMPNN install (run on a GPU compute node).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
export PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-$PROJECT_ROOT/ProteinMPNN}"

source "$REPO_ROOT/scripts/proteinmpnn_env.sh"

echo "=== ProteinMPNN verify ==="
echo "Host: $(hostname)"
echo "PROTEINMPNN_ROOT: $PROTEINMPNN_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
test -f "$PROTEINMPNN_ROOT/protein_mpnn_run.py"
test -f "$PROTEINMPNN_ROOT/vanilla_model_weights/v_48_020.pt"

# Smoke test on bundled example if present
EXAMPLE_PDB="$PROTEINMPNN_ROOT/inputs/1QYS.pdb"
OUT="/tmp/mpnn_verify_$$"
if [[ -f "$EXAMPLE_PDB" ]]; then
    python "$PROTEINMPNN_ROOT/protein_mpnn_run.py" \
        --pdb_path "$EXAMPLE_PDB" \
        --pdb_path_chains "A" \
        --out_folder "$OUT" \
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --model_name v_48_020 \
        --path_to_model_weights "$PROTEINMPNN_ROOT/vanilla_model_weights" \
        --seed 37
    ls "$OUT"/seqs/*.fa
    echo "ProteinMPNN smoke test OK"
    rm -rf "$OUT"
else
    echo "Skip smoke test (example PDB not found); imports + weights OK"
fi

echo "=== ProteinMPNN verify complete ==="
