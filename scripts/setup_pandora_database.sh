#!/bin/bash
# Submit PANDORA database download (required once before PMGen Step 1).
#
# Official upstream: pandora-fetch  →  ~/PANDORA_databases/default
# Longleaf default:   $PROJECT_ROOT/PANDORA_databases/default  (on /work)
#
# Usage:
#   export PROJECT_ROOT=/work/users/v/m/vmbharan/.../minibinder_prediction
#   cd $PROJECT_ROOT/binding-prediction
#   bash scripts/setup_pandora_database.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/pandora_db_path.sh
source "$SCRIPT_DIR/pandora_db_path.sh"

mkdir -p logs

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PANDORA_DB_ROOT=$PANDORA_DB_ROOT"
echo ""
echo "Submitting PANDORA fetch job (pre-built DB from zenodo, ~5–20 min)..."

JOB_ID=$(sbatch --export=ALL slurm/setup_pandora_database.sbatch | awk '{print $NF}')
echo "Submitted: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO_ROOT/logs/pandora_fetch_${JOB_ID}.out"
echo ""
echo "Verify after job completes:"
echo "  conda activate PMGen"
echo "  python -c \"import PANDORA; from PANDORA.Database import Database; Database.load(); print('OK', PANDORA.PANDORA_data)\""
echo ""
echo "If fetch fails, rebuild from IMGT (slow):"
echo "  sbatch --export=ALL slurm/build_pandora_database.sbatch"
