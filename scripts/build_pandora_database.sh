#!/bin/bash
# Build PANDORA database from scratch (slow — only if pandora-fetch fails).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/pandora_db_path.sh"

N_JOBS="${PANDORA_DB_JOBS:-4}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PMGEN_ENV:-PMGen}"

echo "=== PANDORA database construct (from IMGT — slow) ==="
echo "Host: $(hostname)"
echo "PANDORA_DB_ROOT: $PANDORA_DB_ROOT"
echo "Parallel jobs: $N_JOBS"

python <<PY
from PANDORA.Database import Database

db_path = "${PANDORA_DB_ROOT}"
save = f"{db_path}/database/PANDORA_database.pkl"
Database.create_db_folders(db_path)
db = Database.Database()
db.construct_database(save=save, data_dir=db_path, n_jobs=${N_JOBS})
print("construct_database OK:", save)
PY

echo "=== PANDORA construct complete ==="
