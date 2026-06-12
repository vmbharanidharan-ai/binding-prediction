#!/bin/bash
# Download pre-built PANDORA database (official: pandora-fetch / zenodo).
# Prefer this over Database.construct_database() (30–60+ min IMGT rebuild).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/pandora_db_path.sh
source "$SCRIPT_DIR/pandora_db_path.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PMGEN_ENV:-PMGen}"

echo "=== PANDORA database fetch ==="
echo "Host: $(hostname)"
echo "PANDORA_DB_ROOT: $PANDORA_DB_ROOT"
echo "PANDORA package: $(python -c 'import PANDORA; print(PANDORA.PANDORA_path)')"

mkdir -p "$(dirname "$PANDORA_DB_ROOT")"

# Official CLI: pandora-fetch -d <path>  (downloads zenodo tarball)
pandora-fetch -d "$PANDORA_DB_ROOT"

python <<PY
import os
import PANDORA
from PANDORA.Database import Database

pkl = os.path.join(PANDORA.PANDORA_data, "database", "PANDORA_database.pkl")
print("PANDORA_data:", PANDORA.PANDORA_data)
print("config.json data_folder:", PANDORA.PANDORA_data)
if not os.path.isfile(pkl):
    raise SystemExit(f"MISSING: {pkl}")
db = Database.load()
print("Database.load() OK — MHCI templates:", len(db.MHCI_data))
PY

echo "=== PANDORA database ready ==="
