#!/bin/bash
# Initialize a new neo_binder project directory on Longleaf.
#
# Usage:
#   bash scripts/init_project.sh --new-root /work/users/$USER/cohort_2/minibinder_prediction
#   bash scripts/init_project.sh --new-root /path/new --old-root /path/old
#
# Creates work/, clones binding-prediction (if needed), symlinks heavy assets,
# and writes $NEW_ROOT/.env for path exports.

set -euo pipefail

NEW_ROOT=""
OLD_ROOT=""
GIT_BRANCH="${GIT_BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/vmbharanidharan-ai/binding-prediction.git}"

usage() {
    cat <<'EOF'
Usage: bash scripts/init_project.sh --new-root /path/to/new/project [options]

Options:
  --new-root PATH   Project root (required). Creates work/ and binding-prediction/.
  --old-root PATH   Symlink heavy assets from an existing installation.
  --branch NAME     Git branch to checkout after clone (default: main).
  -h, --help        Show this help.

Heavy assets symlinked when --old-root is set:
  rfdiffusion.sif, RFdiffusion, ProteinMPNN, alphafoldenv, colabfold_params, .cache

After setup:
  source $NEW_ROOT/.env
  cd $PROJECT_ROOT/binding-prediction
  ./slurm/run_pair.sh -p PEPTIDE -a HLA_A0201 --gene GENE --step 1 --make-only
  ./slurm/submit_step.sh 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --new-root) NEW_ROOT="$2"; shift 2 ;;
        --old-root) OLD_ROOT="$2"; shift 2 ;;
        --branch)   GIT_BRANCH="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$NEW_ROOT" ]]; then
    usage >&2
    exit 1
fi

NEW_ROOT="$(cd "$NEW_ROOT" 2>/dev/null && pwd || echo "$NEW_ROOT")"
mkdir -p "$NEW_ROOT/work"

echo "=== Initializing neo_binder project ==="
echo "New root:  $NEW_ROOT"
echo "Work dir:  $NEW_ROOT/work"
[[ -n "$OLD_ROOT" ]] && echo "Old root:  $OLD_ROOT"

REPO_DIR="$NEW_ROOT/binding-prediction"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Cloning binding-prediction..."
    if [[ -d "$REPO_DIR" ]]; then
        if [[ -n "$(ls -A "$REPO_DIR" 2>/dev/null)" ]]; then
            echo "ERROR: $REPO_DIR exists but is not a git repo." >&2
            exit 1
        fi
        rmdir "$REPO_DIR"
    fi
    git clone "$REPO_URL" "$REPO_DIR"
    git -C "$REPO_DIR" checkout "$GIT_BRANCH"
else
    echo "binding-prediction already present — skipping clone"
fi

ASSETS=(rfdiffusion.sif RFdiffusion ProteinMPNN alphafoldenv colabfold_params .cache)

if [[ -n "$OLD_ROOT" ]]; then
    OLD_ROOT="$(cd "$OLD_ROOT" 2>/dev/null && pwd || echo "$OLD_ROOT")"
    if [[ ! -d "$OLD_ROOT" ]]; then
        echo "WARNING: --old-root not found: $OLD_ROOT" >&2
    else
        echo "Symlinking heavy assets from $OLD_ROOT..."
        for asset in "${ASSETS[@]}"; do
            src="$OLD_ROOT/$asset"
            dst="$NEW_ROOT/$asset"
            if [[ -e "$dst" ]]; then
                echo "  · $asset (already exists)"
            elif [[ -e "$src" ]]; then
                ln -s "$src" "$dst"
                echo "  ✓ $asset"
            else
                echo "  ⚠ $asset not found in old root — skipping"
            fi
        done
    fi
else
    echo "No --old-root provided; install or symlink heavy assets under $NEW_ROOT"
    echo "  See README.md → Quick start (Longleaf) → one-time external tool setup"
fi

cat > "$NEW_ROOT/.env" <<'EOF'
# Source before submitting jobs: source $PROJECT_ROOT/../.env  (from binding-prediction/)
# or: source /work/users/$USER/.../minibinder_prediction/.env
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NEO_BINDER_WORK_ROOT="$PROJECT_ROOT/work"
export INPUT_TSV="${INPUT_TSV:-$PROJECT_ROOT/binding-prediction/data/step5_input.tsv}"
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source $NEW_ROOT/.env"
echo "  cd \$PROJECT_ROOT/binding-prediction"
echo "  ./slurm/run_pair.sh -p PEPTIDE -a HLA_A0201 --gene GENE --step 1 --make-only"
echo "  ./slurm/submit_step.sh 0   # then 1, 2, 3, 3.5, 4, 5"
echo ""
echo "One-time per user (if not done): conda env create -f environment.yml -n neo_binder"
