#!/bin/bash
# Initialize a new neo_binder project directory on Longleaf.
#
# Usage:
#   bash scripts/init_project.sh --new-root /work/users/$USER/cohort_2/minibinder_prediction \
#     --old-root /work/users/$USER/cohort_1/minibinder_prediction
#
# Creates work/, clones binding-prediction, symlinks heavy assets, and writes:
#   $NEW_ROOT/.env          — all path exports (auto-loaded by SLURM)
#   $NEW_ROOT/activate.sh   — one command to enter the project shell

set -euo pipefail

NEW_ROOT=""
OLD_ROOT=""
GIT_BRANCH="${GIT_BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/vmbharanidharan-ai/binding-prediction.git}"
REGISTER_BASHRC=false
PROJECT_ALIAS=""

usage() {
    cat <<'EOF'
Usage: bash scripts/init_project.sh --new-root /path/to/new/project [options]

Options:
  --new-root PATH       Project root (required). Creates work/ and binding-prediction/.
  --old-root PATH       Symlink heavy assets from an existing installation.
  --branch NAME         Git branch to checkout after clone (default: main).
  --alias NAME          Short name for activate alias (e.g. srsf2 → source activate.sh)
  --register-bashrc     Append "source .../activate.sh" to ~/.bashrc.d/neo_binder_NAME.sh
  -h, --help            Show this help.

Writes $NEW_ROOT/.env with PROJECT_ROOT, tool paths, and default INPUT_TSV.
SLURM jobs auto-source ../.env when run from binding-prediction/ (no manual export needed).

After setup:
  source $NEW_ROOT/activate.sh
  python scripts/create_input.py
  ./slurm/submit_step.sh 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --new-root) NEW_ROOT="$2"; shift 2 ;;
        --old-root) OLD_ROOT="$2"; shift 2 ;;
        --branch)   GIT_BRANCH="$2"; shift 2 ;;
        --alias)    PROJECT_ALIAS="$2"; shift 2 ;;
        --register-bashrc) REGISTER_BASHRC=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$NEW_ROOT" ]]; then
    usage >&2
    exit 1
fi

NEW_ROOT="$(mkdir -p "$NEW_ROOT" && cd "$NEW_ROOT" && pwd)"
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
mkdir -p "$REPO_DIR/data/generated"

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

INPUT_DEFAULT="$REPO_DIR/data/generated/input.tsv"

# Semi-permanent cluster paths — auto-sourced by slurm/common_paths.sh via ../.env
{
    echo "# Neo binder project environment — source: source $NEW_ROOT/activate.sh"
    echo "export PROJECT_ROOT=\"$NEW_ROOT\""
    echo "export NEO_BINDER_WORK_ROOT=\"\$PROJECT_ROOT/work\""
    echo "export INPUT_TSV=\"\${INPUT_TSV:-$INPUT_DEFAULT}\""
    echo "export PMGEN_ROOT=\"\$PROJECT_ROOT/PMGen\""
    echo "export RFDIFFUSION_ROOT=\"\$PROJECT_ROOT/RFdiffusion\""
    echo "export RFDIFFUSION_CONTAINER=\"\$PROJECT_ROOT/rfdiffusion.sif\""
    echo "export PROTEINMPNN_ROOT=\"\$PROJECT_ROOT/ProteinMPNN\""
    echo "export ALPHAFOLD_ENV=\"\$PROJECT_ROOT/alphafoldenv\""
    echo "export COLABFOLD_BIN=\"\$ALPHAFOLD_ENV/bin/colabfold_batch\""
    echo "export COLABFOLD_DATA_DIR=\"\$PROJECT_ROOT/colabfold_params\""
    echo "export COLABFOLD_PARAMS_DIR=\"\$PROJECT_ROOT/colabfold_params\""
    echo "export XDG_CACHE_HOME=\"\$PROJECT_ROOT/.cache\""
    if [[ -n "$OLD_ROOT" ]]; then
        echo "export OLD_ROOT=\"$OLD_ROOT\""
    fi
} > "$NEW_ROOT/.env"

cat > "$NEW_ROOT/activate.sh" <<EOF
#!/bin/bash
# Enter this neo_binder project (paths + repo directory).
set -a
source "\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)/.env"
set +a
if [[ -f "\$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate neo_binder 2>/dev/null || true
fi
cd "\$PROJECT_ROOT/binding-prediction"
echo "PROJECT_ROOT=\$PROJECT_ROOT"
echo "NEO_BINDER_WORK_ROOT=\$NEO_BINDER_WORK_ROOT"
echo "INPUT_TSV=\$INPUT_TSV"
echo "cwd: \$(pwd)"
EOF
chmod +x "$NEW_ROOT/activate.sh"

if [[ -z "$PROJECT_ALIAS" ]]; then
    PROJECT_ALIAS="$(basename "$NEW_ROOT")"
fi

if [[ "$REGISTER_BASHRC" == true ]]; then
    mkdir -p "$HOME/.bashrc.d"
    _rc_snippet="$HOME/.bashrc.d/neo_binder_${PROJECT_ALIAS}.sh"
    echo "alias neo_${PROJECT_ALIAS}='source \"$NEW_ROOT/activate.sh\"'" > "$_rc_snippet"
    if ! grep -q 'bashrc.d/neo_binder_' "$HOME/.bashrc" 2>/dev/null; then
        cat >> "$HOME/.bashrc" <<'BASHRC'

# Neo binder project aliases (optional; created by init_project.sh --register-bashrc)
for _neo_env in "$HOME"/.bashrc.d/neo_binder_*.sh; do
    [[ -f "$_neo_env" ]] && source "$_neo_env"
done
unset _neo_env
BASHRC
    fi
    echo "Registered alias: neo_${PROJECT_ALIAS}  →  source $NEW_ROOT/activate.sh"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "  source $NEW_ROOT/activate.sh"
echo "  python scripts/create_input.py"
echo "  ./slurm/submit_step.sh 0"
echo ""
echo "Paths in $NEW_ROOT/.env are auto-loaded by ./slurm/submit_step.sh and SLURM jobs."
if [[ "$REGISTER_BASHRC" != true ]]; then
    echo "Optional: re-run with --register-bashrc --alias $PROJECT_ALIAS for login auto-activation."
fi
echo ""
echo "One-time per user (if not done): conda env create -f environment.yml -n neo_binder"
