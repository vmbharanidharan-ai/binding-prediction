#!/bin/bash
# Initialize a new project and optionally submit the first pipeline step.
#
# Usage:
#   bash scripts/run_cohort.sh \
#     --root /work/users/$USER/cohort_2/minibinder_prediction \
#     --old-root /work/users/$USER/cohort_1/minibinder_prediction \
#     --input data/generated/my_cohort.tsv \
#     --start-step 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEW_ROOT=""
OLD_ROOT=""
INPUT_TSV=""
START_STEP="0"
GIT_BRANCH="${GIT_BRANCH:-main}"
SUBMIT=false

usage() {
    cat <<'EOF'
Usage: bash scripts/run_cohort.sh --root /path/to/new/project [options]

Options:
  --root, --new-root PATH   New project root (required).
  --old-root PATH           Symlink heavy assets from existing install.
  --input PATH              Input TSV (relative to binding-prediction/ or absolute).
  --start-step N            First step to submit via submit_step.sh (default: 0).
  --branch NAME             Git branch for init_project.sh (default: main).
  --submit                  Submit --start-step after init (requires --input).
  -h, --help                Show this help.

Examples:
  # Init only
  bash scripts/run_cohort.sh --root /work/users/$USER/cohort_2/minibinder_prediction \
    --old-root /work/users/$USER/cohort_1/minibinder_prediction

  # Init + submit step 0
  bash scripts/run_cohort.sh --root /work/users/$USER/cohort_2/minibinder_prediction \
    --old-root /work/users/$USER/cohort_1/minibinder_prediction \
    --input data/generated/cohort.tsv --start-step 0 --submit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root|--new-root) NEW_ROOT="$2"; shift 2 ;;
        --old-root)        OLD_ROOT="$2"; shift 2 ;;
        --input)           INPUT_TSV="$2"; SUBMIT=true; shift 2 ;;
        --start-step)      START_STEP="$2"; shift 2 ;;
        --branch)          GIT_BRANCH="$2"; shift 2 ;;
        --submit)          SUBMIT=true; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$NEW_ROOT" ]]; then
    usage >&2
    exit 1
fi

INIT_ARGS=(--new-root "$NEW_ROOT" --branch "$GIT_BRANCH")
[[ -n "$OLD_ROOT" ]] && INIT_ARGS+=(--old-root "$OLD_ROOT")

bash "$SCRIPT_DIR/init_project.sh" "${INIT_ARGS[@]}"

# shellcheck source=/dev/null
source "$NEW_ROOT/.env"

cd "$PROJECT_ROOT/binding-prediction"

if [[ -n "$INPUT_TSV" ]]; then
    if [[ "$INPUT_TSV" != /* ]]; then
        INPUT_TSV="$PROJECT_ROOT/binding-prediction/$INPUT_TSV"
    fi
    if [[ ! -f "$INPUT_TSV" ]]; then
        echo "ERROR: Input TSV not found: $INPUT_TSV" >&2
        exit 1
    fi
    export INPUT_TSV
    echo "INPUT_TSV=$INPUT_TSV"
fi

if [[ "$SUBMIT" == true ]]; then
    if [[ -z "${INPUT_TSV:-}" ]]; then
        echo "ERROR: --submit requires --input" >&2
        exit 1
    fi
    echo "Submitting step $START_STEP..."
    ./slurm/submit_step.sh "$START_STEP"
else
    echo ""
    echo "Project ready. To run the pipeline:"
    echo "  source $NEW_ROOT/.env"
    echo "  cd \$PROJECT_ROOT/binding-prediction"
    echo "  export INPUT_TSV=/path/to/your_input.tsv"
    echo "  ./slurm/submit_step.sh 0"
fi
