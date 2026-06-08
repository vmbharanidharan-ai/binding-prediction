#!/bin/bash
# Create input from typed details and submit one pipeline step for a single pair.
#
# Interactive (prompts for ALL fields including Step 5 metadata):
#   ./slurm/run_pair.sh -i --step 1
#
# Command-line (all metadata):
#   ./slurm/run_pair.sh --step 1 -p SIINFEKL -a HLA_A0201 \
#       --gene EGFR --junction J1 --mhcflurry 0.5 --netmhcpan 0.1 \
#       --carriers 12 --psr 0.85 --frameshift 0
#
# Reuse saved input from first step (no re-prompting):
#   source data/generated/current_pair.env
#   ./slurm/submit_step.sh 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STEP=""
INPUT_TSV=""
INTERACTIVE=false
PROMPT_METADATA=false
NEW_INPUT=false
PEPTIDE=""
ALLELE=""
EXTRA_PY_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./slurm/run_pair.sh [options]

Run one pipeline step for a single peptide–HLA pair.

Input options (enter everything once — saved for all steps):
  --interactive, -i       Prompt for peptide, allele, AND all Step 5 metadata
  --prompt-metadata       Also prompt for Step 5 fields when -p/-a given on CLI
  --input <tsv>           Use existing input TSV (skip generation)
  --new-input             Force new input even if current_pair.env exists

  --peptide, -p           Peptide sequence
  --allele, -a            HLA allele (e.g. HLA_A0201)

Step 5 metadata (from neoJunction — enter now or at -i prompt):
  --gene                  Source gene
  --junction              Junction ID
  --mhcflurry             MHCflurry presentation percentile
  --netmhcpan             NetMHCpan EL rank
  --carriers              Patients carrying this neoantigen in cohort
  --psr                   PSR tumor (0–1)
  --frameshift            Frameshift flag (0 or 1)

Run options:
  --step <0-5>            Pipeline step to submit
  --make-only             Create input TSV only, do not sbatch
  -h, --help              Show this help

Workflow:
  1. ./slurm/run_pair.sh -i --step 1     # enter ALL details once
  2. source data/generated/current_pair.env
  3. ./slurm/submit_step.sh 2            # reuses same input automatically
  4. ./slurm/submit_step.sh 3
  ...
EOF
}

MAKE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)         STEP="$2"; shift 2 ;;
        --input)        INPUT_TSV="$2"; shift 2 ;;
        --peptide|-p)   PEPTIDE="$2"; shift 2 ;;
        --allele|-a)    ALLELE="$2"; shift 2 ;;
        --interactive|-i) INTERACTIVE=true; shift ;;
        --prompt-metadata) PROMPT_METADATA=true; shift ;;
        --new-input)    NEW_INPUT=true; shift ;;
        --gene)         EXTRA_PY_ARGS+=(--gene "$2"); shift 2 ;;
        --junction)     EXTRA_PY_ARGS+=(--junction "$2"); shift 2 ;;
        --mhcflurry)    EXTRA_PY_ARGS+=(--mhcflurry "$2"); shift 2 ;;
        --netmhcpan)    EXTRA_PY_ARGS+=(--netmhcpan "$2"); shift 2 ;;
        --carriers)     EXTRA_PY_ARGS+=(--carriers "$2"); shift 2 ;;
        --psr)          EXTRA_PY_ARGS+=(--psr "$2"); shift 2 ;;
        --frameshift)   EXTRA_PY_ARGS+=(--frameshift "$2"); shift 2 ;;
        --make-only)    MAKE_ONLY=true; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p data/generated logs

# Reuse saved input from a previous run unless --new-input or --input specified
if [[ -z "$INPUT_TSV" && "$NEW_INPUT" == false && -f data/generated/current_pair.env ]]; then
    # shellcheck source=/dev/null
    source data/generated/current_pair.env
    if [[ -f "$INPUT_TSV" ]]; then
        echo "Reusing saved input: $INPUT_TSV"
        echo "(use --new-input to create a fresh input TSV)"
    else
        INPUT_TSV=""
    fi
fi

# Build input TSV if not provided or not found
if [[ -z "$INPUT_TSV" || ! -f "$INPUT_TSV" ]]; then
    PY_ARGS=(python scripts/make_input.py)
    if $INTERACTIVE; then
        PY_ARGS+=(--interactive)
    elif [[ -z "$PEPTIDE" || -z "$ALLELE" ]]; then
        echo "No saved input found. Enter all peptide–HLA details:"
        PY_ARGS+=(--interactive)
    else
        PY_ARGS+=(--peptide "$PEPTIDE" --allele "$ALLELE")
        if $PROMPT_METADATA; then
            PY_ARGS+=(--prompt-metadata)
        fi
    fi
    PY_ARGS+=("${EXTRA_PY_ARGS[@]}")
    "${PY_ARGS[@]}"
    # shellcheck source=/dev/null
    source data/generated/current_pair.env
    if [[ -z "$INPUT_TSV" || ! -f "$INPUT_TSV" ]]; then
        echo "Failed to create input TSV." >&2
        exit 1
    fi
fi

echo ""
echo "Using input: $INPUT_TSV"
export INPUT_TSV

if $MAKE_ONLY; then
    echo ""
    echo "Input ready. Submit steps with:"
    echo "  source data/generated/current_pair.env"
    echo "  ./slurm/submit_step.sh 1"
    exit 0
fi

if [[ -z "$STEP" ]]; then
    echo ""
    read -r -p "Which step to submit (0-5)? " STEP
fi

exec "$SCRIPT_DIR/submit_step.sh" "$STEP"
