#!/bin/bash
# Submit a single pipeline step on Longleaf.
# Usage: ./slurm/submit_step.sh <step>
#   step: 0 | 1 | 2 | 3 | 4 | 5 | embeddings | step1 | step2 | ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STEP="${1:-}"
case "$STEP" in
  0|embeddings|step0) JOB="step0_embeddings" ;;
  1|step1)           JOB="step1" ;;
  2|step2)           JOB="step2" ;;
  3|step3)           JOB="step3" ;;
  4|step4)           JOB="step4" ;;
  5|step5)           JOB="step5" ;;
  *)
    echo "Usage: $0 <step>"
    echo ""
    echo "Steps (submit in order, step0 can run parallel with step1):"
    echo "  0 / embeddings  — ESM-2 embeddings"
    echo "  1 / step1       — ColabFold peptide–HLA structures"
    echo "  2 / step2       — Structure scoring + clustering"
    echo "  3 / step3       — RFdiffusion binder design"
    echo "  4 / step4       — Binder validation (multimer)"
    echo "  5 / step5       — ML ranking"
    echo ""
    echo "Example:"
    echo "  export NEO_BINDER_WORK_ROOT=/work/users/\$USER/neo_binder"
    echo "  $0 1"
    echo "  # after step1 finishes, inspect work/step2_scored/parsed_structures.tsv"
    echo "  $0 2"
    exit 1
    ;;
esac

mkdir -p logs

# Reuse input TSV saved from first interactive run
if [[ -z "${INPUT_TSV:-}" && -f data/generated/current_pair.env ]]; then
    # shellcheck source=/dev/null
    source data/generated/current_pair.env
fi
export INPUT_TSV="${INPUT_TSV:-data/step5_input.tsv}"
export PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -n "$PROJECT_ROOT" && -z "${NEO_BINDER_WORK_ROOT:-}" ]]; then
    export NEO_BINDER_WORK_ROOT="$PROJECT_ROOT/work"
fi
export NEO_BINDER_WORK_ROOT="${NEO_BINDER_WORK_ROOT:-}"
export PMGEN_ROOT="${PMGEN_ROOT:-${PROJECT_ROOT}/PMGen}"
export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
export ALPHAFOLD_ENV="${ALPHAFOLD_ENV:-${PROJECT_ROOT}/alphafoldenv}"
export COLABFOLD_BIN="${COLABFOLD_BIN:-${ALPHAFOLD_ENV}/bin/colabfold_batch}"
export COLABFOLD_DATA_DIR="${COLABFOLD_DATA_DIR:-${PROJECT_ROOT}/colabfold_data}"

echo "Input TSV:      $INPUT_TSV"
echo "Work root:      $NEO_BINDER_WORK_ROOT"
echo "ColabFold bin:  $COLABFOLD_BIN"
echo "PMGen root:     $PMGEN_ROOT"

if [[ -z "$NEO_BINDER_WORK_ROOT" ]]; then
    echo "ERROR: Set NEO_BINDER_WORK_ROOT or PROJECT_ROOT before submitting."
    exit 1
fi

# Inherited SBATCH_* vars from the login shell can break submission on Longleaf.
unset SBATCH_QOS SBATCH_ACCOUNT SBATCH_PARTITION SLURM_QOS 2>/dev/null || true

echo "Submitting slurm/${JOB}.sbatch ..."
sbatch --export=ALL "slurm/${JOB}.sbatch"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f logs/${JOB}_*.out"
case "$JOB" in
  step0_embeddings) SUMMARY_STEP="embeddings" ;;
  *)                SUMMARY_STEP="$JOB" ;;
esac
echo "Summary:  python utils/step_summary.py --step ${SUMMARY_STEP}"
