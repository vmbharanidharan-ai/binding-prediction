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
echo "Submitting slurm/${JOB}.sbatch ..."
sbatch "slurm/${JOB}.sbatch"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f logs/${JOB}_*.out"
case "$JOB" in
  step0_embeddings) SUMMARY_STEP="embeddings" ;;
  *)                SUMMARY_STEP="$JOB" ;;
esac
echo "Summary:  python utils/step_summary.py --step ${SUMMARY_STEP}"
