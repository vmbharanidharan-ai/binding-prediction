#!/bin/bash
# ProteinMPNN runtime for Longleaf GPU jobs.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PROTEINMPNN_ENV:-proteinmpnn}"

module load cuda 2>/dev/null || true

export PROTEINMPNN_ROOT="${PROTEINMPNN_ROOT:-${PROJECT_ROOT}/ProteinMPNN}"
