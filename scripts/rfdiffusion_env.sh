#!/bin/bash
# RFdiffusion / SE3nv runtime for Longleaf GPU jobs.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RFDIFFUSION_ENV:-SE3nv}"

module load cuda 2>/dev/null || true
# Prefer conda cudatoolkit + pip DGL over module CUDA libs when possible.
unset LD_LIBRARY_PATH

export RFDIFFUSION_ROOT="${RFDIFFUSION_ROOT:-${PROJECT_ROOT}/RFdiffusion}"
