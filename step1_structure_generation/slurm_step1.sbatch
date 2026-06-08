#!/bin/bash
# Deprecated: use slurm/step1.sbatch or ./slurm/submit_step.sh 1
exec sbatch "$(dirname "$0")/../slurm/step1.sbatch" "$@"
