#!/bin/bash
# Deprecated: use slurm/step3.sbatch or ./slurm/submit_step.sh 3
exec sbatch "$(dirname "$0")/../slurm/step3.sbatch" "$@"
