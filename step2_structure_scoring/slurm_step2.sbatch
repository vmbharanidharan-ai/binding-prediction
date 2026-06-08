#!/bin/bash
# Deprecated: use slurm/step2.sbatch or ./slurm/submit_step.sh 2
exec sbatch "$(dirname "$0")/../slurm/step2.sbatch" "$@"
