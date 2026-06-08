#!/bin/bash
# Deprecated: use slurm/step4.sbatch or ./slurm/submit_step.sh 4
exec sbatch "$(dirname "$0")/../slurm/step4.sbatch" "$@"
