# Build RFdiffusion Apptainer Container on Longleaf

One-time build per `PROJECT_ROOT` (or share one `.sif` across datasets).

## Quick start

```bash
export PROJECT_ROOT=/work/users/$USER/your_dataset/minibinder_prediction
cd $PROJECT_ROOT/binding-prediction

# GPU node (required for --nv validation)
srun --partition=gpu --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
module load apptainer

bash scripts/build_rfdiffusion_container.sh
bash scripts/verify_rfdiffusion_container.sh
```

Or submit: `sbatch slurm/build_rfdiffusion_container.sbatch`

## Output

- Image: `$PROJECT_ROOT/rfdiffusion.sif` (~2–3 GB)
- Weights stay on host: `$PROJECT_ROOT/RFdiffusion/models/` (bind-mounted read-only)

## Definition file

`containers/rfdiffusion.def` pins:
- torch 1.9.1+cu111, numpy 1.23.5, scipy 1.10.1, dgl 1.0.0
- RFdiffusion + SE3Transformer from upstream GitHub

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build hangs on timezone prompt | Use current `rfdiffusion.def` (sets `DEBIAN_FRONTEND=noninteractive` + `TZ=UTC`); kill build and rebuild |
| `apptainer: command not found` | `module load apptainer` |
| Build permission denied | `apptainer build --fakeroot --nv ...` |
| DGL CUDA test fails | Rebuild with `--nv`; run verify on GPU node |
| Weights not found at inference | Download `*.pt` into `RFdiffusion/models/` |

## Reuse across datasets

Copy or symlink a working `rfdiffusion.sif` into each dataset's `PROJECT_ROOT`. Only weights and input PDBs are dataset-specific.
