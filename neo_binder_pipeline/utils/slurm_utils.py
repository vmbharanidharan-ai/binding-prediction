"""SLURM job management and restart-safe execution utilities."""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load pipeline config, expanding environment variables in paths."""
    with open(config_path) as fh:
        raw = fh.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


def ensure_work_dirs(config: dict) -> None:
    """Create all pipeline working directories."""
    paths = config.get("paths", {})
    for key, path in paths.items():
        if key.endswith("_dir") or key.endswith("_outputs") or key == "work_root":
            Path(path).mkdir(parents=True, exist_ok=True)
    Path(config["paths"].get("logs_dir", "./logs")).mkdir(parents=True, exist_ok=True)


def get_completed_ids(output_tsv: str, id_column: str = "job_id") -> set:
    """Return IDs already processed (for restart-safe execution)."""
    if not Path(output_tsv).exists():
        return set()
    df = pd.read_csv(output_tsv, sep="\t")
    if id_column not in df.columns:
        return set()
    return set(df[id_column].astype(str))


def filter_pending(
    input_df: pd.DataFrame,
    completed_ids: set,
    id_column: str = "job_id",
) -> pd.DataFrame:
    """Filter input DataFrame to only pending (unprocessed) rows."""
    if id_column not in input_df.columns:
        input_df = input_df.copy()
        input_df[id_column] = input_df.apply(
            lambda r: f"{r.get('peptide', '')}_{r.get('allele', '')}", axis=1
        )
    return input_df[~input_df[id_column].astype(str).isin(completed_ids)]


def append_tsv(df: pd.DataFrame, output_path: str) -> None:
    """Append DataFrame to TSV, writing header only if file is new."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, sep="\t", index=False, mode="a", header=write_header)


def submit_slurm(script_path: str, extra_args: Optional[List[str]] = None) -> str:
    """Submit a SLURM batch script and return the job ID."""
    cmd = ["sbatch", script_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id


def write_checkpoint(checkpoint_path: str, step: str, status: str) -> None:
    """Write a checkpoint file for pipeline restart."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(f"{step}\t{status}\n")


def read_last_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Read the last completed pipeline step from checkpoint file."""
    if not Path(checkpoint_path).exists():
        return None
    with open(checkpoint_path) as fh:
        lines = fh.readlines()
    if not lines:
        return None
    return lines[-1].strip().split("\t")[0]
