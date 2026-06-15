"""ColabFold multimer FASTA formatting and batch CLI helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Mapping, Sequence

from utils.fasta_utils import read_fasta, write_single_fasta


def is_colabfold_complex_fasta(path: str, min_chains: int = 2) -> bool:
    """Return True if FASTA is a single colon-separated ColabFold multimer entry."""
    fasta_path = Path(path)
    if not fasta_path.exists():
        return False
    records = read_fasta(str(fasta_path))
    if len(records) != 1:
        return False
    sequence = next(iter(records.values()))
    return sequence.count(":") >= min_chains - 1


def write_colabfold_complex_fasta(
    header: str,
    chain_sequences: Sequence[str],
    path: str,
) -> None:
    """Write ColabFold multimer FASTA: one header, chains joined by ':'."""
    if len(chain_sequences) < 2:
        raise ValueError("ColabFold multimer requires at least two chains")
    write_single_fasta(header, ":".join(chain_sequences), path)


def build_colabfold_batch_args(
    fasta_path: str,
    output_dir: str,
    step_cfg: Mapping[str, object],
) -> List[str]:
    """Build colabfold_batch CLI args for a multimer job."""
    args = [
        fasta_path,
        output_dir,
        "--num-models",
        str(step_cfg["num_models"]),
        "--num-recycle",
        str(step_cfg["num_recycle"]),
    ]

    model_type = step_cfg.get("colabfold_model_type")
    if model_type:
        args.extend(["--model-type", str(model_type)])

    pair_mode = step_cfg.get("colabfold_pair_mode")
    if pair_mode:
        args.extend(["--pair-mode", str(pair_mode)])

    rank_by = step_cfg.get("colabfold_rank_by")
    if rank_by:
        args.extend(["--rank", str(rank_by)])

    data_dir = step_cfg.get("colabfold_data_dir") or os.environ.get("COLABFOLD_DATA_DIR")
    if data_dir:
        args.extend(["--data", str(data_dir)])

    return args


def read_colabfold_job_log(job_dir: str, tail_lines: int = 40) -> str:
    """Return the tail of ColabFold log.txt from a job output directory."""
    log_path = Path(job_dir) / "log.txt"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    if not lines:
        return ""
    excerpt = "\n".join(lines[-tail_lines:])
    return f"\n--- ColabFold log.txt ({log_path}) ---\n{excerpt}\n"
