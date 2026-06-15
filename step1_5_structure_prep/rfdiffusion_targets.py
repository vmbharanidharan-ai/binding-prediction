"""Resolve RFdiffusion target PDB paths (full vs Step 1.5 truncated)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


def load_truncated_structure_map(config: Mapping[str, object]) -> dict[str, str]:
    """Return job_id → truncated_pdb_path when Step 1.5 manifest exists."""
    step_cfg = config.get("step1_5", {})
    if not step_cfg.get("enabled", False):
        return {}

    manifest = Path(config["paths"]["step1_5_outputs"]) / "truncated_structures.tsv"
    if not manifest.exists() or manifest.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(manifest, sep="\t")
    except pd.errors.EmptyDataError:
        return {}

    if "job_id" not in df.columns or "truncated_pdb_path" not in df.columns:
        return {}

    out = {}
    for _, row in df.iterrows():
        path = str(row["truncated_pdb_path"])
        if Path(path).exists():
            out[str(row["job_id"])] = path
    return out


def resolve_rfdiffusion_pdb_path(
    job_id: str,
    default_pdb_path: str,
    config: Mapping[str, object],
    truncated_map: Optional[dict[str, str]] = None,
) -> str:
    """Prefer Step 1.5 truncated PDB for RFdiffusion when enabled and available."""
    step_cfg = config.get("step1_5", {})
    if not step_cfg.get("use_for_rfdiffusion", True):
        return default_pdb_path
    if not step_cfg.get("enabled", False):
        return default_pdb_path

    mapping = truncated_map if truncated_map is not None else load_truncated_structure_map(config)
    return mapping.get(str(job_id), default_pdb_path)
