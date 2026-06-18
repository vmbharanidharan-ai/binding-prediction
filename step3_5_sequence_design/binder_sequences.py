"""Resolve designed binder sequences from Step 3.5 for Step 4."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


def load_designed_binder_map(config: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Return design_id → {binder_sequence, sequence_fasta, ...} when Step 3.5 ran."""
    step_cfg = config.get("step3_5", {})
    if not step_cfg.get("enabled", True):
        return {}

    manifest = Path(config["paths"]["step3_5_outputs"]) / "designed_binders.tsv"
    if not manifest.exists() or manifest.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(manifest, sep="\t")
    except pd.errors.EmptyDataError:
        return {}

    if "design_id" not in df.columns or "binder_sequence" not in df.columns:
        return {}

    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        seq = str(row.get("binder_sequence", "") or "").strip()
        if not seq:
            continue
        out[str(row["design_id"])] = {
            "binder_sequence": seq,
            "sequence_fasta": str(row.get("sequence_fasta", "") or ""),
            "mpnn_score": str(row.get("mpnn_score", "") or ""),
            "binder_chain": str(row.get("binder_chain", "") or ""),
        }
    return out


def resolve_binder_sequence(
    design_id: str,
    binder_row: Mapping[str, object],
    config: Mapping[str, object],
    designed_map: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[str, str]:
    """
    Prefer Step 3.5 ProteinMPNN sequence; fall back to FASTA or poly-Ala placeholder.

    Returns (sequence, source) where source is mpnn|fasta|placeholder.
    """
    mapping = designed_map if designed_map is not None else load_designed_binder_map(config)
    if str(design_id) in mapping:
        return mapping[str(design_id)]["binder_sequence"], "mpnn"

    fasta_path = str(binder_row.get("sequence_fasta", "") or "")
    if fasta_path and Path(fasta_path).exists():
        from utils.fasta_utils import read_fasta

        records = read_fasta(fasta_path)
        seq = next(iter(records.values()), "")
        if seq and not (len(set(seq)) == 1 and "A" in seq):
            return seq, "fasta"

    length = int(binder_row.get("binder_length_max", binder_row.get("binder_length", 65)) or 65)
    return "A" * length, "placeholder"
