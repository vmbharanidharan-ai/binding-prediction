"""Rank peptide–HLA structures by structural confidence (NOT binding affinity)."""

import argparse
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import load_config


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    s = s.astype(float)
    min_val, max_val = s.min(), s.max()
    if max_val == min_val:
        return pd.Series(0.5, index=s.index)
    return (s - min_val) / (max_val - min_val)


def compute_structure_confidence_score(df: pd.DataFrame) -> pd.Series:
    """
    Structural reliability score — NOT biological binding prediction.

    structure_confidence_score =
        normalized_plDDT
      - normalized_PAE
      + cluster_agreement_bonus
      + contact_stability_score

    No arbitrary biological weights applied.
    """
    norm_plddt = normalize_series(df["interface_plddt_mean"].fillna(0))
    norm_pae = normalize_series(df["interface_pae_mean"].fillna(0))
    cluster_bonus = normalize_series(df["cluster_size"].fillna(1))
    contact_score = normalize_series(df["contact_count"].fillna(0))

    score = norm_plddt - norm_pae + cluster_bonus + contact_score
    return score


def rank_structures(
    clustered_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Select top 1–3 representative structures per peptide–HLA pair."""
    config = load_config(config_path)
    logger = setup_logger("step2_rank", config["paths"]["logs_dir"])
    top_n = config["step2"]["top_structures_per_peptide"]

    df = pd.read_csv(clustered_tsv, sep="\t")
    reps = df[df["is_representative"] == True].copy()  # noqa: E712
    reps["structure_confidence_score"] = compute_structure_confidence_score(reps)

    ranked_rows = []
    for job_id, group in reps.groupby("job_id"):
        top = group.nlargest(top_n, "structure_confidence_score")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            ranked_rows.append({**row.to_dict(), "structure_rank": rank})

    result = pd.DataFrame(ranked_rows)
    result = result.sort_values(
        ["job_id", "structure_rank"], ascending=[True, True]
    ).reset_index(drop=True)

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    logger.info(
        f"Ranked {len(result)} top structures across "
        f"{result['job_id'].nunique()} peptide–HLA pairs → {output_tsv}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Rank structures by confidence")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    rank_structures(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
