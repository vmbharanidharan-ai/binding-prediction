#!/usr/bin/env python3
"""Run MHCflurry binding-affinity prediction for a peptide–allele pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from utils.allele_utils import pipeline_to_mhcflurry_allele
from utils.logging import setup_logger
from utils.slurm_utils import load_config

NA_METRICS = {
    "mhcflurry_ic50_nm": float("nan"),
    "mhcflurry_affinity_percentile": float("nan"),
    "mhcflurry_presentation_score": float("nan"),
}


def write_output_tsv(output_dir: Path, pair_id: str, metrics: dict[str, float]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair_id}_mhcflurry_ba.tsv"
    row = {"pair_id": pair_id, **metrics}
    pd.DataFrame([row]).to_csv(out_path, sep="\t", index=False)
    return out_path


def run_mhcflurry_ba(
    peptide: str,
    allele: str,
    pair_id: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
) -> Path:
    """Predict MHCflurry affinity/presentation and write a one-row TSV."""
    config = load_config(config_path)
    logger = setup_logger("mhcflurry_ba", config["paths"]["logs_dir"])
    out_dir = Path(output_dir)

    if not config.get("mhcflurry", {}).get("ba_mode", True):
        logger.info("mhcflurry.ba_mode=false — skipping BA prediction.")
        return write_output_tsv(out_dir, pair_id, NA_METRICS)

    mhcflurry_allele = pipeline_to_mhcflurry_allele(allele)
    logger.info(f"MHCflurry allele: {allele} -> {mhcflurry_allele}")

    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError:
        logger.error(
            "mhcflurry is not installed in the active environment. "
            "Install with: pip install mhcflurry"
        )
        return write_output_tsv(out_dir, pair_id, NA_METRICS)

    try:
        predictor = Class1PresentationPredictor.load()
        result = predictor.predict(
            peptides=[peptide],
            alleles=[mhcflurry_allele],
            include_affinity_percentile=True,
        )
        if result.empty:
            logger.warning("MHCflurry returned no predictions.")
            return write_output_tsv(out_dir, pair_id, NA_METRICS)

        row = result.iloc[0]
        metrics = {
            "mhcflurry_ic50_nm": float(row.get("affinity", float("nan"))),
            "mhcflurry_affinity_percentile": float(row.get("affinity_percentile", float("nan"))),
            "mhcflurry_presentation_score": float(row.get("presentation_score", float("nan"))),
        }
        out_path = write_output_tsv(out_dir, pair_id, metrics)
        logger.info(f"MHCflurry BA metrics written: {out_path}")
        return out_path
    except Exception as exc:
        logger.error(f"MHCflurry prediction failed: {exc}")
        return write_output_tsv(out_dir, pair_id, NA_METRICS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MHCflurry BA prediction")
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    run_mhcflurry_ba(
        peptide=args.peptide,
        allele=args.allele,
        pair_id=args.pair_id,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
