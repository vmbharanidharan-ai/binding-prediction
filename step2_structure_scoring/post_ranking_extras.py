"""Run Rosetta interface and MHCflurry BA after Step 2 ranking."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.logging import setup_logger
from utils.slurm_utils import load_config


def run_post_ranking_extras(
    ranked_structures_tsv: str,
    config_path: str = "config/config.yaml",
) -> None:
    """Invoke Rosetta InterfaceAnalyzer and MHCflurry BA for top-ranked structures."""
    config = load_config(config_path)
    logger = setup_logger("step2_post_rank", config["paths"]["logs_dir"])
    work_root = Path(config["paths"]["work_root"])

    ranked_path = Path(ranked_structures_tsv)
    if not ranked_path.exists():
        logger.warning(f"Ranked structures not found: {ranked_path}")
        return

    ranked_df = pd.read_csv(ranked_path, sep="\t")
    if ranked_df.empty:
        logger.warning("ranked_structures.tsv is empty — skipping post-ranking extras.")
        return

    if "pdb_path" not in ranked_df.columns:
        logger.error("ranked_structures.tsv missing pdb_path column.")
        return

    top_df = (
        ranked_df.sort_values("structure_rank")
        .groupby("job_id", as_index=False)
        .first()
    )

    rosetta_out = work_root / "rosetta_interface"
    mhcflurry_out = work_root / "mhcflurry_ba"
    rosetta_cfg = config.get("rosetta", {})

    for _, row in top_df.iterrows():
        pair_id = str(row["job_id"])
        peptide = str(row.get("peptide", ""))
        allele = str(row.get("allele", ""))
        top_pdb = str(row["pdb_path"])

        logger.info(f"Post-ranking extras for {pair_id}")

        if rosetta_cfg.get("enabled", True):
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_rosetta_interface.py"),
                    "--pdb",
                    top_pdb,
                    "--pair-id",
                    pair_id,
                    "--output-dir",
                    str(rosetta_out),
                    "--rosetta-bin",
                    rosetta_cfg.get("bin_path", ""),
                    "--config",
                    config_path,
                ],
                check=False,
                cwd=str(REPO_ROOT),
            )
        else:
            logger.info("rosetta.enabled=false — skipping Rosetta interface analysis.")

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_mhcflurry_ba.py"),
                "--peptide",
                peptide,
                "--allele",
                allele,
                "--pair-id",
                pair_id,
                "--output-dir",
                str(mhcflurry_out),
                "--config",
                config_path,
            ],
            check=False,
            cwd=str(REPO_ROOT),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 2 post-ranking extras")
    parser.add_argument("--ranked", required=True, help="Path to ranked_structures.tsv")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_post_ranking_extras(args.ranked, args.config)


if __name__ == "__main__":
    main()
