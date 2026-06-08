"""Parse ColabFold outputs into a structured TSV manifest."""

import argparse
from pathlib import Path

import pandas as pd

from utils.logging import setup_logger
from utils.structure_utils import find_pdb_files, load_colabfold_scores
from utils.slurm_utils import load_config


def parse_colabfold_outputs(
    structure_dir: str,
    manifest_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Parse all ColabFold output PDBs and scores into a unified TSV."""
    config = load_config(config_path)
    logger = setup_logger("step1_parse", config["paths"]["logs_dir"])

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    rows = []

    for _, entry in manifest.iterrows():
        job_id = entry["job_id"]
        job_dir = Path(structure_dir) / job_id
        if not job_dir.exists():
            logger.warning(f"Output directory missing: {job_dir}")
            continue

        pdb_files = find_pdb_files(str(job_dir))
        if not pdb_files:
            logger.warning(f"No PDB files found for {job_id}")
            continue

        scores = load_colabfold_scores(str(job_dir))
        for pdb in pdb_files:
            rows.append(
                {
                    "job_id": job_id,
                    "peptide": entry["peptide"],
                    "allele": entry["allele"],
                    "gene": entry.get("gene", ""),
                    "junction": entry.get("junction", ""),
                    "pdb_path": str(pdb),
                    "model_id": pdb.stem,
                    "plddt_mean": scores.get("plddt_mean", None),
                    "pae_mean": scores.get("pae_mean", None),
                }
            )

    df = pd.DataFrame(rows)
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Parsed {len(df)} structures → {output_tsv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse ColabFold outputs")
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    parse_colabfold_outputs(
        args.structure_dir, args.manifest, args.output, args.config
    )


if __name__ == "__main__":
    main()
