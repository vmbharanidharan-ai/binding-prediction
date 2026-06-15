"""Parse ColabFold outputs into a structured TSV manifest."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.logging import setup_logger
from utils.structure_utils import find_complex_pdb_files, load_colabfold_scores
from utils.slurm_utils import load_config

PARSED_COLUMNS = [
    "job_id",
    "peptide",
    "allele",
    "gene",
    "junction",
    "pdb_path",
    "model_id",
    "plddt_mean",
    "pae_mean",
]


def parse_colabfold_outputs(
    structure_dir: str,
    manifest_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Parse all ColabFold output PDBs and scores into a unified TSV."""
    config = load_config(config_path)
    logger = setup_logger("step1_parse", config["paths"]["logs_dir"])
    min_chains = int(config.get("step1", {}).get("colabfold_min_chains", 2))

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    rows = []

    for _, entry in manifest.iterrows():
        job_id = entry["job_id"]
        job_dir = Path(structure_dir) / job_id
        if not job_dir.exists():
            logger.warning(f"Output directory missing: {job_dir}")
            continue

        pdb_files = find_complex_pdb_files(str(job_dir), min_chains=min_chains)
        if not pdb_files:
            logger.warning(
                f"No {min_chains}-chain complex PDB files found for {job_id} "
                "(monomer-only outputs are ignored)"
            )
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

    df = pd.DataFrame(rows, columns=PARSED_COLUMNS)
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise RuntimeError(
            f"No ColabFold PDB outputs found under {structure_dir}. "
            "Check log.txt in each job subdirectory and colabfold_status.tsv."
        )
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
