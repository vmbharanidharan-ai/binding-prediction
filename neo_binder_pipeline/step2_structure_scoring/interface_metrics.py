"""Compute peptide–HLA interface structural metrics."""

import argparse
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.logging import setup_logger
from utils.structure_utils import (
    count_contacts,
    parse_pdb_all_atoms,
    split_chains_by_length,
)
from utils.slurm_utils import load_config

try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False


def compute_buried_surface_area(pdb_path: str) -> float:
    """Approximate buried surface area using FreeSASA if available."""
    if not HAS_FREESASA:
        return 0.0
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        return float(result.totalArea())
    except Exception:
        return 0.0


def compute_interface_metrics(
    structures_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Compute interface metrics for each peptide–HLA structure.

    Metrics (structural reliability only — NOT binding affinity):
      - interface_plddt_mean
      - interface_pae_mean
      - contact_count (<4Å)
      - buried_surface_area
    """
    config = load_config(config_path)
    logger = setup_logger("step2_metrics", config["paths"]["logs_dir"])
    cutoff = config["step2"]["contact_distance_angstrom"]

    df = pd.read_csv(structures_tsv, sep="\t")
    rows = []

    for _, row in df.iterrows():
        pdb_path = row["pdb_path"]
        if not Path(pdb_path).exists():
            logger.warning(f"PDB not found: {pdb_path}")
            continue

        atoms = parse_pdb_all_atoms(pdb_path)
        pep_atoms, hla_atoms = split_chains_by_length(atoms)
        contacts = count_contacts(pep_atoms, hla_atoms, cutoff=cutoff)
        bsa = compute_buried_surface_area(pdb_path)

        rows.append(
            {
                "job_id": row["job_id"],
                "peptide": row["peptide"],
                "allele": row["allele"],
                "gene": row.get("gene", ""),
                "junction": row.get("junction", ""),
                "pdb_path": pdb_path,
                "model_id": row["model_id"],
                "interface_plddt_mean": row.get("plddt_mean", np.nan),
                "interface_pae_mean": row.get("pae_mean", np.nan),
                "contact_count": contacts,
                "buried_surface_area": bsa,
            }
        )

    result = pd.DataFrame(rows)
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Interface metrics for {len(result)} structures → {output_tsv}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute interface metrics")
    parser.add_argument("--input", required=True, help="Parsed structures TSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    compute_interface_metrics(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
