"""Score binder–peptide–HLA complexes by structural reliability."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logging import setup_logger
from utils.structure_utils import (
    count_contacts,
    find_pdb_files,
    load_colabfold_scores,
    parse_pdb_all_atoms,
)
from utils.slurm_utils import load_config


def compute_binder_structural_score(
    interface_plddt: float,
    interface_pae: float,
    contact_count: int,
    plddt_norm: float = 1.0,
    pae_norm: float = 1.0,
) -> float:
    """
    Binder structural reliability score — NOT biological binding.

    binder_structural_score =
        interface_plddt
      - interface_pae
      + contact_count
    """
    norm_plddt = interface_plddt / 100.0 if interface_plddt > 1 else interface_plddt
    norm_pae = interface_pae / 30.0 if interface_pae > 1 else interface_pae
    norm_contacts = contact_count / 50.0

    return norm_plddt - norm_pae + norm_contacts


def score_binder_complexes(
    multimer_output_dir: str,
    complex_manifest_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Extract binder interface metrics and compute structural scores."""
    config = load_config(config_path)
    logger = setup_logger("step4_scoring", config["paths"]["logs_dir"])
    cutoff = config["step2"]["contact_distance_angstrom"]

    manifest = pd.read_csv(complex_manifest_tsv, sep="\t")
    rows = []

    for _, entry in manifest.iterrows():
        complex_id = entry["complex_id"]
        job_dir = Path(multimer_output_dir) / complex_id
        if not job_dir.exists():
            logger.warning(f"Multimer output missing: {job_dir}")
            continue

        pdb_files = find_pdb_files(str(job_dir))
        if not pdb_files:
            continue

        best_pdb = pdb_files[0]
        scores = load_colabfold_scores(str(job_dir))
        atoms = parse_pdb_all_atoms(str(best_pdb))

        chains = sorted(set(a["chain"] for a in atoms))
        if len(chains) >= 2:
            binder_atoms = [a for a in atoms if a["chain"] == chains[0]]
            target_atoms = [a for a in atoms if a["chain"] != chains[0]]
        else:
            binder_atoms, target_atoms = atoms[: len(atoms) // 2], atoms[len(atoms) // 2 :]

        contacts = count_contacts(binder_atoms, target_atoms, cutoff=cutoff)
        plddt = scores.get("plddt_mean", 0.0)
        pae = scores.get("pae_mean", 0.0)

        binder_score = compute_binder_structural_score(plddt, pae, contacts)

        rows.append(
            {
                "complex_id": complex_id,
                "design_id": entry["design_id"],
                "peptide": entry["peptide"],
                "allele": entry["allele"],
                "pdb_path": str(best_pdb),
                "binder_interface_plddt": plddt,
                "binder_interface_pae": pae,
                "binder_contact_count": contacts,
                "binder_structural_score": binder_score,
            }
        )

    result = pd.DataFrame(rows)
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Binder scores for {len(result)} complexes → {output_tsv}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Score binder complexes")
    parser.add_argument("--multimer-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    score_binder_complexes(
        args.multimer_dir, args.manifest, args.output, args.config
    )


if __name__ == "__main__":
    main()
