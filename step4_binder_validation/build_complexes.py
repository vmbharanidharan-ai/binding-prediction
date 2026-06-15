"""Build binder + peptide–HLA complex FASTA inputs for AlphaFold-Multimer."""

import argparse
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.colabfold_utils import write_colabfold_complex_fasta
from utils.fasta_utils import read_fasta
from utils.hla_helper import resolve_hla_sequence
from utils.logging import setup_logger
from utils.slurm_utils import load_config


def build_binder_complexes(
    binder_designs_tsv: str,
    contig_manifest_tsv: str,
    output_dir: str,
    hla_fasta: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Assemble FASTA inputs for binder + peptide + HLA multimer prediction.

    Chain order: binder, peptide, HLA
    """
    config = load_config(config_path)
    logger = setup_logger("step4_build", config["paths"]["logs_dir"])

    binders = pd.read_csv(binder_designs_tsv, sep="\t")
    contigs = pd.read_csv(contig_manifest_tsv, sep="\t")
    merged = binders.merge(contigs, on="design_id", how="left")
    max_per_peptide = config["step4"].get("max_complexes_per_peptide")
    if max_per_peptide:
        merged = (
            merged.groupby("peptide", group_keys=False)
            .head(max_per_peptide)
            .reset_index(drop=True)
        )
        logger.info(
            f"Capped complexes to {max_per_peptide} per peptide "
            f"({len(merged)} total)"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in merged.iterrows():
        design_id = row["design_id"]
        peptide = row["peptide"]
        resolved = resolve_hla_sequence(row["allele"], config)
        if resolved is None:
            logger.warning(f"HLA sequence missing for {row['allele']}")
            continue
        allele_key, hla_seq = resolved

        binder_seq = ""
        if row.get("sequence_fasta") and Path(row["sequence_fasta"]).exists():
            fasta_records = read_fasta(row["sequence_fasta"])
            binder_seq = next(iter(fasta_records.values()), "")

        if not binder_seq:
            binder_seq = "A" * row.get("binder_length", 65)

        complex_id = f"{design_id}_complex"
        fasta_path = out_path / f"{complex_id}.fasta"
        write_colabfold_complex_fasta(
            complex_id,
            [binder_seq, peptide, hla_seq],
            str(fasta_path),
        )

        rows.append(
            {
                "complex_id": complex_id,
                "design_id": design_id,
                "peptide": peptide,
                "allele": row["allele"],
                "fasta_path": str(fasta_path),
                "binder_backbone": row.get("backbone_pdb", ""),
                "target_pdb": row.get("pdb_path", ""),
            }
        )
        logger.info(f"Built complex: {complex_id}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_path / "complex_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    logger.info(f"Complex manifest → {manifest_path} ({len(manifest)} entries)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build binder complexes")
    parser.add_argument("--binders", required=True)
    parser.add_argument("--contigs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    build_binder_complexes(
        args.binders,
        args.contigs,
        args.output_dir,
        config["paths"]["hla_fasta"],
        args.config,
    )


if __name__ == "__main__":
    main()
