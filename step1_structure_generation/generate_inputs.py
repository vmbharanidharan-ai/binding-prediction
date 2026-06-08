"""Generate ColabFold FASTA inputs from peptide–allele TSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.fasta_utils import normalize_allele_name, write_fasta
from utils.hla_helper import resolve_hla_sequence
from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def generate_colabfold_inputs(
    input_tsv: str,
    output_dir: str,
    hla_fasta: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
) -> pd.DataFrame:
    """
    Generate per-pair ColabFold FASTA files.

    Each file contains peptide + HLA sequence for complex prediction.
    """
    config = load_config(config_path)
    logger = setup_logger("step1_generate", config["paths"]["logs_dir"])

    df = pd.read_csv(input_tsv, sep="\t")
    df["allele_norm"] = df["allele"].apply(normalize_allele_name)
    df["job_id"] = df.apply(
        lambda r: f"{r['peptide']}_{r['allele_norm']}", axis=1
    )

    manifest_path = Path(output_dir) / "input_manifest.tsv"
    completed = get_completed_ids(str(manifest_path), "job_id") if restart else set()
    pending = filter_pending(df, completed, "job_id")

    if pending.empty:
        logger.info("All inputs already generated (restart-safe skip).")
        return pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else df

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in pending.iterrows():
        resolved = resolve_hla_sequence(row["allele"], config)
        if resolved is None:
            logger.warning(f"HLA sequence not found for {row['allele']}, skipping.")
            continue
        allele_key, hla_seq = resolved

        job_id = row["job_id"]
        fasta_path = out_path / f"{job_id}.fasta"
        records = [
            (f"peptide_{row['peptide']}", row["peptide"]),
            (allele_key, hla_seq),
        ]
        write_fasta(records, str(fasta_path))

        manifest_rows.append(
            {
                "job_id": job_id,
                "peptide": row["peptide"],
                "allele": row["allele"],
                "allele_norm": allele_key,
                "fasta_path": str(fasta_path),
                "gene": row.get("gene", ""),
                "junction": row.get("junction", ""),
            }
        )
        logger.info(f"Generated input: {fasta_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_path.exists() and restart:
        existing = pd.read_csv(manifest_path, sep="\t")
        manifest_df = pd.concat([existing, manifest_df], ignore_index=True)

    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    logger.info(f"Manifest written: {manifest_path} ({len(manifest_df)} entries)")
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Generate ColabFold inputs")
    parser.add_argument("--input", required=True, help="Step 5 input TSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for FASTAs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    generate_colabfold_inputs(
        args.input,
        args.output_dir,
        config["paths"]["hla_fasta"],
        args.config,
        restart=not args.no_restart,
    )


if __name__ == "__main__":
    main()
