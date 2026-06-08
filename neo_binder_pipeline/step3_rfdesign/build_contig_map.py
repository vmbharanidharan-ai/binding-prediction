"""Build RFdiffusion contig maps for minibinder design."""

import argparse
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import load_config


def build_contig_map(
    ranked_structures_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Generate RFdiffusion contig specifications for each top structure.

    Contig map defines binder length and target chain spans for diffusion.
    """
    config = load_config(config_path)
    logger = setup_logger("step3_contig", config["paths"]["logs_dir"])
    step_cfg = config["step3"]

    df = pd.read_csv(ranked_structures_tsv, sep="\t")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        job_id = row["job_id"]
        pdb_path = row["pdb_path"]
        binder_len = (step_cfg["binder_length_min"] + step_cfg["binder_length_max"]) // 2

        # RFdiffusion contig: binder scaffold + target chain
        contig = f"{binder_len}-{binder_len}/0 B1-1"
        design_id = f"{job_id}_rank{row['structure_rank']}"

        contig_file = out_path / f"{design_id}.contig"
        contig_file.write_text(contig)

        rows.append(
            {
                "design_id": design_id,
                "job_id": job_id,
                "peptide": row["peptide"],
                "allele": row["allele"],
                "pdb_path": pdb_path,
                "structure_rank": row["structure_rank"],
                "structure_confidence_score": row["structure_confidence_score"],
                "contig_map": contig,
                "contig_file": str(contig_file),
                "binder_length": binder_len,
            }
        )
        logger.info(f"Contig map: {design_id} → {contig}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_path / "contig_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    logger.info(f"Contig manifest → {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build RFdiffusion contig maps")
    parser.add_argument("--input", required=True, help="Ranked structures TSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    build_contig_map(args.input, args.output_dir, args.config)


if __name__ == "__main__":
    main()
