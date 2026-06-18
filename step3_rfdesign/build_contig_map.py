"""Build RFdiffusion contig maps for minibinder design."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from step1_5_structure_prep.rfdiffusion_targets import load_truncated_structure_map, resolve_rfdiffusion_pdb_path
from step3_rfdesign.select_peptide_hotspots import build_rfdiffusion_contig, select_hotspots_from_pdb
from utils.logging import setup_logger
from utils.slurm_utils import load_config


def build_contig_map(
    ranked_structures_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Generate RFdiffusion contig specifications for each top structure.

    Fixes peptide + HLA chains from the (truncated) target PDB, selects
    5–6 peptide hotspots for ppi.hotspot_res, and defines binder length.
    """
    config = load_config(config_path)
    logger = setup_logger("step3_contig", config["paths"]["logs_dir"])
    step_cfg = config["step3"]
    hotspot_cfg = step_cfg.get("hotspots", {})

    df = pd.read_csv(ranked_structures_tsv, sep="\t")
    truncated_map = load_truncated_structure_map(config)
    if truncated_map:
        logger.info(
            f"Using Step 1.5 truncated PDBs for {len(truncated_map)} job(s) in RFdiffusion contigs"
        )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        job_id = row["job_id"]
        pdb_path = resolve_rfdiffusion_pdb_path(job_id, row["pdb_path"], config, truncated_map)
        binder_min = int(step_cfg["binder_length_min"])
        binder_max = int(step_cfg["binder_length_max"])

        peptide_chain = str(hotspot_cfg.get("peptide_chain", "auto"))
        hla_chain = str(hotspot_cfg.get("hla_chain", "auto"))

        contig = build_rfdiffusion_contig(
            pdb_path,
            binder_min,
            binder_max,
            peptide_chain=peptide_chain,
            hla_chain=hla_chain,
        )

        if hotspot_cfg.get("auto_select", True):
            hotspots = select_hotspots_from_pdb(
                pdb_path,
                peptide_chain=peptide_chain,
                hla_chain=hla_chain,
                hotspot_cfg=hotspot_cfg,
            )
            hotspot_res = hotspots.hotspot_res
            logger.info(
                f"Hotspots {job_id}: {hotspot_res} "
                f"(positions {hotspots.sequence_positions}, "
                f"{hotspots.amino_acids}, n_hydrophobic={hotspots.n_hydrophobic})"
            )
        else:
            hotspot_res = ""
            hotspots = None

        design_id = f"{job_id}_rank{row['structure_rank']}"
        contig_file = out_path / f"{design_id}.contig"
        contig_file.write_text(contig)

        row_data = {
            "design_id": design_id,
            "job_id": job_id,
            "peptide": row["peptide"],
            "allele": row["allele"],
            "pdb_path": pdb_path,
            "source_pdb_path": row["pdb_path"],
            "structure_rank": row["structure_rank"],
            "structure_confidence_score": row["structure_confidence_score"],
            "contig_map": contig,
            "contig_file": str(contig_file),
            "binder_length_min": binder_min,
            "binder_length_max": binder_max,
            "hotspot_res": hotspot_res,
        }
        if hotspots is not None:
            row_data.update(
                {
                    "hotspot_positions": ",".join(str(p) for p in hotspots.sequence_positions),
                    "hotspot_amino_acids": hotspots.amino_acids,
                    "hotspot_n_hydrophobic": hotspots.n_hydrophobic,
                }
            )

        rows.append(row_data)
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
