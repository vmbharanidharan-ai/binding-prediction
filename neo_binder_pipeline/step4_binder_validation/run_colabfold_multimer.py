"""Run ColabFold multimer validation for binder–peptide–HLA complexes."""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def run_colabfold_multimer(
    complex_manifest_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
    dry_run: bool = False,
) -> None:
    """Execute ColabFold multimer prediction for each binder complex."""
    config = load_config(config_path)
    logger = setup_logger("step4_multimer", config["paths"]["logs_dir"])
    step_cfg = config["step4"]

    manifest = pd.read_csv(complex_manifest_tsv, sep="\t")
    status_path = Path(output_dir) / "multimer_status.tsv"
    completed = get_completed_ids(str(status_path), "complex_id") if restart else set()
    pending = filter_pending(manifest, completed, "complex_id")

    if pending.empty:
        logger.info("All multimer jobs completed (restart-safe skip).")
        return

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    status_rows = []

    for _, row in pending.iterrows():
        complex_id = row["complex_id"]
        job_out = out_root / complex_id

        if (job_out / "done.flag").exists() and restart:
            status_rows.append(
                {"complex_id": complex_id, "status": "completed", "output_dir": str(job_out)}
            )
            continue

        job_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            step_cfg["colabfold_multimer_cmd"],
            row["fasta_path"],
            str(job_out),
            "--num-models", str(step_cfg["num_models"]),
            "--num-recycle", str(step_cfg["num_recycle"]),
        ]

        logger.info(f"Running multimer: {complex_id}")
        if dry_run:
            status_rows.append(
                {"complex_id": complex_id, "status": "dry_run", "output_dir": str(job_out)}
            )
            continue

        try:
            subprocess.run(cmd, check=True)
            (job_out / "done.flag").touch()
            status_rows.append(
                {"complex_id": complex_id, "status": "completed", "output_dir": str(job_out)}
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Multimer failed for {complex_id}: {e}")
            status_rows.append(
                {"complex_id": complex_id, "status": "failed", "output_dir": str(job_out)}
            )

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart:
            existing = pd.read_csv(status_path, sep="\t")
            status_df = pd.concat([existing, status_df], ignore_index=True)
            status_df = status_df.drop_duplicates(subset=["complex_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run ColabFold multimer validation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_colabfold_multimer(
        args.manifest,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
