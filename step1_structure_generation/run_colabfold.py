"""Run ColabFold batch structure prediction for peptide–HLA pairs."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def run_colabfold_batch(
    manifest_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
    dry_run: bool = False,
) -> None:
    """Execute colabfold_batch for each pending peptide–HLA pair."""
    config = load_config(config_path)
    logger = setup_logger("step1_colabfold", config["paths"]["logs_dir"])
    step_cfg = config["step1"]

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    status_path = Path(output_dir) / "colabfold_status.tsv"
    completed = get_completed_ids(str(status_path), "job_id") if restart else set()
    pending = filter_pending(manifest, completed, "job_id")

    if pending.empty:
        logger.info("All ColabFold jobs completed (restart-safe skip).")
        return

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    status_rows = []
    for _, row in pending.iterrows():
        job_id = row["job_id"]
        fasta_path = row["fasta_path"]
        job_out = out_root / job_id

        if (job_out / "done.flag").exists() and restart:
            logger.info(f"Skipping completed job: {job_id}")
            status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
            continue

        job_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            step_cfg["colabfold_cmd"],
            fasta_path,
            str(job_out),
            "--num-models", str(step_cfg["num_models"]),
            "--num-recycle", str(step_cfg["num_recycle"]),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        if dry_run:
            status_rows.append({"job_id": job_id, "status": "dry_run", "output_dir": str(job_out)})
            continue

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            (job_out / "done.flag").touch()
            status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
            logger.info(f"Completed: {job_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ColabFold failed for {job_id}: {e}")
            status_rows.append({"job_id": job_id, "status": "failed", "output_dir": str(job_out)})

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart:
            existing = pd.read_csv(status_path, sep="\t")
            status_df = pd.concat([existing, status_df], ignore_index=True)
            status_df = status_df.drop_duplicates(subset=["job_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run ColabFold batch")
    parser.add_argument("--manifest", required=True, help="Input manifest TSV")
    parser.add_argument("--output-dir", required=True, help="Structure output directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_colabfold_batch(
        args.manifest,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
