"""Run ColabFold batch structure prediction for peptide–HLA pairs."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.colabfold_utils import build_colabfold_batch_args, read_colabfold_job_log
from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config
from utils.structure_utils import find_complex_pdb_files


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
    min_chains = int(step_cfg.get("colabfold_min_chains", 2))

    status_rows = []
    for _, row in pending.iterrows():
        job_id = row["job_id"]
        fasta_path = row["fasta_path"]
        job_out = out_root / job_id

        if (job_out / "done.flag").exists() and restart:
            if find_complex_pdb_files(str(job_out), min_chains=min_chains):
                logger.info(f"Skipping completed job: {job_id}")
                status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
                continue
            logger.warning(f"Stale done.flag without PDBs for {job_id}; re-running ColabFold")
            (job_out / "done.flag").unlink()

        job_out.mkdir(parents=True, exist_ok=True)

        colabfold_cmd = step_cfg["colabfold_cmd"]
        repo_root = Path(__file__).resolve().parent.parent
        colabfold_env_sh = repo_root / "scripts" / "colabfold_env.sh"
        colabfold_args = build_colabfold_batch_args(fasta_path, str(job_out), step_cfg)

        if colabfold_env_sh.exists():
            bash_cmd = (
                f"source {shlex.quote(str(colabfold_env_sh))} && "
                f"{shlex.quote(colabfold_cmd)} " + " ".join(shlex.quote(a) for a in colabfold_args)
            )
            run_cmd = ["bash", "-lc", bash_cmd]
            logger.info(f"Running: {bash_cmd}")
        else:
            run_cmd = [colabfold_cmd, *colabfold_args]
            logger.info(f"Running: {' '.join(run_cmd)}")
        if dry_run:
            status_rows.append({"job_id": job_id, "status": "dry_run", "output_dir": str(job_out)})
            continue

        try:
            subprocess.run(run_cmd, check=True, capture_output=False, env=os.environ.copy())
            if not find_complex_pdb_files(str(job_out), min_chains=min_chains):
                raise RuntimeError(
                    f"ColabFold finished but no {min_chains}-chain complex PDB in {job_out}"
                )
            (job_out / "done.flag").touch()
            status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
            logger.info(f"Completed: {job_id}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            log_excerpt = read_colabfold_job_log(str(job_out))
            logger.error(f"ColabFold failed for {job_id}: {e}{log_excerpt}")
            status_rows.append({"job_id": job_id, "status": "failed", "output_dir": str(job_out)})

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart and status_path.stat().st_size > 0:
            try:
                existing = pd.read_csv(status_path, sep="\t")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            if not existing.empty:
                status_df = pd.concat([existing, status_df], ignore_index=True)
                status_df = status_df.drop_duplicates(subset=["job_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)

    failed = [r["job_id"] for r in status_rows if r.get("status") == "failed"]
    if failed:
        raise RuntimeError(
            f"ColabFold failed for: {', '.join(failed)}. "
            "Check log.txt under each job directory and colabfold_status.tsv."
        )


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
