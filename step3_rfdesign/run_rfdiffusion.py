"""Run RFdiffusion for minibinder scaffold generation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def run_rfdiffusion(
    contig_manifest_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Execute RFdiffusion for each contig map entry.

    Outputs backbone PDBs and designed sequences (FASTA).
    Requires GPU.
    """
    config = load_config(config_path)
    logger = setup_logger("step3_rfdiffusion", config["paths"]["logs_dir"])
    step_cfg = config["step3"]

    manifest = pd.read_csv(contig_manifest_tsv, sep="\t")
    status_path = Path(output_dir) / "rfdiffusion_status.tsv"
    completed = get_completed_ids(str(status_path), "design_id") if restart else set()
    pending = filter_pending(manifest, completed, "design_id")

    if pending.empty:
        logger.info("All RFdiffusion jobs completed (restart-safe skip).")
        return

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    status_rows = []

    for _, row in pending.iterrows():
        design_id = row["design_id"]
        job_out = out_root / design_id

        if (job_out / "done.flag").exists() and restart:
            logger.info(f"Skipping completed design: {design_id}")
            status_rows.append(
                {"design_id": design_id, "status": "completed", "output_dir": str(job_out)}
            )
            continue

        job_out.mkdir(parents=True, exist_ok=True)
        rfdiff_root = os.environ.get(
            "RFDIFFUSION_ROOT",
            str(Path(__file__).resolve().parent.parent.parent / "RFdiffusion"),
        )
        weights_dir = step_cfg.get("rfdiffusion_weights") or os.environ.get(
            "RFDIFFUSION_WEIGHTS", f"{rfdiff_root}/models"
        )
        hydra_args = [
            f"inference.input_pdb={row['pdb_path']}",
            f"inference.output_prefix={job_out / design_id}",
            f"contigmap.contigs=[{row['contig_map']}]",
            f"diffuser.T={step_cfg['diffusion_steps']}",
            f"inference.num_designs={step_cfg['num_designs_per_structure']}",
            f"inference.model_directory_path={weights_dir}",
        ]
        hotspot_res = str(row.get("hotspot_res", "") or "").strip()
        if hotspot_res:
            hydra_args.append(f"ppi.hotspot_res=[{hotspot_res}]")
            logger.info(f"RFdiffusion hotspots for {design_id}: {hotspot_res}")

        repo_root = Path(__file__).resolve().parent.parent
        rfdiff_env_sh = repo_root / "scripts" / "rfdiffusion_env.sh"
        rfdiffusion_cmd = step_cfg["rfdiffusion_cmd"]
        if rfdiffusion_cmd.strip().startswith("python "):
            inference_invocation = rfdiffusion_cmd
        else:
            inference_script = Path(rfdiffusion_cmd)
            if not inference_script.is_absolute():
                inference_script = Path(rfdiff_root) / "scripts" / "run_inference.py"
            inference_invocation = f"python {inference_script}"

        bash_cmd = (
            f"source {rfdiff_env_sh} && cd {rfdiff_root} && "
            f"{inference_invocation} " + " ".join(hydra_args)
            if rfdiff_env_sh.exists()
            else f"cd {rfdiff_root} && {inference_invocation} " + " ".join(hydra_args)
        )

        logger.info(f"Running RFdiffusion: {design_id}")
        if dry_run:
            status_rows.append(
                {"design_id": design_id, "status": "dry_run", "output_dir": str(job_out)}
            )
            continue

        try:
            subprocess.run(["bash", "-lc", bash_cmd], check=True)
            (job_out / "done.flag").touch()
            status_rows.append(
                {"design_id": design_id, "status": "completed", "output_dir": str(job_out)}
            )
            logger.info(f"RFdiffusion completed: {design_id}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"RFdiffusion failed for {design_id}: {e}")
            status_rows.append(
                {"design_id": design_id, "status": "failed", "output_dir": str(job_out)}
            )

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart:
            existing = pd.read_csv(status_path, sep="\t")
            status_df = pd.concat([existing, status_df], ignore_index=True)
            status_df = status_df.drop_duplicates(subset=["design_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)

    # Collect output manifest
    binder_rows = []
    for design_dir in out_root.iterdir():
        if not design_dir.is_dir():
            continue
        pdbs = list(design_dir.glob("*.pdb"))
        fastas = list(design_dir.glob("*.fa")) + list(design_dir.glob("*.fasta"))
        for pdb in pdbs:
            binder_rows.append(
                {
                    "design_id": design_dir.name,
                    "backbone_pdb": str(pdb),
                    "sequence_fasta": str(fastas[0]) if fastas else "",
                }
            )

    if binder_rows:
        binder_df = pd.DataFrame(binder_rows)
        binder_df.to_csv(out_root / "binder_designs.tsv", sep="\t", index=False)
        logger.info(f"Binder designs manifest → {out_root / 'binder_designs.tsv'}")


def main():
    parser = argparse.ArgumentParser(description="Run RFdiffusion binder design")
    parser.add_argument("--manifest", required=True, help="Contig manifest TSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_rfdiffusion(
        args.manifest,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
