"""Run RFdiffusion for minibinder scaffold generation."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def _resolve_inference_invocation(rfdiff_root: Path, step_cfg: dict) -> str:
    """Return python invocation for run_inference.py relative to RFDIFFUSION_ROOT."""
    cmd = str(step_cfg.get("rfdiffusion_cmd", "python scripts/run_inference.py")).strip()
    script = cmd.split(maxsplit=1)[1] if cmd.startswith("python ") else cmd
    script_path = Path(script)

    if script_path.is_absolute():
        return f"python {script_path}"

    parts = script_path.parts
    if parts and parts[0].lower() == "rfdiffusion":
        script_path = Path(*parts[1:])

    candidate = rfdiff_root / script_path
    if not candidate.exists():
        candidate = rfdiff_root / "scripts" / "run_inference.py"
    if not candidate.exists():
        raise FileNotFoundError(f"RFdiffusion inference script not found under {rfdiff_root}")
    return f"python {candidate.relative_to(rfdiff_root)}"


def _resolve_weights_dir(rfdiff_root: Path, step_cfg: dict) -> str:
    """Resolve RFdiffusion model checkpoint directory."""
    for candidate in (
        step_cfg.get("rfdiffusion_weights"),
        os.environ.get("RFDIFFUSION_WEIGHTS"),
        str(rfdiff_root / "models"),
    ):
        if not candidate:
            continue
        path = str(candidate).strip().rstrip("}")
        if Path(path).exists():
            return path
    default = str(rfdiff_root / "models")
    if Path(default).exists():
        return default
    raise FileNotFoundError(
        f"RFdiffusion weights not found. Expected {default} or set RFDIFFUSION_WEIGHTS."
    )


def _hydra_overrides(
    row,
    job_out: Path,
    design_id: str,
    weights_dir: str,
    step_cfg: dict,
) -> list:
    """Build Hydra CLI overrides; values with spaces must stay single shell tokens."""
    overrides = [
        f"inference.input_pdb={row['pdb_path']}",
        f"inference.output_prefix={job_out / design_id}",
        f"contigmap.contigs=[{row['contig_map']}]",
        f"diffuser.T={step_cfg['diffusion_steps']}",
        f"inference.num_designs={step_cfg['num_designs_per_structure']}",
        f"inference.model_directory_path={weights_dir}",
    ]
    hotspot_res = str(row.get("hotspot_res", "") or "").strip()
    if hotspot_res:
        overrides.append(f"ppi.hotspot_res=[{hotspot_res}]")
    return overrides


def _build_inference_shell_cmd(
    rfdiff_root: Path,
    rfdiff_env_sh: Optional[Path],
    inference_invocation: str,
    hydra_args: list,
) -> str:
    """Assemble a shell command with safe quoting for Hydra overrides (non-login shell)."""
    quoted = " ".join(shlex.quote(arg) for arg in hydra_args)
    body = f"cd {shlex.quote(str(rfdiff_root))} && {inference_invocation} {quoted}"
    if rfdiff_env_sh and rfdiff_env_sh.exists():
        return (
            f"unset VIRTUAL_ENV; "
            f"source {shlex.quote(str(rfdiff_env_sh))} && "
            f"export LD_LIBRARY_PATH=${{CONDA_PREFIX}}/lib:${{CONDA_PREFIX}}/lib64 && {body}"
        )
    return body


def _write_job_log(job_out: Path, result, label: str) -> None:
    log_path = job_out / "rfdiffusion.log"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {label} ===\n")
        if result is None:
            return
        if result.stdout:
            fh.write("--- stdout ---\n")
            fh.write(result.stdout)
        if result.stderr:
            fh.write("--- stderr ---\n")
            fh.write(result.stderr)


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
        rfdiff_root = Path(
            os.environ.get(
                "RFDIFFUSION_ROOT",
                str(Path(__file__).resolve().parent.parent.parent / "RFdiffusion"),
            )
        )
        weights_dir = _resolve_weights_dir(rfdiff_root, step_cfg)
        hydra_args = _hydra_overrides(row, job_out, design_id, weights_dir, step_cfg)
        hotspot_res = str(row.get("hotspot_res", "") or "").strip()
        if hotspot_res:
            logger.info(f"RFdiffusion hotspots for {design_id}: {hotspot_res}")

        repo_root = Path(__file__).resolve().parent.parent
        rfdiff_env_sh = repo_root / "scripts" / "rfdiffusion_env.sh"
        inference_invocation = _resolve_inference_invocation(rfdiff_root, step_cfg)

        bash_cmd = _build_inference_shell_cmd(
            rfdiff_root,
            rfdiff_env_sh if rfdiff_env_sh.exists() else None,
            inference_invocation,
            hydra_args,
        )

        logger.info(f"Running RFdiffusion: {design_id}")
        logger.info(f"RFdiffusion shell command: {bash_cmd}")
        if dry_run:
            status_rows.append(
                {"design_id": design_id, "status": "dry_run", "output_dir": str(job_out)}
            )
            continue

        try:
            result = subprocess.run(
                ["bash", "-c", bash_cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            _write_job_log(job_out, result, "success")
            if result.stdout:
                logger.info(result.stdout[-4000:])
            (job_out / "done.flag").touch()
            status_rows.append(
                {"design_id": design_id, "status": "completed", "output_dir": str(job_out)}
            )
            logger.info(f"RFdiffusion completed: {design_id}")
        except subprocess.CalledProcessError as e:
            _write_job_log(job_out, e, "failed")
            if e.stderr:
                logger.error("RFdiffusion stderr (tail):\n%s", e.stderr[-8000:])
            if e.stdout:
                logger.error("RFdiffusion stdout (tail):\n%s", e.stdout[-4000:])
            logger.error(
                "Full RFdiffusion log: %s",
                job_out / "rfdiffusion.log",
            )
            logger.error(f"RFdiffusion failed for {design_id}: {e}")
            status_rows.append(
                {"design_id": design_id, "status": "failed", "output_dir": str(job_out)}
            )
        except FileNotFoundError as e:
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

    failed = [r["design_id"] for r in status_rows if r.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"RFdiffusion failed for: {', '.join(failed)}")

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
