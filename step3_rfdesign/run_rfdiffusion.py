"""Run RFdiffusion for minibinder scaffold generation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from step3_rfdesign.select_peptide_hotspots import validate_hotspots_in_pdb
from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def _resolve_container_image(step_cfg: dict) -> str:
    """Resolve Apptainer image path from config or environment."""
    raw = (
        step_cfg.get("container_image")
        or os.environ.get("RFDIFFUSION_CONTAINER")
        or ""
    )
    if not raw:
        project_root = os.environ.get("PROJECT_ROOT", "")
        if project_root:
            return str(Path(project_root) / "rfdiffusion.sif")
        raise FileNotFoundError(
            "RFdiffusion container not found. Set RFDIFFUSION_CONTAINER or step3.container_image."
        )
    path = str(raw).strip().rstrip("}")
    path = path.replace("${PROJECT_ROOT}", os.environ.get("PROJECT_ROOT", ""))
    if not Path(path).exists():
        raise FileNotFoundError(f"RFdiffusion container not found: {path}")
    return path


def _build_container_cmd(
    repo_root: Path,
    container_image: str,
    hydra_args: list,
) -> list:
    """Build argv for scripts/run_rfdiffusion_container.sh with Hydra overrides."""
    wrapper = repo_root / "scripts" / "run_rfdiffusion_container.sh"
    if not wrapper.exists():
        raise FileNotFoundError(f"Container wrapper not found: {wrapper}")
    return ["bash", str(wrapper), *hydra_args]


def _resolve_weights_dir(rfdiff_root: Path, step_cfg: dict) -> str:
    """Resolve RFdiffusion model checkpoint directory (bind-mounted into container)."""
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


def _quote_hydra_value(value: str) -> str:
    """Quote Hydra override values that contain spaces or brackets."""
    if any(ch in value for ch in (" ", "[", "]", ",")):
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    return value


def _resolve_ppi_checkpoint(
    step_cfg: dict,
    container_weights_dir: str,
    host_weights_dir: str,
) -> str:
    """Return container path to the PPI checkpoint; verify it exists on the host."""
    ckpt_name = str(step_cfg.get("ppi_checkpoint", "Complex_base_ckpt.pt")).strip()
    host_ckpt = Path(host_weights_dir) / ckpt_name
    if not host_ckpt.exists():
        raise FileNotFoundError(
            f"PPI checkpoint not found: {host_ckpt} "
            f"(bind-mounted to {container_weights_dir}/{ckpt_name})"
        )
    return f"{container_weights_dir.rstrip('/')}/{ckpt_name}"


def _validate_design_inputs(
    pdb_path: str,
    hotspot_res: str,
    host_weights_dir: str,
    step_cfg: dict,
) -> None:
    """Fail fast before launching the GPU container."""
    pdb = Path(pdb_path)
    if not pdb.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb}")
    if not pdb.stat().st_size:
        raise ValueError(f"Input PDB is empty: {pdb}")
    validate_hotspots_in_pdb(pdb_path, hotspot_res)
    if hotspot_res:
        _resolve_ppi_checkpoint(
            step_cfg,
            str(step_cfg.get("container_weights_path", "/opt/rfdiffusion/models")),
            host_weights_dir,
        )


def _hydra_overrides(
    row,
    job_out: Path,
    design_id: str,
    weights_dir: str,
    host_weights_dir: str,
    step_cfg: dict,
) -> list:
    """Build Hydra CLI overrides; values with spaces must stay single shell tokens."""
    contig_value = f"[{row['contig_map']}]"
    overrides = [
        f"inference.input_pdb={row['pdb_path']}",
        f"inference.output_prefix={job_out / design_id}",
        f"contigmap.contigs={_quote_hydra_value(contig_value)}",
        f"diffuser.T={step_cfg['diffusion_steps']}",
        f"inference.num_designs={step_cfg['num_designs_per_structure']}",
        f"inference.model_directory_path={weights_dir}",
    ]
    if not step_cfg.get("write_trajectory", False):
        overrides.append("inference.write_trajectory=false")

    hotspot_res = str(row.get("hotspot_res", "") or "").strip()
    if hotspot_res:
        hotspot_value = f"[{hotspot_res}]"
        overrides.append(f"ppi.hotspot_res={_quote_hydra_value(hotspot_value)}")
        overrides.append(
            f"inference.ckpt_override_path={_resolve_ppi_checkpoint(step_cfg, weights_dir, host_weights_dir)}"
        )
        overrides.append(
            f"denoiser.noise_scale_ca={step_cfg.get('denoiser_noise_scale_ca', 0)}"
        )
        overrides.append(
            f"denoiser.noise_scale_frame={step_cfg.get('denoiser_noise_scale_frame', 0)}"
        )
    return overrides


def _log_failure_excerpt(logger, job_out: Path, tail_chars: int = 12000) -> None:
    """Print the traceback section from rfdiffusion.log when inference fails."""
    log_path = job_out / "rfdiffusion.log"
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in text:
        idx = text.rfind("Traceback")
        excerpt = text[idx : idx + tail_chars]
        logger.error("RFdiffusion traceback (from %s):\n%s", log_path, excerpt)
    elif text.strip():
        logger.error("RFdiffusion log tail (%s):\n%s", log_path, text[-tail_chars:])


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
        container_weights = str(
            step_cfg.get("container_weights_path", "/opt/rfdiffusion/models")
        ).strip()
        hotspot_res = str(row.get("hotspot_res", "") or "").strip()
        if hotspot_res:
            logger.info(f"RFdiffusion hotspots for {design_id}: {hotspot_res}")

        try:
            _validate_design_inputs(
                row["pdb_path"],
                hotspot_res,
                weights_dir,
                step_cfg,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Pre-flight validation failed for %s: %s", design_id, exc)
            status_rows.append(
                {"design_id": design_id, "status": "failed", "output_dir": str(job_out)}
            )
            continue

        repo_root = Path(__file__).resolve().parent.parent
        container_image = _resolve_container_image(step_cfg)
        hydra_args = _hydra_overrides(
            row, job_out, design_id, container_weights, weights_dir, step_cfg
        )
        cmd = _build_container_cmd(repo_root, container_image, hydra_args)

        logger.info(f"Running RFdiffusion: {design_id}")
        logger.info(f"Container: {container_image}")
        logger.info(f"RFdiffusion command: {' '.join(cmd)}")
        if dry_run:
            status_rows.append(
                {"design_id": design_id, "status": "dry_run", "output_dir": str(job_out)}
            )
            continue

        try:
            env = os.environ.copy()
            env["RFDIFFUSION_CONTAINER"] = container_image
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
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
            _log_failure_excerpt(logger, job_out)
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
