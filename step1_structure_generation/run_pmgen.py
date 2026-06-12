"""Run PMGen structure prediction via the official PMGen CLI (Step 1 backend)."""

from __future__ import annotations
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from step1_structure_generation.parse_outputs import parse_colabfold_outputs
from utils.hla_helper import resolve_hla_sequence
from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config

# Official PMGen wrapper input columns (see PMGen README Quick Start)
PMGEN_INPUT_COLUMNS = ["peptide", "mhc_seq", "mhc_type", "anchors", "id"]


def _mhc_class_type(allele: str) -> int:
    """Return PMGen mhc_type: 1 = MHC-I, 2 = MHC-II."""
    allele = allele.upper()
    if any(allele.startswith(p) for p in ("HLA_D", "HLA-D", "HLA_E", "HLA-E")):
        return 2
    return 1


def build_pmgen_input(manifest_tsv: str, output_tsv: str, config: dict) -> pd.DataFrame:
    """Build official PMGen wrapper TSV from the pipeline input manifest."""
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    rows = []
    for _, row in manifest.iterrows():
        resolved = resolve_hla_sequence(row["allele"], config)
        if resolved is None:
            continue
        _, mhc_seq = resolved
        rows.append(
            {
                "peptide": row["peptide"],
                "mhc_seq": mhc_seq,
                "mhc_type": _mhc_class_type(row["allele"]),
                "anchors": "",
                "id": row["job_id"],
            }
        )
    df = pd.DataFrame(rows, columns=PMGEN_INPUT_COLUMNS)
    if df.empty:
        raise ValueError(
            f"No PMGen input rows produced from {manifest_tsv}. "
            "Check HLA allele resolution (hla/hla_resolver.py)."
        )
    out = Path(output_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return df


def build_official_pmgen_command(
    pmgen_root: Path,
    pmgen_input: Path,
    pmgen_work: Path,
    step_cfg: dict,
    pmgen_env_sh: Path | None,
    pmgen_env: str = "PMGen",
) -> str:
    """Shell command matching the official PMGen README quick-start."""
    num_models = int(step_cfg.get("num_models", 3))
    models = step_cfg.get("pmgen_models") or [f"model_{i}_ptm" for i in range(1, num_models + 1)]
    models = models[:num_models]
    run_mode = step_cfg.get("pmgen_run_mode", "single")

    if pmgen_env_sh and pmgen_env_sh.exists():
        env_setup = f"source {shlex.quote(str(pmgen_env_sh))}"
    else:
        env_setup = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {shlex.quote(pmgen_env)} && "
            "module load cuda 2>/dev/null || true && unset LD_LIBRARY_PATH"
        )

    # Official flag order: --mode wrapper --run single --df ... --output_dir ... --initial_guess
    pmgen_args = [
        "python",
        "run_PMGen.py",
        "--mode",
        "wrapper",
        "--run",
        run_mode,
        "--df",
        str(pmgen_input),
        "--output_dir",
        str(pmgen_work),
    ]
    if step_cfg.get("pmgen_initial_guess", True):
        pmgen_args.append("--initial_guess")
    pmgen_args.extend(["--num_recycles", str(step_cfg.get("num_recycle", 1))])
    pmgen_args.append("--models")
    pmgen_args.extend(models)
    if step_cfg.get("pmgen_no_netmhcpan", True):
        pmgen_args.append("--no_netmhcpan")

    parts = [
        f"cd {shlex.quote(str(pmgen_root))}",
        env_setup,
        shlex.join(pmgen_args),
    ]
    return " && ".join(parts)


def find_pmgen_pdbs(pmgen_output_dir: Path, job_id: str) -> list[Path]:
    """Locate PDBs under official output layout: alphafold/{id}/."""
    candidates: list[Path] = []

    job_dir = pmgen_output_dir / "alphafold" / job_id
    if job_dir.exists():
        candidates.extend(sorted(job_dir.glob("*.pdb")))
        candidates.extend(sorted(job_dir.glob("*model*.pdb")))

    alphafold_root = pmgen_output_dir / "alphafold"
    if alphafold_root.exists():
        for pdb in sorted(alphafold_root.rglob("*.pdb")):
            if job_id in pdb.as_posix() and "model" in pdb.name.lower():
                candidates.append(pdb)

    for pdb in sorted(alphafold_root.glob("*.pdb")) if alphafold_root.exists() else []:
        if job_id in pdb.name and "model" in pdb.name.lower():
            candidates.append(pdb)

    seen: set[str] = set()
    unique: list[Path] = []
    for pdb in candidates:
        key = str(pdb.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(pdb)
    return unique


def normalize_pmgen_outputs(
    pmgen_output_dir: Path,
    structure_dir: Path,
    manifest_tsv: str,
    max_models: int,
    logger,
) -> None:
    """Copy PMGen alphafold/ PDBs into pipeline step1_structures layout."""
    manifest = pd.read_csv(manifest_tsv, sep="\t")

    for _, row in manifest.iterrows():
        job_id = row["job_id"]
        job_out = structure_dir / job_id
        job_out.mkdir(parents=True, exist_ok=True)

        if (job_out / "done.flag").exists():
            logger.info(f"Skipping already normalized job: {job_id}")
            continue

        pdbs = find_pmgen_pdbs(pmgen_output_dir, job_id)
        if not pdbs:
            logger.warning(f"No PMGen PDBs found for {job_id} under {pmgen_output_dir}/alphafold/")
            continue

        for idx, src in enumerate(pdbs[:max_models], start=1):
            dest = job_out / f"rank_{idx:03d}.pdb"
            shutil.copy2(src, dest)
            logger.info(f"Normalized {src.name} → {dest}")

        (job_out / "done.flag").touch()
        logger.info(f"PMGen normalization complete for {job_id} ({min(len(pdbs), max_models)} models)")


def run_pmgen_batch(
    manifest_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
    dry_run: bool = False,
) -> None:
    """Execute official PMGen wrapper mode and normalize outputs for downstream steps."""
    config = load_config(config_path)
    logger = setup_logger("step1_pmgen", config["paths"]["logs_dir"])
    step_cfg = config["step1"]

    pmgen_root = Path(os.path.expandvars(step_cfg.get("pmgen_root") or os.environ.get("PMGEN_ROOT", ""))).resolve()
    if not pmgen_root.exists():
        raise FileNotFoundError(
            f"PMGen not found at {pmgen_root}. "
            "Clone https://github.com/soedinglab/PMGen.git and run scripts/setup_pmgen_longleaf.sh"
        )
    if not (pmgen_root / "run_PMGen.py").exists():
        raise FileNotFoundError(f"run_PMGen.py not found under {pmgen_root}")

    manifest = pd.read_csv(manifest_tsv, sep="\t")
    structure_dir = Path(output_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)

    pmgen_work = structure_dir / "_pmgen_runs"
    pmgen_work.mkdir(parents=True, exist_ok=True)
    pmgen_input = pmgen_work / "pmgen_input.tsv"
    build_pmgen_input(manifest_tsv, str(pmgen_input), config)

    status_path = structure_dir / "pmgen_status.tsv"
    completed = get_completed_ids(str(status_path), "job_id") if restart else set()
    pending = filter_pending(manifest, completed, "job_id")

    if pending.empty:
        logger.info("All PMGen jobs completed (restart-safe skip).")
    else:
        repo_root = Path(__file__).resolve().parent.parent
        pmgen_env_sh = repo_root / "scripts" / "pmgen_env.sh"
        cmd_str = build_official_pmgen_command(
            pmgen_root,
            pmgen_input,
            pmgen_work,
            step_cfg,
            pmgen_env_sh if pmgen_env_sh.exists() else None,
            step_cfg.get("pmgen_env", "PMGen"),
        )
        logger.info(f"Running PMGen (official CLI): {cmd_str}")
        if dry_run:
            logger.info("Dry run — PMGen command not executed.")
        else:
            result = subprocess.run(
                ["bash", "-lc", cmd_str],
                cwd=str(pmgen_root),
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
            if result.returncode != 0:
                if result.stderr:
                    logger.error(result.stderr[-8000:] if len(result.stderr) > 8000 else result.stderr)
                raise subprocess.CalledProcessError(
                    result.returncode, cmd_str, output=result.stdout, stderr=result.stderr
                )

        status_rows = []
        for _, row in pending.iterrows():
            job_id = row["job_id"]
            pdbs = find_pmgen_pdbs(pmgen_work, job_id)
            status = "completed" if pdbs else "failed"
            status_rows.append({"job_id": job_id, "status": status, "n_pdbs": len(pdbs)})

        if status_rows:
            status_df = pd.DataFrame(status_rows)
            if status_path.exists() and restart:
                existing = pd.read_csv(status_path, sep="\t")
                status_df = pd.concat([existing, status_df], ignore_index=True)
                status_df = status_df.drop_duplicates(subset=["job_id"], keep="last")
            status_df.to_csv(status_path, sep="\t", index=False)

    normalize_pmgen_outputs(
        pmgen_work,
        structure_dir,
        manifest_tsv,
        max_models=int(step_cfg.get("num_models", 3)),
        logger=logger,
    )

    paths = config["paths"]
    parse_colabfold_outputs(
        structure_dir=str(structure_dir),
        manifest_tsv=manifest_tsv,
        output_tsv=f"{paths['step2_outputs']}/parsed_structures.tsv",
        config_path=config_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Run PMGen structure prediction (official CLI wrapper)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_pmgen_batch(
        args.manifest,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
