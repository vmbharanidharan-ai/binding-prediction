"""Assign binder sequences to RFdiffusion backbones with ProteinMPNN."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.fasta_utils import write_single_fasta
from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


def _chain_lengths(pdb_path: str) -> Dict[str, int]:
    resnums: Dict[str, set] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "A"
            resnum = int(line[22:26])
            resnums.setdefault(chain, set()).add(resnum)
    return {chain: len(res) for chain, res in resnums.items()}


def infer_binder_chain(
    pdb_path: str,
    binder_length_min: int = 50,
    binder_length_max: int = 80,
    manual_chain: str = "auto",
) -> str:
    """Identify the RFdiffusion-designed binder chain in a complex PDB."""
    if manual_chain and manual_chain != "auto":
        return manual_chain

    lengths = _chain_lengths(pdb_path)
    if not lengths:
        raise ValueError(f"No ATOM records in {pdb_path}")

    if len(lengths) == 1:
        return next(iter(lengths))

    # RFdiffusion often outputs binder + merged receptor (2 chains), not peptide/HLA/binder.
    in_range = [
        (chain, length)
        for chain, length in lengths.items()
        if binder_length_min <= length <= binder_length_max
    ]
    if len(in_range) == 1:
        return in_range[0][0]
    if len(in_range) > 1:
        return max(in_range, key=lambda item: item[1])[0]

    sorted_chains = sorted(lengths.items(), key=lambda item: item[1])
    peptide_chain = sorted_chains[0][0]
    hla_chain = sorted_chains[-1][0]

    candidates = [
        (chain, length)
        for chain, length in lengths.items()
        if chain not in {peptide_chain, hla_chain}
        and binder_length_min <= length <= binder_length_max
    ]
    if not candidates:
        candidates = [
            (chain, length)
            for chain, length in lengths.items()
            if chain not in {peptide_chain, hla_chain}
        ]
    if not candidates:
        raise ValueError(f"Could not infer binder chain in {pdb_path} (chains={lengths})")

    if len(candidates) == 1:
        return candidates[0][0]
    return max(candidates, key=lambda item: item[1])[0]


def model_weights_dir(mpnn_root: Path, step_cfg: Mapping[str, object]) -> Path:
    if step_cfg.get("ca_only"):
        return mpnn_root / "ca_model_weights"
    if step_cfg.get("use_soluble_model"):
        return mpnn_root / "soluble_model_weights"
    custom = step_cfg.get("path_to_model_weights")
    if custom:
        return Path(str(custom))
    return mpnn_root / "vanilla_model_weights"


def parse_mpnn_output_fasta(
    fasta_path: Path,
    binder_chain: str,
    pdb_path: str,
) -> List[Tuple[float, str, str]]:
    """
    Parse ProteinMPNN seqs/*.fa output.

    Returns list of (score, binder_sequence, header) sorted by score ascending.
    """
    chain_order = sorted(_chain_lengths(pdb_path))
    binder_idx = chain_order.index(binder_chain) if binder_chain in chain_order else -1

    samples: List[Tuple[float, str, str]] = []
    header: Optional[str] = None
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">T="):
                header = line
                continue
            if header and not line.startswith(">"):
                score_match = re.search(r"score=([\d.]+)", header)
                score = float(score_match.group(1)) if score_match else 999.0
                if "/" in line and binder_idx >= 0:
                    parts = line.split("/")
                    seq = parts[binder_idx] if binder_idx < len(parts) else line
                else:
                    seq = line
                samples.append((score, seq, header))
                header = None

    return sorted(samples, key=lambda item: item[0])


def run_proteinmpnn_design(
    binder_designs_tsv: str,
    contig_manifest_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run ProteinMPNN on RFdiffusion backbone PDBs; write designed_binders.tsv."""
    config = load_config(config_path)
    logger = setup_logger("step3_5_mpnn", config["paths"]["logs_dir"])
    step_cfg = config.get("step3_5", {})

    mpnn_root = Path(
        step_cfg.get("proteinmpnn_root")
        or __import__("os").environ.get("PROTEINMPNN_ROOT", "")
        or Path(config["paths"].get("work_root", ".")).parent / "ProteinMPNN"
    )
    mpnn_script = mpnn_root / "protein_mpnn_run.py"
    if not mpnn_script.exists() and not dry_run:
        raise FileNotFoundError(
            f"ProteinMPNN not found at {mpnn_root}. Run: bash scripts/setup_proteinmpnn_longleaf.sh"
        )
    logger.info(f"ProteinMPNN root: {mpnn_root}")

    binders = pd.read_csv(binder_designs_tsv, sep="\t")
    if "backbone_id" not in binders.columns:
        binders["backbone_id"] = binders["backbone_pdb"].map(
            lambda p: Path(str(p)).stem if p and str(p) != "nan" else ""
        )
    contigs = pd.read_csv(contig_manifest_tsv, sep="\t")
    contig_cols = [c for c in [
        "design_id", "job_id", "peptide", "allele",
        "binder_length_min", "binder_length_max", "pdb_path",
    ] if c in contigs.columns]
    merged = binders.merge(contigs[contig_cols], on="design_id", how="left")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "proteinmpnn_status.tsv"
    id_column = "backbone_id"
    completed = get_completed_ids(str(status_path), id_column) if restart else set()
    pending = filter_pending(merged, completed, id_column)

    if pending.empty:
        logger.info("All ProteinMPNN jobs completed (restart-safe skip).")
        manifest_path = out_root / "designed_binders.tsv"
        return pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else pd.DataFrame()

    repo_root = Path(__file__).resolve().parent.parent
    env_sh = repo_root / "scripts" / "proteinmpnn_env.sh"
    weights_dir = model_weights_dir(mpnn_root, step_cfg)
    python_bin = str(step_cfg.get("python_cmd") or "python")

    rows = []
    status_rows = []

    for _, row in pending.iterrows():
        design_id = row["design_id"]
        backbone_id = str(row["backbone_id"])
        backbone_pdb = row.get("backbone_pdb", "")
        if not backbone_id:
            logger.error(f"Missing backbone_id for {design_id}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue
        if not backbone_pdb or not Path(backbone_pdb).exists():
            logger.error(f"Backbone PDB missing for {backbone_id}: {backbone_pdb}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue

        job_out = out_root / backbone_id
        if restart and (job_out / "done.flag").exists():
            logger.info(f"Skipping completed MPNN: {backbone_id}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "completed"})
            continue

        job_out.mkdir(parents=True, exist_ok=True)
        binder_min = int(row.get("binder_length_min", step_cfg.get("binder_length_min", 50)))
        binder_max = int(row.get("binder_length_max", step_cfg.get("binder_length_max", 80)))
        try:
            binder_chain = infer_binder_chain(
                backbone_pdb,
                binder_min,
                binder_max,
                str(step_cfg.get("binder_chain", "auto")),
            )
        except ValueError as exc:
            logger.error(
                "Binder chain inference failed for %s (chains=%s): %s",
                backbone_id,
                _chain_lengths(backbone_pdb),
                exc,
            )
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue

        cmd = [
            python_bin,
            str(mpnn_script),
            "--pdb_path",
            str(backbone_pdb),
            "--pdb_path_chains",
            binder_chain,
            "--out_folder",
            str(job_out),
            "--num_seq_per_target",
            str(int(step_cfg.get("num_seq_per_target", 2))),
            "--sampling_temp",
            str(step_cfg.get("sampling_temp", "0.1")),
            "--model_name",
            str(step_cfg.get("model_name", "v_48_020")),
            "--path_to_model_weights",
            str(weights_dir),
            "--batch_size",
            str(int(step_cfg.get("batch_size", 1))),
            "--seed",
            str(int(step_cfg.get("seed", 37))),
        ]
        if step_cfg.get("use_soluble_model"):
            cmd.append("--use_soluble_model")
        if step_cfg.get("ca_only"):
            cmd.append("--ca_only")
        omit_aas = step_cfg.get("omit_AAs")
        if omit_aas:
            cmd.extend(["--omit_AAs", str(omit_aas)])

        logger.info(f"ProteinMPNN {backbone_id}: design chain {binder_chain} on {backbone_pdb}")

        if dry_run:
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "dry_run"})
            continue

        bash_cmd = (
            f"source {shlex.quote(str(env_sh))} && {' '.join(shlex.quote(a) for a in cmd)}"
            if env_sh.exists()
            else " ".join(shlex.quote(a) for a in cmd)
        )
        try:
            result = subprocess.run(
                ["bash", "-lc", bash_cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info(result.stdout[-2000:])
            (job_out / "done.flag").touch()
        except subprocess.CalledProcessError as exc:
            logger.error(f"ProteinMPNN failed for {backbone_id}: {exc}")
            if exc.stdout:
                logger.error("ProteinMPNN stdout (tail):\n%s", exc.stdout[-4000:])
            if exc.stderr:
                logger.error("ProteinMPNN stderr (tail):\n%s", exc.stderr[-4000:])
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue
        except FileNotFoundError as exc:
            logger.error(f"ProteinMPNN failed for {backbone_id}: {exc}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue

        seq_dir = job_out / "seqs"
        fasta_files = sorted(seq_dir.glob("*.fa")) + sorted(seq_dir.glob("*.fasta"))
        if not fasta_files:
            logger.error(f"No MPNN output FASTA for {backbone_id} in {seq_dir}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue

        samples = parse_mpnn_output_fasta(fasta_files[0], binder_chain, backbone_pdb)
        if not samples:
            logger.error(f"Could not parse MPNN sequences for {backbone_id}")
            status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "failed"})
            continue

        best_score, best_seq, best_header = samples[0]
        best_fasta = job_out / f"{backbone_id}_best.fa"
        write_single_fasta(f"{backbone_id},score={best_score}", best_seq, str(best_fasta))

        rows.append(
            {
                "backbone_id": backbone_id,
                "design_id": design_id,
                "job_id": row.get("job_id", ""),
                "peptide": row.get("peptide", ""),
                "allele": row.get("allele", ""),
                "backbone_pdb": backbone_pdb,
                "binder_chain": binder_chain,
                "binder_sequence": best_seq,
                "sequence_fasta": str(best_fasta),
                "mpnn_score": best_score,
                "mpnn_sample_header": best_header,
                "num_sequences_generated": len(samples),
                "mpnn_output_dir": str(job_out),
            }
        )
        status_rows.append({"backbone_id": backbone_id, "design_id": design_id, "status": "completed"})
        logger.info(
            f"Designed {backbone_id}: chain {binder_chain}, len={len(best_seq)}, "
            f"score={best_score:.4f}"
        )

    manifest_path = out_root / "designed_binders.tsv"
    if rows:
        new_df = pd.DataFrame(rows)
        if manifest_path.exists() and restart:
            try:
                existing = pd.read_csv(manifest_path, sep="\t")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            if not existing.empty:
                new_df = pd.concat([existing, new_df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["backbone_id"], keep="last")
        new_df.to_csv(manifest_path, sep="\t", index=False)
        logger.info(f"Designed binders manifest → {manifest_path} ({len(new_df)} rows)")

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart and status_path.stat().st_size > 0:
            try:
                existing_status = pd.read_csv(status_path, sep="\t")
            except pd.errors.EmptyDataError:
                existing_status = pd.DataFrame()
            if not existing_status.empty:
                status_df = pd.concat([existing_status, status_df], ignore_index=True)
                status_df = status_df.drop_duplicates(subset=["backbone_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)

    failed = [r["backbone_id"] for r in status_rows if r.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"ProteinMPNN failed for: {', '.join(failed)}")

    return pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Design binder sequences with ProteinMPNN")
    parser.add_argument("--binders", required=True, help="Step 3 binder_designs.tsv")
    parser.add_argument("--contigs", required=True, help="Step 3 contig_manifest.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_proteinmpnn_design(
        args.binders,
        args.contigs,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
