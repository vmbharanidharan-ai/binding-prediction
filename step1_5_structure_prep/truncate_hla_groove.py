"""Truncate ColabFold peptide–HLA complexes to the HLA binding groove for RFdiffusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select

from utils.logging import setup_logger
from utils.slurm_utils import filter_pending, get_completed_ids, load_config


class GrooveTruncationSelect(Select):
    """Keep full peptide chain and HLA groove residues on the heavy chain."""

    def __init__(
        self,
        peptide_chain: str,
        hla_chain: str,
        hla_start: int,
        hla_end: int,
    ) -> None:
        self.peptide_chain = peptide_chain
        self.hla_chain = hla_chain
        self.hla_start = hla_start
        self.hla_end = hla_end

    def accept_residue(self, residue) -> bool:
        chain_id = residue.get_parent().get_id()
        resnum = residue.get_id()[1]
        if chain_id == self.peptide_chain:
            return True
        if chain_id == self.hla_chain and self.hla_start <= resnum <= self.hla_end:
            return True
        return False


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


def infer_peptide_hla_chains(
    pdb_path: str,
    peptide_chain: str = "auto",
    hla_chain: str = "auto",
) -> Tuple[str, str]:
    """Infer peptide (short) and HLA (long) chain IDs from a two-chain complex."""
    lengths = _chain_lengths(pdb_path)
    if len(lengths) < 2:
        raise ValueError(f"Expected at least 2 chains in {pdb_path}, found {sorted(lengths)}")

    if peptide_chain != "auto" and hla_chain != "auto":
        return peptide_chain, hla_chain

    sorted_chains = sorted(lengths.items(), key=lambda item: item[1])
    pep = peptide_chain if peptide_chain != "auto" else sorted_chains[0][0]
    hla = hla_chain if hla_chain != "auto" else sorted_chains[-1][0]
    if pep == hla:
        chains = sorted(lengths)
        pep, hla = chains[0], chains[1]
    return pep, hla


def _plddt_stats(pdb_path: str, chain_id: str, resi_start: Optional[int] = None, resi_end: Optional[int] = None) -> Dict[str, float]:
    """Summarize B-factors (ColabFold pLDDT) for a chain or residue range."""
    bfactors = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if (line[21].strip() or "A") != chain_id:
                continue
            resnum = int(line[22:26])
            if resi_start is not None and resnum < resi_start:
                continue
            if resi_end is not None and resnum > resi_end:
                continue
            if line[12:16].strip() == "CA":
                bfactors.append(float(line[60:66]))
    if not bfactors:
        return {"plddt_mean": float("nan"), "plddt_min": float("nan"), "plddt_max": float("nan"), "n_residues": 0}
    arr = np.array(bfactors)
    return {
        "plddt_mean": float(np.mean(arr)),
        "plddt_min": float(np.min(arr)),
        "plddt_max": float(np.max(arr)),
        "n_residues": int(len(arr)),
    }


def truncate_complex_pdb(
    source_pdb: str,
    output_pdb: str,
    hla_start: int,
    hla_end: int,
    peptide_chain: str = "auto",
    hla_chain: str = "auto",
) -> Dict[str, object]:
    """Write truncated peptide–HLA groove PDB and return metadata."""
    pep_chain, hla_chain_id = infer_peptide_hla_chains(source_pdb, peptide_chain, hla_chain)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", source_pdb)

    selector = GrooveTruncationSelect(pep_chain, hla_chain_id, hla_start, hla_end)
    out_path = Path(output_pdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), selector)

    pep_stats = _plddt_stats(source_pdb, pep_chain)
    hla_stats = _plddt_stats(source_pdb, hla_chain_id, hla_start, hla_end)

    return {
        "source_pdb_path": source_pdb,
        "truncated_pdb_path": str(out_path),
        "peptide_chain": pep_chain,
        "hla_chain": hla_chain_id,
        "hla_residue_start": hla_start,
        "hla_residue_end": hla_end,
        "peptide_residues": pep_stats["n_residues"],
        "hla_residues_kept": hla_stats["n_residues"],
        "peptide_plddt_mean": pep_stats["plddt_mean"],
        "hla_plddt_mean_kept": hla_stats["plddt_mean"],
        "hla_plddt_min_kept": hla_stats["plddt_min"],
    }


def _pick_best_model_per_job(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one structure per job_id — prefer rank_001 / lowest model id."""
    if df.empty:
        return df

    def _rank_key(row: pd.Series) -> Tuple[int, int, str]:
        model_id = str(row.get("model_id", ""))
        rank_hint = 999
        if "rank_001" in model_id or "ranked_0" in model_id:
            rank_hint = 1
        elif "rank_002" in model_id or "ranked_1" in model_id:
            rank_hint = 2
        plddt = float(row.get("plddt_mean", 0) or 0)
        return (rank_hint, -plddt, model_id)

    rows = []
    for job_id, group in df.groupby("job_id"):
        best = sorted(group.to_dict("records"), key=_rank_key)[0]
        rows.append(best)
    return pd.DataFrame(rows)


def truncate_structures(
    parsed_structures_tsv: str,
    output_dir: str,
    config_path: str = "config/config.yaml",
    restart: bool = True,
) -> pd.DataFrame:
    """Truncate peptide–HLA complexes from Step 1 parsed_structures.tsv."""
    config = load_config(config_path)
    logger = setup_logger("step1_5_truncate", config["paths"]["logs_dir"])
    step_cfg = config.get("step1_5", {})

    hla_start = int(step_cfg.get("hla_residue_start", 25))
    hla_end = int(step_cfg.get("hla_residue_end", 180))
    peptide_chain = str(step_cfg.get("peptide_chain", "auto"))
    hla_chain = str(step_cfg.get("hla_chain", "auto"))
    best_only = bool(step_cfg.get("best_model_only", True))

    df = pd.read_csv(parsed_structures_tsv, sep="\t")
    if best_only:
        df = _pick_best_model_per_job(df)
        logger.info(f"Truncating best model per job ({len(df)} complexes)")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "truncation_status.tsv"
    completed = get_completed_ids(str(status_path), "job_id") if restart else set()
    pending = filter_pending(df, completed, "job_id")

    if pending.empty:
        logger.info("All truncation jobs completed (restart-safe skip).")
        manifest_path = out_root / "truncated_structures.tsv"
        return pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else pd.DataFrame()

    rows = []
    status_rows = []
    for _, row in pending.iterrows():
        job_id = row["job_id"]
        source_pdb = row["pdb_path"]
        job_out = out_root / job_id
        truncated_pdb = job_out / f"{job_id}_truncated.pdb"

        if restart and truncated_pdb.exists() and (job_out / "done.flag").exists():
            logger.info(f"Skipping completed truncation: {job_id}")
            status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
            continue

        if not Path(source_pdb).exists():
            logger.error(f"Source PDB missing for {job_id}: {source_pdb}")
            status_rows.append({"job_id": job_id, "status": "failed", "output_dir": str(job_out)})
            continue

        try:
            meta = truncate_complex_pdb(
                source_pdb,
                str(truncated_pdb),
                hla_start=hla_start,
                hla_end=hla_end,
                peptide_chain=peptide_chain,
                hla_chain=hla_chain,
            )
            (job_out / "done.flag").touch()
            rows.append(
                {
                    "job_id": job_id,
                    "peptide": row["peptide"],
                    "allele": row["allele"],
                    "gene": row.get("gene", ""),
                    "junction": row.get("junction", ""),
                    "model_id": row.get("model_id", ""),
                    **meta,
                }
            )
            status_rows.append({"job_id": job_id, "status": "completed", "output_dir": str(job_out)})
            logger.info(
                f"Truncated {job_id}: peptide chain {meta['peptide_chain']} "
                f"({meta['peptide_residues']} res), HLA {meta['hla_chain']} "
                f"{hla_start}-{hla_end} ({meta['hla_residues_kept']} res), "
                f"pLDDT peptide={meta['peptide_plddt_mean']:.1f}, "
                f"HLA kept={meta['hla_plddt_mean_kept']:.1f}"
            )
        except Exception as exc:
            logger.error(f"Truncation failed for {job_id}: {exc}")
            status_rows.append({"job_id": job_id, "status": "failed", "output_dir": str(job_out)})

    manifest_path = out_root / "truncated_structures.tsv"
    if rows:
        new_df = pd.DataFrame(rows)
        if manifest_path.exists() and restart:
            try:
                existing = pd.read_csv(manifest_path, sep="\t")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            if not existing.empty:
                new_df = pd.concat([existing, new_df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["job_id"], keep="last")
        new_df.to_csv(manifest_path, sep="\t", index=False)
        logger.info(f"Truncated structures manifest → {manifest_path} ({len(new_df)} rows)")

    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if status_path.exists() and restart and status_path.stat().st_size > 0:
            try:
                existing_status = pd.read_csv(status_path, sep="\t")
            except pd.errors.EmptyDataError:
                existing_status = pd.DataFrame()
            if not existing_status.empty:
                status_df = pd.concat([existing_status, status_df], ignore_index=True)
                status_df = status_df.drop_duplicates(subset=["job_id"], keep="last")
        status_df.to_csv(status_path, sep="\t", index=False)

    failed = [r["job_id"] for r in status_rows if r.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"Truncation failed for: {', '.join(failed)}")

    return pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate peptide–HLA complexes for RFdiffusion")
    parser.add_argument(
        "--input",
        required=True,
        help="Step 1 parsed_structures.tsv",
    )
    parser.add_argument("--output-dir", required=True, help="Step 1.5 output directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    truncate_structures(
        args.input,
        args.output_dir,
        args.config,
        restart=not args.no_restart,
    )


if __name__ == "__main__":
    main()
