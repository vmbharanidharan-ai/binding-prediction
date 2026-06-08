"""HLA sequence resolution for pipeline steps (IMGT index + legacy FASTA fallback)."""

import subprocess
import sys
from pathlib import Path

from utils.fasta_utils import normalize_allele_name, read_fasta

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_hla_index(auto_setup: bool = True) -> Path:
    """Ensure IMGT HLA index exists; optionally download and build."""
    index_path = REPO_ROOT / "hla" / "hla_index.pkl"
    if index_path.exists():
        return index_path

    if not auto_setup:
        raise FileNotFoundError(
            f"HLA index missing: {index_path}. Run: python hla/setup_hla.py"
        )

    print("[INFO] HLA index not found — downloading IMGT and building index...")
    subprocess.run([sys.executable, str(REPO_ROOT / "hla" / "download_imgt.py")], check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "hla" / "build_hla_index.py")], check=True)
    return index_path


def resolve_hla_sequence(
    allele: str,
    config: dict,
) -> tuple[str, str] | None:
    """
    Resolve allele to (normalized_key, protein_sequence).

    Uses IMGT index when hla.use_imgt is true (default), else legacy FASTA.
    Returns None if allele cannot be resolved and skip_if_absent is true.
    """
    hla_cfg = config.get("hla", {})
    allele_key = normalize_allele_name(allele)

    if hla_cfg.get("use_imgt", True):
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from hla.hla_resolver import HLAResolver

            ensure_hla_index(hla_cfg.get("auto_setup", True))
            resolver = HLAResolver.from_config(REPO_ROOT / "hla" / "hla_config.yaml")
            seq = resolver.resolve(allele)
            return allele_key, seq
        except Exception as exc:
            if not hla_cfg.get("fallback_to_local_fasta", True):
                raise
            print(f"[WARN] IMGT resolver failed for {allele}: {exc}")

    # Legacy fallback: config/hla_sequences.fasta
    fasta_path = config["paths"]["hla_fasta"]
    hla_seqs = read_fasta(fasta_path)
    if allele_key in hla_seqs:
        return allele_key, hla_seqs[allele_key]

    missing_cfg = hla_cfg.get("missing_allele_handling", {})
    if missing_cfg.get("skip_if_absent", True):
        return None
    raise ValueError(f"HLA allele not found: {allele} (checked IMGT index and {fasta_path})")
