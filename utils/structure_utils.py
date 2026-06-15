"""Structure parsing and geometric utilities."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def parse_pdb_coordinates(pdb_path: str) -> Dict[str, np.ndarray]:
    """Extract CA atom coordinates keyed by chain:residue_number."""
    coords = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip() or "A"
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[f"{chain}:{resnum}"] = np.array([x, y, z])
    return coords


def parse_pdb_all_atoms(pdb_path: str) -> List[Dict]:
    """Parse all ATOM records from a PDB file."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atoms.append(
                {
                    "chain": line[21].strip() or "A",
                    "resnum": int(line[22:26].strip()),
                    "atom": line[12:16].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )
    return atoms


def compute_rmsd(
    coords_a: Dict[str, np.ndarray],
    coords_b: Dict[str, np.ndarray],
    shared_keys: Optional[List[str]] = None,
) -> float:
    """Compute RMSD between two coordinate sets over shared residues."""
    if shared_keys is None:
        shared_keys = sorted(set(coords_a) & set(coords_b))
    if not shared_keys:
        return float("inf")
    diffs = [coords_a[k] - coords_b[k] for k in shared_keys]
    return float(np.sqrt(np.mean([np.dot(d, d) for d in diffs])))


def count_contacts(
    atoms_a: List[Dict],
    atoms_b: List[Dict],
    cutoff: float = 4.0,
) -> int:
    """Count inter-chain atom contacts within cutoff distance (Å)."""
    count = 0
    seen: Set[Tuple] = set()
    for a in atoms_a:
        pa = np.array([a["x"], a["y"], a["z"]])
        for b in atoms_b:
            pb = np.array([b["x"], b["y"], b["z"]])
            dist = np.linalg.norm(pa - pb)
            if dist < cutoff:
                pair = (a["chain"], a["resnum"], b["chain"], b["resnum"])
                if pair not in seen:
                    seen.add(pair)
                    count += 1
    return count


def split_chains_by_length(
    atoms: List[Dict], peptide_max_len: int = 15
) -> Tuple[List[Dict], List[Dict]]:
    """Heuristically split peptide (short chain) from HLA (long chain)."""
    chain_residues: Dict[str, Set[int]] = {}
    for atom in atoms:
        chain_residues.setdefault(atom["chain"], set()).add(atom["resnum"])

    chains = sorted(chain_residues.items(), key=lambda x: len(x[1]))
    peptide_chain = chains[0][0]
    hla_chain = chains[-1][0]

    peptide_atoms = [a for a in atoms if a["chain"] == peptide_chain]
    hla_atoms = [a for a in atoms if a["chain"] == hla_chain]
    return peptide_atoms, hla_atoms


def load_colabfold_scores(output_dir: str) -> Dict[str, float]:
    """Load pLDDT and PAE from ColabFold output JSON if present."""
    scores: Dict[str, float] = {}
    out_path = Path(output_dir)

    json_files = list(out_path.glob("*.json")) + list(out_path.glob("**/*.json"))
    for jf in json_files:
        try:
            with open(jf) as fh:
                data = json.load(fh)
            if "plddt" in data:
                scores["plddt_mean"] = float(np.mean(data["plddt"]))
            if "pae" in data:
                pae = np.array(data["pae"])
                scores["pae_mean"] = float(np.mean(pae))
        except (json.JSONDecodeError, KeyError):
            continue

    pdb_files = list(out_path.glob("*.pdb")) + list(out_path.glob("**/*_model_*.pdb"))
    for pdb in pdb_files:
        b_factors = []
        with open(pdb) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    b_factors.append(float(line[60:66]))
        if b_factors:
            scores.setdefault("plddt_mean", float(np.mean(b_factors)))

    return scores


def find_pdb_files(directory: str, pattern: str = "*.pdb") -> List[Path]:
    """Recursively find PDB files in a directory."""
    return sorted(Path(directory).rglob(pattern))


def get_pdb_chains(pdb_path: str) -> Set[str]:
    """Return chain IDs present in ATOM/HETATM records."""
    chains: Set[str] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                chains.add(line[21].strip() or "A")
    return chains


def is_complex_pdb(pdb_path: str, min_chains: int = 2) -> bool:
    """Return True if PDB contains at least min_chains distinct chains."""
    return len(get_pdb_chains(pdb_path)) >= min_chains


def find_complex_pdb_files(
    directory: str,
    min_chains: int = 2,
    pattern: str = "*.pdb",
) -> List[Path]:
    """Find PDB files that contain a multimer/complex (multiple chains)."""
    return [p for p in find_pdb_files(directory, pattern) if is_complex_pdb(str(p), min_chains)]


def extract_model_id(pdb_path: str) -> int:
    """Extract model number from ColabFold PDB filename."""
    match = re.search(r"model[_-]?(\d+)", Path(pdb_path).stem, re.IGNORECASE)
    return int(match.group(1)) if match else 0
