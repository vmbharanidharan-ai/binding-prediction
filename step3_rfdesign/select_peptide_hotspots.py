"""Select RFdiffusion PPI hotspot residues on the MHC-I peptide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# Hot-spot hierarchy for protein–protein interfaces (Trp > Arg > Tyr > ...).
RESIDUE_SCORE: Dict[str, int] = {
    "W": 10,
    "R": 9,
    "Y": 8,
    "F": 8,
    "K": 7,
    "D": 7,
    "E": 7,
    "M": 6,
    "L": 6,
    "I": 6,
    "H": 6,
    "V": 5,
    "C": 4,
    "T": 3,
    "S": 3,
    "N": 3,
    "Q": 3,
    "A": 1,
    "G": 0,
    "P": -10,
}

# Count toward the ">~3 hydrophobic residues" guideline for binder interfaces.
HYDROPHOBIC_FOR_INTERFACE = frozenset("WFYMLIVC")

SKIP_ALWAYS = frozenset("GP")  # Pro rigid; Gly too small/flexible


@dataclass(frozen=True)
class HotspotSelection:
    """Peptide hotspot residues for RFdiffusion ppi.hotspot_res."""

    chain_id: str
    pdb_resnums: List[int]
    sequence_positions: List[int]
    amino_acids: str
    hotspot_res: str  # e.g. "A4,A5,A6,A7,A8"
    n_hydrophobic: int

    @property
    def hotspot_list(self) -> List[str]:
        return [f"{self.chain_id}{resnum}" for resnum in self.pdb_resnums]


def mhc_i_anchor_positions(peptide_len: int) -> frozenset[int]:
    """MHC-I N- and C-terminal anchor positions (1-indexed sequence positions)."""
    if peptide_len < 2:
        return frozenset()
    anchors = {2}
    if peptide_len >= 8:
        anchors.add(peptide_len)
    return frozenset(anchors)


def preferred_exposed_positions(peptide_len: int) -> List[int]:
    """
    Exposed peptide positions likely contacted by a TCR/minibinder.

    For a 9-mer this is P4–P8; position 3 is a secondary extension candidate.
    """
    anchors = mhc_i_anchor_positions(peptide_len)
    primary = [p for p in range(4, peptide_len) if p not in anchors]
    secondary = [3] if 3 not in anchors and peptide_len >= 3 else []
    return secondary + primary


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


def _chain_resnums(pdb_path: str, chain_id: str) -> List[int]:
    resnums = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "A"
            if chain != chain_id:
                continue
            resnums.add(int(line[22:26]))
    return sorted(resnums)


def peptide_sequence_from_pdb(pdb_path: str, chain_id: str) -> Tuple[str, List[int]]:
    """Return one-letter peptide sequence and matching PDB residue numbers."""
    resnames: Dict[int, str] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "A"
            if chain != chain_id:
                continue
            resnum = int(line[22:26])
            resname = line[17:20].strip()
            resnames[resnum] = THREE_TO_ONE.get(resname, "X")

    ordered_resnums = sorted(resnames)
    seq = "".join(resnames[r] for r in ordered_resnums)
    return seq, ordered_resnums


def _position_eligible(
    position: int,
    aa: str,
    peptide_len: int,
    *,
    allow_n_terminal_small: bool = False,
) -> bool:
    if position in mhc_i_anchor_positions(peptide_len):
        return False
    if aa in SKIP_ALWAYS:
        return False
    if position == 1 and aa in {"A", "G"} and not allow_n_terminal_small:
        return False
    return True


def _score_position(position: int, aa: str, peptide_len: int) -> float:
    score = float(RESIDUE_SCORE.get(aa, 0))
    if position in preferred_exposed_positions(peptide_len):
        score += 3.0
    elif position == 1:
        score -= 2.0
    return score


def _count_hydrophobic(amino_acids: Sequence[str]) -> int:
    return sum(1 for aa in amino_acids if aa in HYDROPHOBIC_FOR_INTERFACE)


def select_peptide_hotspots(
    peptide_seq: str,
    pdb_resnums: Sequence[int],
    chain_id: str = "A",
    *,
    min_hotspots: int = 5,
    max_hotspots: int = 6,
    min_hydrophobic: int = 3,
) -> HotspotSelection:
    """
    Pick 5–6 RFdiffusion hotspots on the peptide chain for pMHC binder design.

    Rules:
      - Skip MHC-I anchor positions (P2 and C-terminal anchor)
      - Skip Pro/Gly; skip N-terminal Ala/Gly (flexible, low contact area)
      - Prefer charged and bulky exposed residues in P4–P8 (P3 optional)
      - Require at least ``min_hydrophobic`` hydrophobic hotspots in the final set
    """
    if len(peptide_seq) != len(pdb_resnums):
        raise ValueError(
            f"Sequence length ({len(peptide_seq)}) != PDB residue count ({len(pdb_resnums)})"
        )

    peptide_len = len(peptide_seq)
    anchors = mhc_i_anchor_positions(peptide_len)
    primary_positions = {p for p in range(4, peptide_len) if p not in anchors}
    secondary_positions = {3} if 3 not in anchors else set()

    candidates: List[Tuple[float, int, str, int]] = []
    for position, aa, pdb_resnum in zip(range(1, peptide_len + 1), peptide_seq, pdb_resnums):
        if not _position_eligible(position, aa, peptide_len):
            continue
        candidates.append((_score_position(position, aa, peptide_len), position, aa, pdb_resnum))

    if not candidates:
        raise ValueError(f"No eligible hotspot candidates for peptide {peptide_seq}")

    candidates.sort(key=lambda item: (-item[0], item[1]))

    primary = [c for c in candidates if c[1] in primary_positions]
    secondary = [c for c in candidates if c[1] in secondary_positions]
    other = [c for c in candidates if c[1] not in primary_positions | secondary_positions]

    if len(primary) >= min_hotspots:
        # e.g. 9-mer P4–P8: use the exposed stretch only (5 residues), not P3 padding
        target_count = min(len(primary), max_hotspots)
        target_count = max(target_count, min_hotspots)
        if len(primary) == min_hotspots and max_hotspots > min_hotspots:
            target_count = min_hotspots
        pool = primary
    else:
        target_count = min_hotspots
        pool = primary + secondary + other

    selected: List[Tuple[int, str, int]] = []
    selected_positions: set[int] = set()

    for item in sorted(pool, key=lambda x: (x[1] not in primary_positions, -x[0], x[1])):
        if len(selected) >= target_count:
            break
        _, position, aa, pdb_resnum = item
        if position in selected_positions:
            continue
        selected.append((position, aa, pdb_resnum))
        selected_positions.add(position)

    if len(selected) < min_hotspots:
        for item in candidates:
            if len(selected) >= min_hotspots:
                break
            _, position, aa, pdb_resnum = item
            if position in selected_positions:
                continue
            selected.append((position, aa, pdb_resnum))
            selected_positions.add(position)

    selected.sort(key=lambda item: item[0])
    amino_acids = [aa for _, aa, _ in selected]

    if len(selected) < min_hotspots:
        raise ValueError(
            f"Only {len(selected)} hotspot candidates for {peptide_seq}; need {min_hotspots}"
        )

    if _count_hydrophobic(amino_acids) < min_hydrophobic:
        for item in candidates:
            _, position, aa, pdb_resnum = item
            if position in selected_positions:
                continue
            if aa not in HYDROPHOBIC_FOR_INTERFACE:
                continue
            if len(selected) >= max_hotspots:
                # Replace lowest-scoring non-hydrophobic if at capacity
                replace_idx = next(
                    (
                        i
                        for i, (_, sel_aa, _) in enumerate(selected)
                        if sel_aa not in HYDROPHOBIC_FOR_INTERFACE
                    ),
                    None,
                )
                if replace_idx is None:
                    continue
                removed_pos = selected[replace_idx][0]
                selected_positions.discard(removed_pos)
                selected[replace_idx] = (position, aa, pdb_resnum)
                selected_positions.add(position)
                amino_acids = [a for _, a, _ in selected]
                if _count_hydrophobic(amino_acids) >= min_hydrophobic:
                    break
            else:
                selected.append((position, aa, pdb_resnum))
                selected_positions.add(position)
                amino_acids = [a for _, a, _ in selected]
                if _count_hydrophobic(amino_acids) >= min_hydrophobic:
                    break

    if _count_hydrophobic(amino_acids) < min_hydrophobic:
        raise ValueError(
            f"Could not select {min_hotspots}-{max_hotspots} peptide hotspots with "
            f">={min_hydrophobic} hydrophobic residues for {peptide_seq}"
        )

    selected.sort(key=lambda item: item[0])
    positions = [pos for pos, _, _ in selected]
    pdb_nums = [resnum for _, _, resnum in selected]
    aas = "".join(aa for _, aa, _ in selected)
    hotspot_res = ",".join(f"{chain_id}{resnum}" for resnum in pdb_nums)

    return HotspotSelection(
        chain_id=chain_id,
        pdb_resnums=pdb_nums,
        sequence_positions=positions,
        amino_acids=aas,
        hotspot_res=hotspot_res,
        n_hydrophobic=_count_hydrophobic(aas),
    )


def select_hotspots_from_pdb(
    pdb_path: str,
    peptide_chain: str = "auto",
    hla_chain: str = "auto",
    hotspot_cfg: Optional[Mapping[str, object]] = None,
) -> HotspotSelection:
    """Infer peptide chain from PDB and select hotspots."""
    cfg = dict(hotspot_cfg or {})
    pep_chain, _ = infer_peptide_hla_chains(pdb_path, peptide_chain, hla_chain)
    seq, pdb_resnums = peptide_sequence_from_pdb(pdb_path, pep_chain)

    manual = cfg.get("manual_hotspots")
    if manual:
        tokens = [str(x).strip() for x in manual if str(x).strip()]
        if len(tokens) == 1 and "," in tokens[0]:
            tokens = [t.strip() for t in tokens[0].split(",") if t.strip()]
        resnums = []
        for token in tokens:
            token = token.strip()
            chain = token[0]
            resnum = int(token[1:])
            if chain != pep_chain:
                raise ValueError(f"Manual hotspot {token} does not match peptide chain {pep_chain}")
            resnums.append(resnum)
        pos_map = {pdb: pos for pos, pdb in enumerate(pdb_resnums, start=1)}
        positions = [pos_map[r] for r in resnums if r in pos_map]
        aas = "".join(seq[p - 1] for p in positions)
        return HotspotSelection(
            chain_id=pep_chain,
            pdb_resnums=resnums,
            sequence_positions=positions,
            amino_acids=aas,
            hotspot_res=",".join(tokens),
            n_hydrophobic=_count_hydrophobic(aas),
        )

    return select_peptide_hotspots(
        seq,
        pdb_resnums,
        pep_chain,
        min_hotspots=int(cfg.get("min_count", 5)),
        max_hotspots=int(cfg.get("max_count", 6)),
        min_hydrophobic=int(cfg.get("min_hydrophobic", 3)),
    )


def build_rfdiffusion_contig(
    pdb_path: str,
    binder_length_min: int,
    binder_length_max: int,
    peptide_chain: str = "auto",
    hla_chain: str = "auto",
) -> str:
    """
    Build RFdiffusion contig string fixing peptide + HLA target chains.

    Example: ``A1-9/0 B25-180/0 50-80``
    """
    pep_chain, hla_chain_id = infer_peptide_hla_chains(pdb_path, peptide_chain, hla_chain)
    pep_resnums = _chain_resnums(pdb_path, pep_chain)
    hla_resnums = _chain_resnums(pdb_path, hla_chain_id)
    if not pep_resnums or not hla_resnums:
        raise ValueError(f"Could not read peptide/HLA residue ranges from {pdb_path}")

    pep_span = f"{pep_chain}{pep_resnums[0]}-{pep_resnums[-1]}"
    hla_span = f"{hla_chain_id}{hla_resnums[0]}-{hla_resnums[-1]}"
    if binder_length_min == binder_length_max:
        binder_span = f"{binder_length_min}-{binder_length_max}"
    else:
        binder_span = f"{binder_length_min}-{binder_length_max}"
    return f"{pep_span}/0 {hla_span}/0 {binder_span}"
