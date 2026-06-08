#!/usr/bin/env python3
"""Build allele → sequence index from IMGT HLA FASTA."""

import argparse
import pickle
import re
import sys
from pathlib import Path

import yaml

HLA_ROOT = Path(__file__).resolve().parent


def normalize_imgt_id(allele_id: str) -> str:
    """Normalize allele IDs to A*02:01:01:01 style."""
    allele_id = allele_id.strip()
    if allele_id.startswith("HLA:"):
        allele_id = allele_id[4:]
    m = re.match(r"^HLA-([ABC])\*(.+)$", allele_id, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}*{m.group(2)}"
    return allele_id


def parse_header_allele(header_line: str) -> str:
    """
    Extract allele name from an IMGT FASTA header.

    IMGT protein FASTA uses: >HLA:HLA00005 A*02:01:01:01 365 bp
    """
    tokens = header_line.strip().split()
    if not tokens:
        return ""
    if len(tokens) >= 2 and "*" in tokens[1]:
        return normalize_imgt_id(tokens[1])
    return normalize_imgt_id(tokens[0])


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse FASTA into {header_id: sequence}."""
    records: dict[str, str] = {}
    header = None
    seq_parts: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header:
                    records[header] = "".join(seq_parts)
                header = parse_header_allele(line[1:])
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            records[header] = "".join(seq_parts)
    return records


def two_field(allele_id: str) -> str:
    """A*02:01:01:01 → A*02:01"""
    if "*" not in allele_id:
        return allele_id
    gene, rest = allele_id.split("*", 1)
    fields = rest.split(":")
    if len(fields) >= 2:
        return f"{gene}*{fields[0]}:{fields[1]}"
    return allele_id


def pipeline_key(allele_id: str) -> str:
    """A*02:01 → HLA_A0201"""
    gene = allele_id.split("*")[0]
    rest = allele_id.split("*", 1)[1] if "*" in allele_id else ""
    rest = rest.replace(":", "")
    return f"HLA_{gene}{rest}"


def to_imgt_two_field(name: str) -> str:
    """HLA_A0201 or HLA-A*02:01 → A*02:01"""
    name = name.strip().upper().replace("HLA_", "HLA-")
    if "*" in name:
        name = name.replace("HLA-", "")
        parts = name.split("*", 1)
        gene = parts[0]
        fields = parts[1].split(":")
        if len(fields) >= 2:
            return f"{gene}*{fields[0]}:{fields[1]}"
        return f"{gene}*{parts[1]}"

    # HLA-A0201 style
    m = re.match(r"HLA-([ABC])(\d{2})(\d{2})", name)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"
    return name


def build_index(raw: dict[str, str]) -> dict[str, str]:
    """Build lookup index with IMGT, 2-field, and pipeline keys."""
    index: dict[str, str] = {}

    for allele_id, seq in raw.items():
        index[allele_id] = seq
        tf = two_field(allele_id)
        if tf not in index:
            index[tf] = seq
        pk = pipeline_key(tf)
        if pk not in index:
            index[pk] = seq

    return index


def main():
    parser = argparse.ArgumentParser(description="Build HLA allele index")
    parser.add_argument("--config", default=str(HLA_ROOT / "hla_config.yaml"))
    parser.add_argument("--fasta", default=None, help="Override input FASTA path")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fasta_path = Path(args.fasta) if args.fasta else HLA_ROOT / "data" / "imgt_hla_proteins.fasta"
    out_path = HLA_ROOT / "hla_index.pkl"

    if not fasta_path.exists():
        print(f"[ERROR] FASTA not found: {fasta_path}")
        print("Run: python hla/download_imgt.py")
        sys.exit(1)

    print(f"[INFO] Parsing {fasta_path}")
    raw = parse_fasta(fasta_path)
    print(f"[INFO] Parsed {len(raw)} IMGT entries")

    index = build_index(raw)
    print(f"[INFO] Built index with {len(index)} lookup keys")
    for probe in ("HLA_A0201", "A*02:01"):
        if probe in index:
            print(f"[INFO] Verified lookup key: {probe}")
        else:
            print(f"[WARN] Missing expected lookup key: {probe}")

    with open(out_path, "wb") as fh:
        pickle.dump(index, fh)
    print(f"[DONE] Saved → {out_path}")


if __name__ == "__main__":
    main()
