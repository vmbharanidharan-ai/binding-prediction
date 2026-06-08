"""FASTA parsing and writing utilities."""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_fasta(path: str) -> Dict[str, str]:
    """Read a FASTA file into a {header: sequence} dict."""
    records = {}
    for record in SeqIO.parse(path, "fasta"):
        records[record.id] = str(record.seq)
    return records


def write_fasta(records: List[Tuple[str, str]], path: str) -> None:
    """Write (header, sequence) pairs to a FASTA file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    seq_records = [
        SeqRecord(Seq(seq), id=header, description="")
        for header, seq in records
    ]
    SeqIO.write(seq_records, path, "fasta")


def write_single_fasta(header: str, sequence: str, path: str) -> None:
    """Write a single sequence to FASTA."""
    write_fasta([(header, sequence)], path)


def iter_fasta_batches(
    records: List[Tuple[str, str]], batch_size: int
) -> Iterator[List[Tuple[str, str]]]:
    """Yield FASTA records in batches."""
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def normalize_allele_name(allele: str) -> str:
    """Convert HLA-A*02:01 style names to HLA_A0201."""
    allele = allele.strip().upper()
    if allele.startswith("HLA_"):
        return allele
    allele = allele.replace("HLA-", "HLA_").replace("*", "").replace(":", "")
    return allele
