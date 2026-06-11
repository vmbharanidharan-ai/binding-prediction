"""Allele name conversion helpers."""

import re


def pipeline_to_mhcflurry_allele(allele: str) -> str:
    """
    Convert pipeline allele names to MHCflurry format.

    HLA_A0201 -> HLA-A*02:01
    HLA-A*02:01 -> unchanged
    """
    allele = allele.strip()
    if "*" in allele:
        return allele.replace("HLA_", "HLA-").upper()

    normalized = allele.upper().replace("HLA-", "HLA_")
    match = re.match(r"HLA_([ABC])(\d{2})(\d{2})", normalized)
    if match:
        gene, field1, field2 = match.groups()
        return f"HLA-{gene}*{field1}:{field2}"

    return allele.replace("_", "-")
