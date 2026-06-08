#!/usr/bin/env python3
"""Resolve HLA allele names to protein sequences via IMGT index."""

import argparse
import pickle
import re
import sys
from pathlib import Path

import yaml

HLA_ROOT = Path(__file__).resolve().parent


class HLAResolver:
    """Resolve pipeline or IMGT allele names to protein sequences."""

    def __init__(self, index_path: Path, config: dict | None = None):
        self.config = config or {}
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"HLA index not found: {self.index_path}\n"
                "Run: python hla/setup_hla.py"
            )
        with open(self.index_path, "rb") as fh:
            self.hla_index: dict[str, str] = pickle.load(fh)

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "HLAResolver":
        config_path = Path(config_path or HLA_ROOT / "hla_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        return cls(HLA_ROOT / "hla_index.pkl", cfg)

    def _normalize_candidates(self, allele: str) -> list[str]:
        allele = allele.strip()
        candidates = [
            allele,
            allele.upper(),
            allele.replace("_", "-"),
            allele.upper().replace("_", "-"),
            self._to_imgt_two_field(allele),
            self._to_pipeline_key(allele),
        ]
        seen = set()
        unique = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _to_imgt_two_field(self, allele: str) -> str:
        allele = allele.strip().upper().replace("HLA_", "HLA-")
        if "*" in allele:
            allele = allele.replace("HLA-", "")
            gene, rest = allele.split("*", 1)
            fields = rest.split(":")
            if len(fields) >= 2:
                return f"{gene}*{fields[0]}:{fields[1]}"
            return f"{gene}*{rest}"

        m = re.match(r"HLA-?([ABC])(\d{2})(\d{2})", allele.replace("HLA-", "HLA"))
        if m:
            return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"
        m2 = re.match(r"^([ABC])(\d{2})(\d{2})$", allele.replace("HLA-", ""))
        if m2:
            return f"{m2.group(1)}*{m2.group(2)}:{m2.group(3)}"
        return allele

    def _to_pipeline_key(self, allele: str) -> str:
        tf = self._to_imgt_two_field(allele)
        if "*" not in tf:
            return allele.upper()
        gene = tf.split("*")[0]
        code = tf.split("*", 1)[1].replace(":", "")
        return f"HLA_{gene}{code}"

    def _closest_match(self, allele: str) -> str | None:
        target = self._to_imgt_two_field(allele)
        if "*" not in target:
            return None
        gene = target.split("*")[0]
        prefix = target  # e.g. A*02:01

        best = None
        for key in self.hla_index:
            if key.startswith(f"{gene}*") and key.startswith(prefix[:4]):
                if best is None or len(key) < len(best):
                    best = key
            elif key.startswith(f"HLA_{gene}") and prefix.replace("*", "").replace(":", "") in key:
                best = key
                break

        if best:
            return self.hla_index[best]

        # Prefix match on 2-field
        gene_prefix = f"{gene}*"
        matches = [k for k in self.hla_index if k.startswith(gene_prefix)]
        if not matches:
            return None
        tf_fields = prefix.split("*", 1)[1].split(":")
        for m in sorted(matches):
            if f"{tf_fields[0]}:" in m:
                return self.hla_index[m]
        return self.hla_index[matches[0]]

    def resolve(self, allele: str) -> str:
        """
        Resolve allele name to protein sequence.

        Accepts: HLA_A0201, HLA-A*02:01, A*02:01
        """
        for candidate in self._normalize_candidates(allele):
            if candidate in self.hla_index:
                return self.hla_index[candidate]

        strategy = self.config.get("fallback_strategy", "closest_match")
        if strategy == "closest_match":
            seq = self._closest_match(allele)
            if seq:
                return seq

        raise ValueError(
            f"HLA allele not found: {allele}. "
            f"Tried: {self._normalize_candidates(allele)}. "
            "Run python hla/setup_hla.py to refresh the IMGT index."
        )

    def has_allele(self, allele: str) -> bool:
        try:
            self.resolve(allele)
            return True
        except ValueError:
            return False


def get_resolver(config_path: str | Path | None = None) -> HLAResolver:
    """Load resolver, building index first if missing."""
    if str(HLA_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(HLA_ROOT.parent))
    from utils.hla_helper import ensure_hla_index

    ensure_hla_index(auto_setup=True)
    return HLAResolver.from_config(config_path)


def main():
    parser = argparse.ArgumentParser(description="Test HLA resolver")
    parser.add_argument("allele", nargs="?", default="HLA_A0201")
    args = parser.parse_args()

    resolver = get_resolver()
    seq = resolver.resolve(args.allele)
    print(f"Allele: {args.allele}")
    print(f"Length: {len(seq)} aa")
    print(f"Sequence preview: {seq[:60]}...")


if __name__ == "__main__":
    main()
