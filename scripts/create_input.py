#!/usr/bin/env python3
"""
Interactive script to create input.tsv for the neo_binder pipeline.
Prompts for peptide–allele rows and writes a cohort TSV.

Usage:
  source $PROJECT_ROOT/.env && python scripts/create_input.py
  python scripts/create_input.py --output data/generated/my_cohort.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "peptide",
    "allele",
    "gene",
    "junction",
    "mhcflurry_presentation_percentile",
    "netmhcpan_EL_rank",
    "n_carriers_in_cohort",
    "PSR_tumor",
    "frameshift_flag",
]

CURRENT_PAIR_ENV = Path("data/generated/current_pair.env")


def default_output_path() -> Path:
    project_root = os.environ.get("PROJECT_ROOT", "").strip()
    if project_root:
        return Path(project_root) / "binding-prediction" / "data/generated/input.tsv"
    return Path("data/generated/input.tsv")


def save_input_env(tsv_path: Path) -> None:
    """Persist INPUT_TSV for submit_step.sh and SLURM jobs."""
    CURRENT_PAIR_ENV.parent.mkdir(parents=True, exist_ok=True)
    resolved = tsv_path.resolve()
    CURRENT_PAIR_ENV.write_text(f'INPUT_TSV="{resolved}"\n')


class InputBuilder:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.rows: List[Dict] = []

    def prompt_field(self, field: str, hint: str = "", default: str | None = None) -> str:
        prompt = field
        if hint:
            prompt += f" ({hint})"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        return value

    def validate_peptide(self, pep: str) -> bool:
        if not pep:
            print("  ✗ Peptide cannot be empty")
            return False
        if not pep.isupper() or not pep.isalpha():
            print("  ✗ Peptide must be uppercase letters only")
            return False
        return True

    def validate_allele(self, allele: str) -> str | None:
        if not allele:
            print("  ✗ Allele cannot be empty")
            return None
        return allele.upper().replace("-", "_").replace("*", "").replace(":", "")

    def validate_percentile(self, value: str, field: str) -> float | None:
        try:
            val = float(value)
            if val < 0 or val > 100:
                print(f"  ✗ {field} should be 0–100 (or 0–1 for fractions)")
                return None
            return val
        except ValueError:
            print(f"  ✗ {field} must be numeric")
            return None

    def validate_integer(self, value: str, field: str, min_val: int = 0) -> int | None:
        try:
            val = int(value)
            if val < min_val:
                print(f"  ✗ {field} must be >= {min_val}")
                return None
            return val
        except ValueError:
            print(f"  ✗ {field} must be an integer")
            return None

    def validate_binary(self, value: str, field: str) -> int | None:
        try:
            val = int(value)
            if val not in (0, 1):
                print(f"  ✗ {field} must be 0 or 1")
                return None
            return val
        except ValueError:
            print(f"  ✗ {field} must be 0 or 1")
            return None

    def prompt_row(self, row_num: int = 1) -> Dict | None:
        print(f"\n--- Peptide {row_num} ---")
        row: Dict = {}

        while True:
            peptide = self.prompt_field("Peptide", "e.g., AIMDLVMMV (uppercase)").upper()
            if self.validate_peptide(peptide):
                row["peptide"] = peptide
                break

        while True:
            allele = self.prompt_field("Allele", "e.g., HLA_A0201 or HLA-A*02:01")
            validated = self.validate_allele(allele)
            if validated:
                row["allele"] = validated
                break

        row["gene"] = self.prompt_field("Gene", "e.g., SF3B1", default="TEST")
        row["junction"] = self.prompt_field("Junction", "e.g., J1", default="J1")

        while True:
            val = self.prompt_field("MHCFlurry presentation %ile", "0–100 (lower=stronger)", default="0.5")
            pct = self.validate_percentile(val, "MHCFlurry")
            if pct is not None:
                row["mhcflurry_presentation_percentile"] = pct
                break

        while True:
            val = self.prompt_field("NetMHCpan EL rank", "0–1 (lower=stronger)", default="0.2")
            pct = self.validate_percentile(val, "NetMHCpan")
            if pct is not None:
                row["netmhcpan_EL_rank"] = pct
                break

        while True:
            val = self.prompt_field("n_carriers_in_cohort", "integer >= 0", default="1")
            carriers = self.validate_integer(val, "n_carriers_in_cohort")
            if carriers is not None:
                row["n_carriers_in_cohort"] = carriers
                break

        while True:
            val = self.prompt_field("PSR_tumor", "0–1 (tumor/self ratio)", default="0.5")
            psr = self.validate_percentile(val, "PSR_tumor")
            if psr is not None:
                row["PSR_tumor"] = psr
                break

        while True:
            val = self.prompt_field("frameshift_flag", "0=no, 1=yes", default="0")
            fs = self.validate_binary(val, "frameshift_flag")
            if fs is not None:
                row["frameshift_flag"] = fs
                break

        return row

    def interactive_build(self) -> bool:
        print("=" * 60)
        print("Neo-binder Input TSV Builder")
        print("=" * 60)
        print(f"Output: {self.output_path}\n")

        row_num = 1
        while True:
            row = self.prompt_row(row_num)
            if row:
                self.rows.append(row)
                print(f"  ✓ Added peptide {row_num}")
                row_num += 1

            more = input("\nAdd another peptide? (y/n) [y]: ").strip().lower()
            if more in ("n", "no"):
                break

        if not self.rows:
            print("\n✗ No peptides added. Exiting.")
            return False

        return self.save()

    def save(self) -> bool:
        if not self.rows:
            print("✗ No rows to save")
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)[REQUIRED_COLUMNS]
        df.to_csv(self.output_path, sep="\t", index=False)
        save_input_env(self.output_path)

        resolved = self.output_path.resolve()
        print(f"\n✓ Saved {len(self.rows)} peptide(s) to:\n  {resolved}\n")
        print("Summary:")
        print(df.to_string(index=False))
        print("\nNext steps:")
        print(f"  export INPUT_TSV={resolved}")
        print("  ./slurm/submit_step.sh 0")
        print("  ./slurm/submit_step.sh 1")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively create input.tsv for the neo_binder pipeline"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output TSV path (default: $PROJECT_ROOT/binding-prediction/data/generated/input.tsv)",
        default=None,
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else default_output_path()
    builder = InputBuilder(output)
    return 0 if builder.interactive_build() else 1


if __name__ == "__main__":
    sys.exit(main())
