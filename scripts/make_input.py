#!/usr/bin/env python3
"""Create a pipeline input TSV from CLI args or interactive prompts."""

import argparse
import sys
from pathlib import Path

COLUMNS = [
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

# Grouped for interactive prompts — collected upfront for Step 5 ML ranking
FIELD_GROUPS = {
    "core": {
        "title": "CORE (required for all steps)",
        "fields": ["peptide", "allele"],
    },
    "biology": {
        "title": "BIOLOGICAL METADATA (for Step 5 ML ranking — from neoJunction Step 5)",
        "fields": [
            "gene",
            "junction",
            "mhcflurry_presentation_percentile",
            "netmhcpan_EL_rank",
            "n_carriers_in_cohort",
            "PSR_tumor",
            "frameshift_flag",
        ],
    },
}

PROMPTS = {
    "peptide": "Peptide sequence",
    "allele": "HLA allele (e.g. HLA_A0201 or HLA-A*02:01)",
    "gene": "Source gene",
    "junction": "Junction ID (used for ML train/test split)",
    "mhcflurry_presentation_percentile": "MHCflurry presentation percentile (lower = stronger)",
    "netmhcpan_EL_rank": "NetMHCpan EL rank (lower = stronger binder)",
    "n_carriers_in_cohort": "Number of patients carrying this neoantigen",
    "PSR_tumor": "Proportion of tumor cells expressing variant (0–1)",
    "frameshift_flag": "Frameshift-derived? (0=no, 1=yes)",
}

DEFAULTS = {
    "gene": "TEST",
    "junction": "J1",
    "mhcflurry_presentation_percentile": "1.0",
    "netmhcpan_EL_rank": "0.5",
    "n_carriers_in_cohort": "1",
    "PSR_tumor": "0.5",
    "frameshift_flag": "0",
}

CURRENT_PAIR_ENV = Path("data/generated/current_pair.env")


def normalize_allele(allele: str) -> str:
    allele = allele.strip().upper()
    if allele.startswith("HLA_"):
        return allele
    return allele.replace("HLA-", "HLA_").replace("*", "").replace(":", "")


def prompt_value(field: str, required: bool = False) -> str:
    label = PROMPTS[field]
    default = DEFAULTS.get(field, "")
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"  {label}{suffix}: ").strip()
        if not value and default:
            return default
        if value or not required:
            return value
        print("    Required — please enter a value.")


def prompt_interactive(include_biology: bool = True) -> dict:
    """Prompt for all input fields, grouped by section."""
    row = {}

    print("")
    print("=" * 60)
    print("  PIPELINE INPUT — enter all details now (saved for all steps)")
    print("=" * 60)

    group = FIELD_GROUPS["core"]
    print(f"\n── {group['title']} ──")
    for field in group["fields"]:
        row[field] = prompt_value(field, required=True)

    if include_biology:
        group = FIELD_GROUPS["biology"]
        print(f"\n── {group['title']} ──")
        print("  (Press Enter to accept defaults; used later in Step 5 ranking)")
        for field in group["fields"]:
            row[field] = prompt_value(field)

    row["allele"] = normalize_allele(row["allele"])
    row["peptide"] = row["peptide"].strip().upper()
    return row


def build_row(args: argparse.Namespace) -> dict:
    if args.interactive:
        return prompt_interactive(include_biology=not args.core_only)

    row = {col: getattr(args, col) or DEFAULTS.get(col, "") for col in COLUMNS}

    # If peptide/allele given on CLI but metadata flags missing, prompt biology section
    if args.prompt_metadata:
        print("\nPeptide and allele provided. Enter Step 5 metadata:")
        for field in FIELD_GROUPS["biology"]["fields"]:
            if getattr(args, field, None) is None:
                row[field] = prompt_value(field)

    row["allele"] = normalize_allele(row["allele"])
    row["peptide"] = row["peptide"].strip().upper()
    return row


def write_tsv(row: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(COLUMNS)
    values = "\t".join(str(row[c]) for c in COLUMNS)
    output.write_text(f"{header}\n{values}\n")
    return output


def save_current_pair_env(tsv_path: Path) -> None:
    """Persist input path so later steps reuse the same TSV without re-prompting."""
    CURRENT_PAIR_ENV.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PAIR_ENV.write_text(f'INPUT_TSV="{tsv_path.resolve()}"\n')


def print_summary(row: dict, out: Path) -> None:
    print("")
    print("=" * 60)
    print("  INPUT SAVED")
    print("=" * 60)
    print(f"  File:    {out}")
    print(f"  Peptide: {row['peptide']}")
    print(f"  Allele:  {row['allele']}")
    print("")
    print("  Step 5 metadata:")
    for field in FIELD_GROUPS["biology"]["fields"]:
        print(f"    {field}: {row[field]}")
    print("")
    print(f"  Reuse in later steps: source {CURRENT_PAIR_ENV}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create a single peptide–HLA input TSV for the binder pipeline."
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Prompt for all fields (core + Step 5 metadata)",
    )
    parser.add_argument(
        "--prompt-metadata", action="store_true",
        help="Prompt for Step 5 metadata even when peptide/allele given on CLI",
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help="Interactive mode: only ask peptide and allele",
    )
    parser.add_argument("--peptide", "-p", help="Peptide sequence")
    parser.add_argument("--allele", "-a", help="HLA allele")
    parser.add_argument("--gene", default=None)
    parser.add_argument("--junction", default=None)
    parser.add_argument("--mhcflurry", dest="mhcflurry_presentation_percentile", default=None)
    parser.add_argument("--netmhcpan", dest="netmhcpan_EL_rank", default=None)
    parser.add_argument("--carriers", dest="n_carriers_in_cohort", default=None)
    parser.add_argument("--psr", dest="PSR_tumor", default=None)
    parser.add_argument("--frameshift", dest="frameshift_flag", default=None)
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output TSV path (default: data/generated/<peptide>_<allele>.tsv)",
    )
    args = parser.parse_args()

    if not args.interactive and (not args.peptide or not args.allele):
        parser.error("Provide --peptide and --allele, or use --interactive")

    row = build_row(args)
    out = Path(args.output) if args.output else Path("data/generated") / f"{row['peptide']}_{row['allele']}.tsv"

    write_tsv(row, out)
    save_current_pair_env(out)
    print_summary(row, out)

    # Validate allele against IMGT index if available
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from utils.hla_helper import resolve_hla_sequence
        from utils.slurm_utils import load_config

        cfg = load_config(str(repo_root / "config" / "config.yaml"))
        resolved = resolve_hla_sequence(row["allele"], cfg)
        if resolved:
            _, seq = resolved
            print(f"  HLA sequence: resolved ({len(seq)} aa) via IMGT index ✓")
        else:
            print("  WARNING: allele not found in IMGT index or local FASTA")
    except Exception as exc:
        print(f"  HLA check skipped: {exc}")

    # Machine-readable path for shell scripts
    print(f"Input TSV written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
