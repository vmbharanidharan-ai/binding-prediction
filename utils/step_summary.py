"""Print a human-readable summary of outputs after each pipeline step."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils.slurm_utils import load_config

STEP_OUTPUTS = {
    "embeddings": [
        ("ESM-2 embeddings", lambda c: Path(c["paths"]["embeddings_dir"]) / "esm2_embeddings.tsv"),
        ("ProtT5 embeddings", lambda c: Path(c["paths"]["embeddings_dir"]) / "prott5_embeddings.tsv"),
    ],
    "step1": [
        ("Input manifest", lambda c: Path(c["paths"]["inputs_dir"]) / "step1/input_manifest.tsv"),
        ("Structure PDBs", lambda c: Path(c["paths"]["step1_outputs"])),
        ("Parsed structures table", lambda c: Path(c["paths"]["step2_outputs"]) / "parsed_structures.tsv"),
    ],
    "step1_5": [
        ("Truncated structures table", lambda c: Path(c["paths"]["step1_5_outputs"]) / "truncated_structures.tsv"),
        ("Truncated PDBs", lambda c: Path(c["paths"]["step1_5_outputs"])),
    ],
    "step2": [
        ("Interface metrics", lambda c: Path(c["paths"]["step2_outputs"]) / "interface_metrics.tsv"),
        ("Clustered structures", lambda c: Path(c["paths"]["step2_outputs"]) / "clustered_structures.tsv"),
        ("Ranked structures", lambda c: Path(c["paths"]["step2_outputs"]) / "ranked_structures.tsv"),
    ],
    "step3": [
        ("Contig manifest", lambda c: Path(c["paths"]["step3_outputs"]) / "contigs/contig_manifest.tsv"),
        ("Binder designs", lambda c: Path(c["paths"]["step3_outputs"]) / "binder_designs.tsv"),
        ("Binder PDBs", lambda c: Path(c["paths"]["step3_outputs"])),
    ],
    "step4": [
        ("Complex manifest", lambda c: Path(c["paths"]["step4_outputs"]) / "complexes/complex_manifest.tsv"),
        ("Multimer outputs", lambda c: Path(c["paths"]["step4_outputs"]) / "multimer"),
        ("Binder scores", lambda c: Path(c["paths"]["step4_outputs"]) / "binder_scores.tsv"),
    ],
    "step5": [
        ("Feature matrix", lambda c: Path(c["paths"]["step5_outputs"]) / "features.tsv"),
        ("Trained model", lambda c: Path(c["step5"]["model_path"])),
        ("Final rankings", lambda c: Path(c["paths"]["step5_outputs"]) / "final_rankings.tsv"),
    ],
}

NEXT_STEP = {
    "embeddings": "sbatch slurm/step1.sbatch  (or run step1 in parallel)",
    "step1": "./slurm/submit_step.sh 1.5  (optional) or submit_step.sh 2",
    "step1_5": "./slurm/submit_step.sh 2",
    "step2": "sbatch slurm/step3.sbatch",
    "step3": "sbatch slurm/step4.sbatch",
    "step4": "sbatch slurm/step5.sbatch",
    "step5": "Done — inspect work/step5_ranked/final_rankings.tsv",
}


def _describe_path(path: Path) -> tuple[bool, str]:
    if path.suffix == ".tsv" and path.exists():
        if path.stat().st_size == 0:
            return True, "0 rows (empty file)"
        try:
            n = len(pd.read_csv(path, sep="\t"))
        except pd.errors.EmptyDataError:
            return True, "0 rows (empty file)"
        return True, f"{n} rows"
    if path.is_dir():
        pdbs = list(path.rglob("*.pdb"))
        tsvs = list(path.rglob("*.tsv"))
        if pdbs:
            return True, f"{len(pdbs)} PDB files"
        if tsvs:
            return True, f"{len(tsvs)} TSV files"
        return path.exists(), "directory"
    if path.exists():
        return True, "found"
    return False, "NOT FOUND"


def print_step_summary(step: str, config_path: str = "config/config.yaml") -> None:
    """Print output locations and counts for a completed step."""
    config = load_config(config_path)
    work_root = Path(config["paths"]["work_root"])

    print("")
    print("=" * 60)
    print(f"  STEP COMPLETE: {step.upper()}")
    print(f"  Work root: {work_root}")
    print("=" * 60)

    for label, path_fn in STEP_OUTPUTS.get(step, []):
        path = path_fn(config)
        exists, detail = _describe_path(path)
        status = "✓" if exists and detail != "NOT FOUND" else "✗"
        print(f"  {status} {label}")
        print(f"      {path}")
        print(f"      ({detail})")

    preview = {
        "step1": Path(config["paths"]["step2_outputs"]) / "parsed_structures.tsv",
        "step1_5": Path(config["paths"]["step1_5_outputs"]) / "truncated_structures.tsv",
        "step2": Path(config["paths"]["step2_outputs"]) / "ranked_structures.tsv",
        "step3": Path(config["paths"]["step3_outputs"]) / "binder_designs.tsv",
        "step4": Path(config["paths"]["step4_outputs"]) / "binder_scores.tsv",
        "step5": Path(config["paths"]["step5_outputs"]) / "final_rankings.tsv",
    }
    if step in preview and preview[step].exists() and preview[step].stat().st_size > 0:
        try:
            df = pd.read_csv(preview[step], sep="\t")
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if not df.empty:
            print("")
            print(f"  Preview ({preview[step].name}, first 5 rows):")
            print(df.head(5).to_string(index=False))

    print("")
    print(f"  Next: {NEXT_STEP.get(step, 'see README')}")
    print("=" * 60)
    print("")


def main():
    parser = argparse.ArgumentParser(description="Summarize pipeline step outputs")
    parser.add_argument("--step", required=True, choices=list(STEP_OUTPUTS.keys()))
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    print_step_summary(args.step, args.config)


if __name__ == "__main__":
    main()
