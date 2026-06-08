#!/usr/bin/env python3
"""
Neo Binder Pipeline — end-to-end orchestrator.

Runs all 5 stages with restart-safe checkpointing.
Embeddings run in parallel with structure steps (always included).
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.logging import setup_logger
from utils.slurm_utils import (
    ensure_work_dirs,
    load_config,
    read_last_checkpoint,
    write_checkpoint,
)


STEPS = [
    "embeddings",
    "step1",
    "step2",
    "step3",
    "step4",
    "step5",
]


def run_cmd(cmd: list, logger, dry_run: bool = False) -> None:
    """Execute a subprocess command with logging."""
    logger.info(f"Running: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_embeddings(config: dict, input_tsv: str, logger, dry_run: bool = False) -> None:
    """Stage 0: ESM-2 + ProtT5 embeddings (always run, independent of structure)."""
    emb_dir = config["paths"]["embeddings_dir"]
    esm2_out = f"{emb_dir}/esm2_embeddings.tsv"
    prott5_out = f"{emb_dir}/prott5_embeddings.tsv"

    run_cmd(
        ["python", "embeddings/esm2_embeddings.py", "--input", input_tsv, "--output", esm2_out],
        logger, dry_run,
    )
    run_cmd(
        ["python", "embeddings/prott5_embeddings.py", "--input", input_tsv, "--output", prott5_out],
        logger, dry_run,
    )


def run_step1(config: dict, input_tsv: str, logger, dry_run: bool = False) -> None:
    """Stage 1: Peptide–HLA structure generation via ColabFold."""
    paths = config["paths"]
    inputs_dir = f"{paths['inputs_dir']}/step1"
    structures_dir = paths["step1_outputs"]

    run_cmd(
        ["python", "step1_structure_generation/generate_inputs.py",
         "--input", input_tsv, "--output-dir", inputs_dir],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step1_structure_generation/run_colabfold.py",
         "--manifest", f"{inputs_dir}/input_manifest.tsv",
         "--output-dir", structures_dir],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step1_structure_generation/parse_outputs.py",
         "--structure-dir", structures_dir,
         "--manifest", f"{inputs_dir}/input_manifest.tsv",
         "--output", f"{paths['step2_outputs']}/parsed_structures.tsv"],
        logger, dry_run,
    )


def run_step2(config: dict, logger, dry_run: bool = False) -> None:
    """Stage 2: Structural scoring, clustering, and ranking."""
    paths = config["paths"]
    s2 = paths["step2_outputs"]

    run_cmd(
        ["python", "step2_structure_scoring/interface_metrics.py",
         "--input", f"{s2}/parsed_structures.tsv",
         "--output", f"{s2}/interface_metrics.tsv"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step2_structure_scoring/cluster_structures.py",
         "--input", f"{s2}/interface_metrics.tsv",
         "--output", f"{s2}/clustered_structures.tsv"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step2_structure_scoring/rank_structures.py",
         "--input", f"{s2}/clustered_structures.tsv",
         "--output", f"{s2}/ranked_structures.tsv"],
        logger, dry_run,
    )


def run_step3(config: dict, logger, dry_run: bool = False) -> None:
    """Stage 3: RFdiffusion minibinder design."""
    paths = config["paths"]
    s2 = paths["step2_outputs"]
    s3 = paths["step3_outputs"]

    run_cmd(
        ["python", "step3_rfdesign/build_contig_map.py",
         "--input", f"{s2}/ranked_structures.tsv",
         "--output-dir", f"{s3}/contigs"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step3_rfdesign/run_rfdiffusion.py",
         "--manifest", f"{s3}/contigs/contig_manifest.tsv",
         "--output-dir", s3],
        logger, dry_run,
    )


def run_step4(config: dict, logger, dry_run: bool = False) -> None:
    """Stage 4: Binder validation via AlphaFold-Multimer."""
    paths = config["paths"]
    s3 = paths["step3_outputs"]
    s4 = paths["step4_outputs"]

    run_cmd(
        ["python", "step4_binder_validation/build_complexes.py",
         "--binders", f"{s3}/binder_designs.tsv",
         "--contigs", f"{s3}/contigs/contig_manifest.tsv",
         "--output-dir", f"{s4}/complexes"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step4_binder_validation/run_colabfold_multimer.py",
         "--manifest", f"{s4}/complexes/complex_manifest.tsv",
         "--output-dir", f"{s4}/multimer"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step4_binder_validation/binder_scoring.py",
         "--multimer-dir", f"{s4}/multimer",
         "--manifest", f"{s4}/complexes/complex_manifest.tsv",
         "--output", f"{s4}/binder_scores.tsv"],
        logger, dry_run,
    )


def run_step5(config: dict, input_tsv: str, logger, dry_run: bool = False) -> None:
    """Stage 5: ML ranking with XGBoost."""
    paths = config["paths"]
    s2 = paths["step2_outputs"]
    s4 = paths["step4_outputs"]
    s5 = paths["step5_outputs"]
    emb = paths["embeddings_dir"]

    features_out = f"{s5}/features.tsv"
    ranked_out = f"{s5}/final_rankings.tsv"

    run_cmd(
        ["python", "step5_ml_model/feature_engineering.py",
         "--input", input_tsv,
         "--output", features_out,
         "--ranked", f"{s2}/ranked_structures.tsv",
         "--binders", f"{s4}/binder_scores.tsv",
         "--esm2", f"{emb}/esm2_embeddings.tsv",
         "--prott5", f"{emb}/prott5_embeddings.tsv"],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step5_ml_model/train_xgboost_ranker.py",
         "--features", features_out,
         "--output", config["step5"]["model_path"]],
        logger, dry_run,
    )
    run_cmd(
        ["python", "step5_ml_model/inference.py",
         "--features", features_out,
         "--model", config["step5"]["model_path"],
         "--output", ranked_out],
        logger, dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Neo Binder Pipeline — peptide–HLA binder design"
    )
    parser.add_argument("--input", default="data/step5_input.tsv", help="Input TSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--steps", nargs="+", choices=STEPS, default=STEPS,
        help="Pipeline steps to run",
    )
    parser.add_argument("--from-step", default=None, help="Resume from this step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_work_dirs(config)
    logger = setup_logger("pipeline", config["paths"]["logs_dir"])

    checkpoint_path = f"{config['paths']['work_root']}/checkpoint.tsv"
    steps_to_run = list(args.steps)

    if args.from_step:
        idx = STEPS.index(args.from_step)
        steps_to_run = STEPS[idx:]

    step_fns = {
        "embeddings": lambda: run_embeddings(config, args.input, logger, args.dry_run),
        "step1": lambda: run_step1(config, args.input, logger, args.dry_run),
        "step2": lambda: run_step2(config, logger, args.dry_run),
        "step3": lambda: run_step3(config, logger, args.dry_run),
        "step4": lambda: run_step4(config, logger, args.dry_run),
        "step5": lambda: run_step5(config, args.input, logger, args.dry_run),
    }

    logger.info(f"Starting pipeline: {steps_to_run}")
    for step in steps_to_run:
        logger.info(f"=== {step.upper()} ===")
        try:
            step_fns[step]()
            write_checkpoint(checkpoint_path, step, "completed")
        except Exception as e:
            write_checkpoint(checkpoint_path, step, f"failed: {e}")
            logger.error(f"Step {step} failed: {e}")
            raise

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
