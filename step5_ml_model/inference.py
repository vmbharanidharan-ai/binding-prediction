"""Run XGBoost ranker inference on engineered feature matrix."""

import argparse
import pickle
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import load_config


def run_inference(
    features_tsv: str,
    model_path: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Score and rank peptide–allele–binder candidates.

    Output columns: peptide, allele, binder_score, rank, confidence
    """
    config = load_config(config_path)
    logger = setup_logger("step5_inference", config["paths"]["logs_dir"])

    with open(model_path, "rb") as fh:
        model_data = pickle.load(fh)

    ranker = model_data["model"]
    feature_cols = model_data["feature_columns"]

    df = pd.read_csv(features_tsv, sep="\t")
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].fillna(0).values
    scores = ranker.predict(X)

    result = df[["peptide", "allele"]].copy()
    if "gene" in df.columns:
        result["gene"] = df["gene"]
    if "junction" in df.columns:
        result["junction"] = df["junction"]

    result["binder_score"] = scores
    result["rank"] = result["binder_score"].rank(ascending=False, method="dense").astype(int)

    score_range = result["binder_score"].max() - result["binder_score"].min()
    if score_range > 0:
        result["confidence"] = (result["binder_score"] - result["binder_score"].min()) / score_range
    else:
        result["confidence"] = 0.5

    result = result.sort_values("rank").reset_index(drop=True)

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Ranked {len(result)} candidates → {output_tsv}")
    return result


def main():
    parser = argparse.ArgumentParser(description="XGBoost ranker inference")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", default="step5_ml_model/model.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    run_inference(args.features, args.model, args.output, args.config)


if __name__ == "__main__":
    main()
