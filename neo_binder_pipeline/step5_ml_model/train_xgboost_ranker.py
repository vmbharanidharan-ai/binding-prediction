"""Train XGBoost pairwise ranker on multi-modal peptide features."""

import argparse
import pickle
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from step5_ml_model.feature_engineering import CORE_FEATURES, get_feature_columns
from utils.logging import setup_logger
from utils.slurm_utils import load_config


def create_relevance_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create integer relevance labels for XGBoost rank:pairwise.

    Higher presentation confidence + tumor prevalence → higher relevance grade.
    Used only for training supervision, not pre-ML scoring.
    """
    scores = np.zeros(len(df))
    if "mhcflurry_presentation_percentile" in df.columns:
        scores += (100 - df["mhcflurry_presentation_percentile"].fillna(50)) / 100
    if "PSR_tumor" in df.columns:
        scores += df["PSR_tumor"].fillna(0)
    if "n_carriers_in_cohort" in df.columns:
        max_carriers = df["n_carriers_in_cohort"].max()
        if max_carriers > 0:
            scores += df["n_carriers_in_cohort"].fillna(0) / max_carriers

    # XGBoost ranker requires non-negative integer relevance grades
    if len(scores) == 0:
        return scores.astype(int)
    quantiles = np.quantile(scores, [0.25, 0.5, 0.75]) if len(scores) > 3 else [0, 0, 0]
    labels = np.zeros(len(scores), dtype=int)
    labels[scores > quantiles[0]] = 1
    labels[scores > quantiles[1]] = 2
    labels[scores > quantiles[2]] = 3
    return labels


def train_ranker(
    features_tsv: str,
    model_output: str,
    config_path: str = "config/config.yaml",
) -> xgb.XGBRanker:
    """Train XGBoost ranker with group-wise split to prevent leakage."""
    config = load_config(config_path)
    logger = setup_logger("step5_train", config["paths"]["logs_dir"])
    step_cfg = config["step5"]
    split_by = step_cfg["train_test_split_by"]

    df = pd.read_csv(features_tsv, sep="\t")
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).values
    y = create_relevance_labels(df)

    if split_by in df.columns:
        groups = df[split_by].values
    else:
        groups = df["peptide"].values

    unique_groups = np.unique(groups)
    group_ids = np.array([np.where(unique_groups == g)[0][0] for g in groups])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_groups = group_ids[train_idx]
    test_groups = group_ids[test_idx]

    # XGBRanker requires group sizes (not group IDs)
    def group_sizes(group_arr):
        unique, counts = np.unique(group_arr, return_counts=True)
        return counts.tolist()

    params = step_cfg["xgboost_params"]
    ranker = xgb.XGBRanker(
        objective=params["objective"],
        learning_rate=params["eta"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_estimators=params["n_estimators"],
        random_state=42,
    )

    ranker.fit(
        X_train,
        y_train,
        group=group_sizes(train_groups),
        eval_set=[(X_test, y_test)],
        eval_group=[group_sizes(test_groups)],
        verbose=True,
    )

    model_data = {
        "model": ranker,
        "feature_columns": feature_cols,
        "split_by": split_by,
        "config": step_cfg,
    }

    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as fh:
        pickle.dump(model_data, fh)

    logger.info(f"Model saved → {model_output}")
    logger.info(f"Features used: {len(feature_cols)}")
    return ranker


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost ranker")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="step5_ml_model/model.pkl")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.output or config["step5"]["model_path"]
    train_ranker(args.features, model_path, args.config)


if __name__ == "__main__":
    main()
