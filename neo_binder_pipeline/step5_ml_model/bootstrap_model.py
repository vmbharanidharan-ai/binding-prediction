"""Bootstrap a minimal model.pkl from sample input for pipeline testing."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from step5_ml_model.feature_engineering import engineer_features, get_feature_columns
from step5_ml_model.train_xgboost_ranker import train_ranker


def main():
    input_tsv = REPO_ROOT / "data" / "step5_input.tsv"
    features_tsv = REPO_ROOT / "work" / "bootstrap_features.tsv"
    model_path = REPO_ROOT / "step5_ml_model" / "model.pkl"

    features_tsv.parent.mkdir(parents=True, exist_ok=True)
    engineer_features(
        str(input_tsv),
        ranked_structures_tsv=None,
        binder_scores_tsv=None,
        esm2_tsv=None,
        prott5_tsv=None,
        output_tsv=str(features_tsv),
    )
    train_ranker(str(features_tsv), str(model_path))
    print(f"Bootstrap model saved: {model_path}")


if __name__ == "__main__":
    main()
