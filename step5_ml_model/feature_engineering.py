"""Feature engineering for XGBoost ranker — biological + structural + embeddings."""

import argparse
from pathlib import Path
from typing import List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.logging import setup_logger
from utils.slurm_utils import load_config

CORE_FEATURES = [
    "mhcflurry_presentation_percentile",
    "netmhcpan_EL_rank",
    "n_carriers_in_cohort",
    "PSR_tumor",
    "frameshift_flag",
]

STRUCTURAL_FEATURES = [
    "structure_confidence_score",
    "binder_structural_score",
    "interface_plddt_mean",
    "interface_pae_mean",
    "contact_count",
    "buried_surface_area",
    "binder_interface_plddt",
    "binder_interface_pae",
    "binder_contact_count",
]

# Added after Step 2 post-ranking extras. Model retraining on real cohort data
# is recommended once these features are validated on Longleaf.
ROSETTA_FEATURES = [
    "dG_separated",
    "dSASA_int",
    "delta_unsatHbonds",
]

MHCFLURRY_BA_FEATURES = [
    "mhcflurry_ic50_log10",
    "mhcflurry_affinity_percentile",
    "mhcflurry_presentation_score",
]


def parse_embedding_column(series: pd.Series) -> np.ndarray:
    """Parse comma-separated embedding strings into a 2D array."""
    vectors = []
    for val in series:
        if pd.isna(val) or val == "":
            vectors.append(np.zeros(1280))
        else:
            vectors.append(np.array([float(x) for x in str(val).split(",")]))
    return np.array(vectors)


def _load_optional_pair_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    return df


def _drop_all_na_feature_columns(
    df: pd.DataFrame,
    columns: list[str],
    logger,
    label: str,
) -> list[str]:
    kept = []
    for col in columns:
        if col not in df.columns:
            logger.warning(f"{label}: column {col} missing — skipping.")
            continue
        if df[col].isna().all():
            logger.warning(f"{label}: column {col} is all NA — dropping from features.")
            df.drop(columns=[col], inplace=True)
            continue
        kept.append(col)
    return kept


def merge_post_ranking_features(
    df: pd.DataFrame,
    work_root: Path,
    logger,
) -> pd.DataFrame:
    """Merge Rosetta interface and MHCflurry BA TSVs produced after Step 2."""
    if "job_id" not in df.columns:
        df["job_id"] = df.apply(
            lambda r: f"{r['peptide']}_{r['allele'].replace('HLA-', 'HLA_').replace('*', '').replace(':', '')}",
            axis=1,
        )

    rosetta_rows = []
    mhcflurry_rows = []
    for pair_id in df["job_id"].astype(str).unique():
        rosetta_path = work_root / "rosetta_interface" / f"{pair_id}_rosetta.tsv"
        mhcflurry_path = work_root / "mhcflurry_ba" / f"{pair_id}_mhcflurry_ba.tsv"

        rosetta_df = _load_optional_pair_tsv(rosetta_path)
        if rosetta_df is not None:
            rosetta_rows.append(rosetta_df)

        mhcflurry_df = _load_optional_pair_tsv(mhcflurry_path)
        if mhcflurry_df is not None:
            if "mhcflurry_ic50_nm" in mhcflurry_df.columns:
                mhcflurry_df = mhcflurry_df.copy()
                mhcflurry_df["mhcflurry_ic50_log10"] = np.log10(
                    mhcflurry_df["mhcflurry_ic50_nm"].fillna(0) + 1
                )
            mhcflurry_rows.append(mhcflurry_df)

    if rosetta_rows:
        rosetta_all = pd.concat(rosetta_rows, ignore_index=True)
        rosetta_by_pair = rosetta_all.set_index("pair_id")
        for col in ROSETTA_FEATURES:
            if col in rosetta_by_pair.columns:
                df[col] = df["job_id"].map(rosetta_by_pair[col])
    else:
        logger.warning("No Rosetta interface TSVs found under work/rosetta_interface/.")

    if mhcflurry_rows:
        mhcflurry_all = pd.concat(mhcflurry_rows, ignore_index=True)
        mhcflurry_by_pair = mhcflurry_all.set_index("pair_id")
        for col in MHCFLURRY_BA_FEATURES + ["mhcflurry_ic50_nm"]:
            if col in mhcflurry_by_pair.columns:
                df[col] = df["job_id"].map(mhcflurry_by_pair[col])
    else:
        logger.warning("No MHCflurry BA TSVs found under work/mhcflurry_ba/.")

    for col in ROSETTA_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    _drop_all_na_feature_columns(df, ROSETTA_FEATURES, logger, "Rosetta")
    _drop_all_na_feature_columns(df, MHCFLURRY_BA_FEATURES, logger, "MHCflurry BA")

    return df


def apply_pca(
    embeddings: np.ndarray, n_components: int = 50, fit: bool = True, pca: Optional[PCA] = None
) -> tuple:
    """Reduce embedding dimensionality via PCA."""
    if pca is None and fit:
        pca = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))
        reduced = pca.fit_transform(embeddings)
    elif pca is not None:
        reduced = pca.transform(embeddings)
    else:
        reduced = embeddings[:, :n_components]
        pca = None
    return reduced, pca


def engineer_features(
    input_tsv: str,
    ranked_structures_tsv: Optional[str],
    binder_scores_tsv: Optional[str],
    esm2_tsv: Optional[str],
    prott5_tsv: Optional[str],
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Merge all feature sources into a single training/inference matrix.

    Embeddings are always included even if structure steps failed.
    """
    config = load_config(config_path)
    logger = setup_logger("step5_features", config["paths"]["logs_dir"])
    n_pca = config["step5"]["pca_components"]

    df = pd.read_csv(input_tsv, sep="\t")

    if ranked_structures_tsv and Path(ranked_structures_tsv).exists():
        ranked = pd.read_csv(ranked_structures_tsv, sep="\t")
        ranked_best = ranked.sort_values("structure_rank").groupby("job_id").first().reset_index()
        ranked_best["job_id"] = ranked_best.apply(
            lambda r: f"{r['peptide']}_{r['allele'].replace('HLA-', 'HLA_').replace('*','').replace(':','')}",
            axis=1,
        )
        df["job_id"] = df.apply(
            lambda r: f"{r['peptide']}_{r['allele'].replace('HLA-', 'HLA_').replace('*','').replace(':','')}",
            axis=1,
        )
        df = df.merge(
            ranked_best[["job_id"] + [c for c in STRUCTURAL_FEATURES if c in ranked_best.columns]],
            on="job_id",
            how="left",
        )

    if binder_scores_tsv and Path(binder_scores_tsv).exists():
        binder = pd.read_csv(binder_scores_tsv, sep="\t")
        binder_best = binder.groupby("peptide").first().reset_index()
        binder_cols = [c for c in binder.columns if c.startswith("binder_")]
        df = df.merge(binder_best[["peptide"] + binder_cols], on="peptide", how="left")

    if esm2_tsv and Path(esm2_tsv).exists():
        esm2_df = pd.read_csv(esm2_tsv, sep="\t")
        esm2_cols = ["peptide", "esm2_embedding"]
        if "iedb_cosine_similarity_max" in esm2_df.columns:
            esm2_cols.append("iedb_cosine_similarity_max")
        df = df.merge(esm2_df[esm2_cols], on="peptide", how="left")

    if prott5_tsv and Path(prott5_tsv).exists():
        prott5_df = pd.read_csv(prott5_tsv, sep="\t")
        df = df.merge(prott5_df[["peptide", "prott5_embedding"]], on="peptide", how="left")

    df = merge_post_ranking_features(df, Path(config["paths"]["work_root"]), logger)

    # PCA on ESM-2 embeddings
    if "esm2_embedding" in df.columns:
        esm2_matrix = parse_embedding_column(df["esm2_embedding"])
        esm2_pca, _ = apply_pca(esm2_matrix, n_pca)
        for i in range(esm2_pca.shape[1]):
            df[f"esm2_pca_{i}"] = esm2_pca[:, i]

    if "prott5_embedding" in df.columns:
        prott5_matrix = parse_embedding_column(df["prott5_embedding"])
        if prott5_matrix.shape[1] > 1:
            prott5_pca, _ = apply_pca(prott5_matrix, min(n_pca, prott5_matrix.shape[1]))
            for i in range(prott5_pca.shape[1]):
                df[f"prott5_pca_{i}"] = prott5_pca[:, i]

    # Fill missing structural features with NaN (embeddings still usable)
    for col in CORE_FEATURES + STRUCTURAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Feature matrix: {df.shape} → {output_tsv}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered list of numeric feature columns for ML."""
    cols = [c for c in CORE_FEATURES if c in df.columns]
    cols += [c for c in STRUCTURAL_FEATURES if c in df.columns]
    cols += [c for c in ROSETTA_FEATURES if c in df.columns and not df[c].isna().all()]
    cols += [c for c in MHCFLURRY_BA_FEATURES if c in df.columns and not df[c].isna().all()]
    cols += [c for c in df.columns if c.startswith("esm2_pca_")]
    cols += [c for c in df.columns if c.startswith("prott5_pca_")]
    if "iedb_cosine_similarity_max" in df.columns:
        cols.append("iedb_cosine_similarity_max")
    return cols


def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ranked", default=None)
    parser.add_argument("--binders", default=None)
    parser.add_argument("--esm2", default=None)
    parser.add_argument("--prott5", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    engineer_features(
        args.input,
        args.ranked,
        args.binders,
        args.esm2,
        args.prott5,
        args.output,
        args.config,
    )


if __name__ == "__main__":
    main()
