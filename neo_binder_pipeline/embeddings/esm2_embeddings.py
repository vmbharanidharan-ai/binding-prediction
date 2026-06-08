"""ESM-2 embedding extraction via HuggingFace transformers."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.logging import setup_logger
from utils.slurm_utils import append_tsv, load_config


def load_esm2_model(model_name: str, device: str = "cuda"):
    """Load ESM-2 model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_sequences(
    sequences: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Generate mean-pooled ESM-2 embeddings (1280-dim) for a list of sequences."""
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2 embedding"):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

        for j, seq in enumerate(batch):
            seq_len = len(seq)
            emb = hidden[j, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
            all_embeddings.append(emb)

    return np.array(all_embeddings)


def compute_iedb_cosine_similarity(
    query_embeddings: np.ndarray,
    iedb_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute max cosine similarity between query peptides and IEDB reference."""
    query_norm = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
    )
    iedb_norm = iedb_embeddings / (
        np.linalg.norm(iedb_embeddings, axis=1, keepdims=True) + 1e-8
    )
    sim_matrix = query_norm @ iedb_norm.T
    return sim_matrix.max(axis=1)


def run_esm2_pipeline(
    input_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
    iedb_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Extract ESM-2 embeddings for peptides and optionally compute IEDB similarity."""
    config = load_config(config_path)
    logger = setup_logger("esm2", config["paths"]["logs_dir"])

    emb_cfg = config["embeddings"]
    device = emb_cfg["device"] if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    df = pd.read_csv(input_tsv, sep="\t")
    peptides = df["peptide"].unique().tolist()

    model, tokenizer = load_esm2_model(emb_cfg["esm2_model"], device)
    embeddings = embed_sequences(
        peptides, model, tokenizer, device, emb_cfg["batch_size"]
    )

    pep_to_emb = dict(zip(peptides, embeddings))
    df["esm2_embedding"] = df["peptide"].map(
        lambda p: ",".join(f"{x:.6f}" for x in pep_to_emb[p])
    )

    if iedb_csv and Path(iedb_csv).exists():
        iedb_df = pd.read_csv(iedb_csv)
        iedb_peptides = iedb_df["peptide"].unique().tolist()
        iedb_emb = embed_sequences(
            iedb_peptides, model, tokenizer, device, emb_cfg["batch_size"]
        )
        query_emb = np.array([pep_to_emb[p] for p in df["peptide"]])
        df["iedb_cosine_similarity_max"] = compute_iedb_cosine_similarity(
            query_emb, iedb_emb
        )

    append_tsv(df, output_tsv)
    logger.info(f"ESM-2 embeddings written to {output_tsv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="ESM-2 embedding extraction")
    parser.add_argument("--input", required=True, help="Input TSV with peptide column")
    parser.add_argument("--output", required=True, help="Output TSV with embeddings")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--iedb", default=None, help="IEDB reference CSV")
    args = parser.parse_args()

    iedb = args.iedb or load_config(args.config)["paths"]["iedb_reference"]
    run_esm2_pipeline(args.input, args.output, args.config, iedb)


if __name__ == "__main__":
    main()
