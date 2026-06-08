"""ProtT5 embedding extraction via HuggingFace transformers."""

import argparse
from typing import List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.logging import setup_logger
from utils.slurm_utils import append_tsv, load_config


def load_prott5_model(model_name: str, device: str = "cuda"):
    """Load ProtT5 encoder model and tokenizer."""
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_sequences_prott5(
    sequences: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Generate mean-pooled ProtT5 embeddings for peptide sequences."""
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="ProtT5 embedding"):
        batch = sequences[i : i + batch_size]
        spaced = [" ".join(list(seq)) for seq in batch]
        inputs = tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

        for j in range(len(batch)):
            emb = hidden[j].mean(dim=0).cpu().numpy()
            all_embeddings.append(emb)

    return np.array(all_embeddings)


def run_prott5_pipeline(
    input_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Extract ProtT5 embeddings for peptides in input TSV."""
    config = load_config(config_path)
    logger = setup_logger("prott5", config["paths"]["logs_dir"])

    emb_cfg = config["embeddings"]
    device = emb_cfg["device"] if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    df = pd.read_csv(input_tsv, sep="\t")
    peptides = df["peptide"].unique().tolist()

    model, tokenizer = load_prott5_model(emb_cfg["prott5_model"], device)
    embeddings = embed_sequences_prott5(
        peptides, model, tokenizer, device, emb_cfg["batch_size"]
    )

    pep_to_emb = dict(zip(peptides, embeddings))
    df["prott5_embedding"] = df["peptide"].map(
        lambda p: ",".join(f"{x:.6f}" for x in pep_to_emb[p])
    )

    append_tsv(df, output_tsv)
    logger.info(f"ProtT5 embeddings written to {output_tsv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="ProtT5 embedding extraction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_prott5_pipeline(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
