#!/usr/bin/env python3
"""Download IMGT/HLA protein FASTA files (A, B, C loci)."""

import argparse
import sys
from pathlib import Path

import requests
import yaml

HLA_ROOT = Path(__file__).resolve().parent


def load_hla_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def download_file(url: str, dest: Path) -> None:
    print(f"[INFO] Downloading {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    print(f"[DONE] Saved {dest} ({dest.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Download IMGT HLA protein FASTA")
    parser.add_argument("--config", default=str(HLA_ROOT / "hla_config.yaml"))
    args = parser.parse_args()

    cfg = load_hla_config(Path(args.config))
    base = cfg["imgt_base_url"].rstrip("/")
    data_dir = HLA_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for fname in cfg["fasta_files"]:
        url = f"{base}/{fname}"
        dest = data_dir / fname.replace(".fasta", "_imgt.fasta")
        download_file(url, dest)
        downloaded.append(dest)

    combined = HLA_ROOT / "data" / "imgt_hla_proteins.fasta"
    with open(combined, "w") as out:
        for src in downloaded:
            out.write(src.read_text())
            out.write("\n")
    print(f"[DONE] Combined {len(downloaded)} files → {combined}")


if __name__ == "__main__":
    sys.exit(main())
