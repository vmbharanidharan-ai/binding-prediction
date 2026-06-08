#!/usr/bin/env python3
"""One-command HLA database setup: download IMGT FASTA + build index."""

import subprocess
import sys
from pathlib import Path

HLA_ROOT = Path(__file__).resolve().parent


def setup() -> None:
    print("=== HLA setup: download IMGT + build index ===")
    subprocess.run([sys.executable, str(HLA_ROOT / "download_imgt.py")], check=True)
    subprocess.run([sys.executable, str(HLA_ROOT / "build_hla_index.py")], check=True)
    print("=== HLA setup complete ===")
    print("Test: python hla/hla_resolver.py HLA_A0201")


if __name__ == "__main__":
    setup()
