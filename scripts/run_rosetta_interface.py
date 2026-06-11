#!/usr/bin/env python3
"""Run Rosetta InterfaceAnalyzer on a peptide–HLA PDB."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from utils.logging import setup_logger
from utils.slurm_utils import load_config

NA_METRICS = {
    "dG_separated": float("nan"),
    "dSASA_int": float("nan"),
    "delta_unsatHbonds": float("nan"),
}

ROSETTA_XML_TEMPLATE = """<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref15" weights="ref2015"/>
  </SCOREFXNS>
  <PROTOCOLS>
    <Add mover_name="InterfaceAnalyzerMover"/>
  </PROTOCOLS>
  <MOVERS>
    <InterfaceAnalyzerMover name="InterfaceAnalyzerMover"
      interface="{interface_string}"
      scorefxn="ref15"
      pack_separated="false"
      tracer_data_print="false"/>
  </MOVERS>
</ROSETTASCRIPTS>
"""


def parse_rosetta_stdout(stdout: str) -> dict[str, float]:
    """Parse InterfaceAnalyzer metrics from Rosetta stdout."""
    metrics = dict(NA_METRICS)
    patterns = {
        "dG_separated": r"dG_separated[:\s]+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        "dSASA_int": r"dSASA_int[:\s]+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        "delta_unsatHbonds": r"delta_unsatHbonds[:\s]+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def write_output_tsv(
    output_dir: Path,
    pair_id: str,
    metrics: dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair_id}_rosetta.tsv"
    row = {"pair_id": pair_id, **metrics}
    pd.DataFrame([row]).to_csv(out_path, sep="\t", index=False)
    return out_path


def run_rosetta_interface(
    pdb_path: str,
    pair_id: str,
    output_dir: str,
    rosetta_bin: str,
    interface_string: str = "A_BC",
    config_path: str = "config/config.yaml",
) -> Path:
    """Run Rosetta InterfaceAnalyzer and write a one-row metrics TSV."""
    config = load_config(config_path)
    logger = setup_logger("rosetta_interface", config["paths"]["logs_dir"])
    out_dir = Path(output_dir)

    pdb = Path(pdb_path)
    if not pdb.exists():
        logger.error(f"PDB not found: {pdb}")
        return write_output_tsv(out_dir, pair_id, NA_METRICS)

    rosetta_path = Path(rosetta_bin).expanduser() if rosetta_bin else Path()
    if not rosetta_bin or not rosetta_path.exists():
        logger.error(
            f"Rosetta binary not found: {rosetta_bin!r}. "
            "Set rosetta.bin_path in config/config.yaml or disable rosetta.enabled."
        )
        return write_output_tsv(out_dir, pair_id, NA_METRICS)

    xml_content = ROSETTA_XML_TEMPLATE.format(interface_string=interface_string)
    with tempfile.TemporaryDirectory(prefix="rosetta_iface_") as tmpdir:
        xml_path = Path(tmpdir) / "interface_analyzer.xml"
        xml_path.write_text(xml_content)

        cmd = [
            str(rosetta_path),
            "-parser:protocol",
            str(xml_path),
            "-s",
            str(pdb.resolve()),
            "-out:file:scorefile",
            str(Path(tmpdir) / "score.sc"),
            "-mute",
            "all",
        ]

        database = config.get("rosetta", {}).get("database_path", "")
        if database:
            cmd.extend(["-database", database])

        logger.info(f"Running Rosetta: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=tmpdir,
            )
            combined = f"{result.stdout}\n{result.stderr}"
            if result.returncode != 0:
                logger.warning(
                    f"Rosetta exited with code {result.returncode}; "
                    "writing NA metrics so downstream steps can continue."
                )
                return write_output_tsv(out_dir, pair_id, NA_METRICS)

            metrics = parse_rosetta_stdout(combined)
            if all(pd.isna(v) for v in metrics.values()):
                logger.warning("Could not parse Rosetta interface metrics from stdout.")
            out_path = write_output_tsv(out_dir, pair_id, metrics)
            logger.info(f"Rosetta metrics written: {out_path}")
            return out_path
        except OSError as exc:
            logger.error(f"Failed to launch Rosetta: {exc}")
            return write_output_tsv(out_dir, pair_id, NA_METRICS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Rosetta InterfaceAnalyzer")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rosetta-bin", default=None)
    parser.add_argument("--interface-string", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rosetta_cfg = config.get("rosetta", {})
    if not rosetta_cfg.get("enabled", True):
        write_output_tsv(Path(args.output_dir), args.pair_id, NA_METRICS)
        return 0

    run_rosetta_interface(
        pdb_path=args.pdb,
        pair_id=args.pair_id,
        output_dir=args.output_dir,
        rosetta_bin=args.rosetta_bin or rosetta_cfg.get("bin_path", ""),
        interface_string=args.interface_string or rosetta_cfg.get("interface_string", "A_BC"),
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
