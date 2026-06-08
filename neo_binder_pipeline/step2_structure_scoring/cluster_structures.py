"""RMSD-based clustering of peptide–HLA structural ensembles."""

import argparse
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.logging import setup_logger
from utils.structure_utils import compute_rmsd, parse_pdb_coordinates
from utils.slurm_utils import load_config


def cluster_by_rmsd(
    metrics_tsv: str,
    output_tsv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Cluster structures per peptide–HLA pair by CA-RMSD.

    Keeps representative structure (medoid) per cluster.
    """
    config = load_config(config_path)
    logger = setup_logger("step2_cluster", config["paths"]["logs_dir"])
    threshold = config["step2"]["rmsd_cluster_threshold"]

    df = pd.read_csv(metrics_tsv, sep="\t")
    rows = []

    for job_id, group in df.groupby("job_id"):
        structures = group.to_dict("records")
        coords_list = []
        valid_indices = []

        for i, s in enumerate(structures):
            coords = parse_pdb_coordinates(s["pdb_path"])
            if coords:
                coords_list.append(coords)
                valid_indices.append(i)

        if not valid_indices:
            continue

        n = len(valid_indices)
        assigned = [-1] * n
        cluster_id = 0

        for i in range(n):
            if assigned[i] >= 0:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, n):
                if assigned[j] >= 0:
                    continue
                rmsd = compute_rmsd(coords_list[i], coords_list[j])
                if rmsd <= threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        cluster_members: dict = defaultdict(list)
        for idx, cid in enumerate(assigned):
            cluster_members[cid].append(valid_indices[idx])

        for cid, members in cluster_members.items():
            if len(members) == 1:
                medoid_idx = members[0]
            else:
                best_sum = float("inf")
                medoid_idx = members[0]
                for mi in members:
                    total = sum(
                        compute_rmsd(coords_list[mi], coords_list[mj])
                        for mj in members
                        if mj != mi
                    )
                    if total < best_sum:
                        best_sum = total
                        medoid_idx = mi

            s = structures[valid_indices[medoid_idx]]
            rows.append(
                {
                    **{k: s[k] for k in s if k != "pdb_path"},
                    "pdb_path": s["pdb_path"],
                    "cluster_id": cid,
                    "cluster_size": len(members),
                    "is_representative": True,
                }
            )

            for mi in members:
                if mi == medoid_idx:
                    continue
                s_nonrep = structures[valid_indices[mi]]
                rows.append(
                    {
                        **{k: s_nonrep[k] for k in s_nonrep if k != "pdb_path"},
                        "pdb_path": s_nonrep["pdb_path"],
                        "cluster_id": cid,
                        "cluster_size": len(members),
                        "is_representative": False,
                    }
                )

        logger.info(f"{job_id}: {cluster_id} clusters from {n} structures")

    result = pd.DataFrame(rows)
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Clustering results → {output_tsv}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Cluster structures by RMSD")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cluster_by_rmsd(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
