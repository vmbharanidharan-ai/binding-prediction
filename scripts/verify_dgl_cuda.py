#!/usr/bin/env python3
"""Verify DGL can run graph ops on CUDA (catches CPU-only DGL installs)."""
import sys

import dgl
import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False")
        return 1

    dev = torch.device("cuda:0")
    u = torch.tensor([0, 1], device=dev)
    v = torch.tensor([1, 2], device=dev)
    try:
        g = dgl.graph((u, v))
        g = g.to(dev)
        _ = g.num_edges()
        # Same code path as RFdiffusion SE3 transformer (graph.edges on CUDA).
        _ = g.edges()
    except Exception as exc:
        print(f"ERROR: DGL CUDA graph ops failed: {exc}")
        print("Fix: bash scripts/repair_se3nv.sh")
        return 1

    print(f"DGL CUDA OK (dgl {dgl.__version__}, torch {torch.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
