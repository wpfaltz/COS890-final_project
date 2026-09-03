"""Plot runtime and lower-bound-quality comparisons from a benchmark CSV.

Usage:
    python scripts/plot_results.py results/benchmark_25.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pivot_runtime = df.pivot(index="instance", columns="method", values="runtime_s")
    pivot_runtime.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Runtime by method")
    axes[0].set_ylabel("seconds")

    pivot_gap = df.pivot(index="instance", columns="method", values="gap") * 100
    pivot_gap.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Optimality gap by method")
    axes[1].set_ylabel("gap (%)")

    fig.tight_layout()
    out_path = out_dir / f"{csv_path.stem}_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
