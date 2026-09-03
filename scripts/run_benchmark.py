"""Run all three methods on a batch of Solomon instances and save a comparison table.

Usage:
    python scripts/run_benchmark.py --customers 25 --time-limit 120 --instances c101 r101 rc101
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from vrptw.instance import load_solomon_instance
from vrptw.lagrangian import lagrangian_relaxation
from vrptw.solvers import solve_branch_and_bound, solve_branch_and_cut

DEFAULT_INSTANCES = ["c101", "c201", "r101", "r201", "rc101", "rc201"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "solomon" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_one(name: str, num_customers: int, time_limit: float) -> list[dict]:
    inst = load_solomon_instance(RAW_DIR / f"{name}.txt", num_customers=num_customers)
    rows = []

    for method, solve_fn in (("branch_and_bound", solve_branch_and_bound), ("branch_and_cut", solve_branch_and_cut)):
        t0 = time.time()
        result = solve_fn(inst, time_limit=time_limit)
        rows.append({
            "instance": name, "customers": num_customers, "method": method,
            "status": result.status, "vehicles": result.num_vehicles,
            "distance": result.total_distance, "lower_bound": result.lower_bound,
            "gap": result.mip_gap, "runtime_s": time.time() - t0,
        })

    t0 = time.time()
    lr = lagrangian_relaxation(inst, time_limit=time_limit)
    rows.append({
        "instance": name, "customers": num_customers, "method": "lagrangian_relaxation",
        "status": "exact_subproblem" if lr.subproblem_exact_throughout else "heuristic_subproblem",
        "vehicles": lr.num_vehicles, "distance": lr.upper_bound, "lower_bound": lr.best_lower_bound,
        "gap": (lr.upper_bound - lr.best_lower_bound) / lr.upper_bound if lr.upper_bound else float("nan"),
        "runtime_s": time.time() - t0,
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--customers", type=int, default=25)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--out-dir", default=None, help="directory for per-instance CSVs (default: results/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in args.instances:
        print(f"solving {name} ({args.customers} customers)...")
        rows = run_one(name, args.customers, args.time_limit)
        all_rows.extend(rows)

        df = pd.DataFrame(rows)
        out_path = out_dir / f"benchmark_{name}_{args.customers}.csv"
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        print(f"saved to {out_path}\n")

    combined = pd.DataFrame(all_rows)
    combined_path = out_dir / f"benchmark_{args.customers}_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"saved combined results to {combined_path}")


if __name__ == "__main__":
    main()
