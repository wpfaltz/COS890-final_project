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
from vrptw.reference_gurobi import solve_gurobi_reference
from vrptw.solvers import solve_branch_and_bound, solve_branch_and_cut

DEFAULT_INSTANCES = [
    "c101", "c102", "c103", "c104", "c105", "c106", "c107", "c108", "c109",
    # "c201", "c202", "c203", "c204", "c205", "c206", "c207", "c208",
    # "r101", "r102", "r103", "r104", "r105", "r106", "r107", "r108", "r109", "r110", "r111", "r112", 
    # "r201", "r202", "r203", "r204", "r205", "r206", "r207", "r208", "r209", "r210", "r211", 
    # "rc101", "rc102", "rc103", "rc104", "rc105", "rc106", "rc107", "rc108",
    # "rc201", "rc202", "rc203", "rc204", "rc205", "rc206", "rc207", "rc208",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "solomon" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_one(name: str, num_customers: int, time_limit: float, include_reference: bool = True) -> list[dict]:
    inst = load_solomon_instance(RAW_DIR / f"{name}.txt", num_customers=num_customers)
    rows = []

    # methods = [("branch_and_bound", solve_branch_and_bound), ("branch_and_cut", solve_branch_and_cut)]
    methods = []
    if include_reference:
        methods.append(("gurobi_reference", solve_gurobi_reference))

    for method, solve_fn in methods:
        t0 = time.time()
        result = solve_fn(inst, time_limit=time_limit)
        rows.append({
            "instance": name, "customers": num_customers, "method": method,
            "status": result.status, "vehicles": result.num_vehicles,
            "distance": result.total_distance, "root_lp_bound": result.root_lp_bound,
            "lower_bound": result.lower_bound, "gap": result.mip_gap,
            "nodes_explored": result.nodes_explored, "runtime_s": time.time() - t0,
        })

    # t0 = time.time()
    # lr = lagrangian_relaxation(inst, time_limit=time_limit)
    # rows.append({
    #     "instance": name, "customers": num_customers, "method": "lagrangian_relaxation",
    #     "status": "exact_subproblem" if lr.subproblem_exact_throughout else "heuristic_subproblem",
    #     "vehicles": lr.num_vehicles, "upper_bound": lr.upper_bound, "root_lp_bound": lr.root_lp_bound,
    #     "lower_bound": lr.best_lower_bound,
    #     "gap": (lr.upper_bound - lr.best_lower_bound) / lr.upper_bound if lr.upper_bound else float("nan"),
    #     "nodes_explored": lr.iterations_run, "runtime_s": time.time() - t0,
    # })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--customers", type=int, default=25)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--no-reference", action="store_true",
                         help="skip Gurobi's own automatic solver (it runs by default as a comparison baseline)")
    parser.add_argument("--out-dir", default=None, help="directory for per-instance CSVs (default: results/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in args.instances:
        print(f"solving {name} ({args.customers} customers)...")
        rows = run_one(name, args.customers, args.time_limit, include_reference=not args.no_reference)
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
