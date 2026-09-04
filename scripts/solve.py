"""Solve a single Solomon VRPTW instance with one of the three methods.

Usage:
    python scripts/solve.py data/solomon/raw/c101.txt --method bc --customers 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrptw.instance import load_solomon_instance
from vrptw.lagrangian import lagrangian_relaxation
from vrptw.reference_gurobi import solve_gurobi_reference
from vrptw.solvers import solve_branch_and_bound, solve_branch_and_cut


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="path to a Solomon .txt instance file")
    parser.add_argument("--method", choices=["bb", "bc", "lr", "ref"], default="bc",
                         help="bb = branch-and-bound, bc = branch-and-cut, lr = Lagrangian relaxation, "
                              "ref = Gurobi's own automatic solver (baseline only, not one of the 3 methods)")
    parser.add_argument("--customers", type=int, default=None, help="truncate to the first N customers")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    inst = load_solomon_instance(args.instance, num_customers=args.customers)
    print(f"Instance {inst.name}: {inst.n_customers} customers, capacity={inst.vehicle_capacity}, "
          f"max_vehicles={inst.max_vehicles}")

    if args.method in ("bb", "bc", "ref"):
        solve_fn = {"bb": solve_branch_and_bound, "bc": solve_branch_and_cut, "ref": solve_gurobi_reference}[args.method]
        result = solve_fn(inst, time_limit=args.time_limit, verbose=args.verbose)
        print(f"status={result.status} vehicles={result.num_vehicles} distance={result.total_distance:.2f} "
              f"lower_bound={result.lower_bound:.2f} root_lp_bound={result.root_lp_bound:.2f} "
              f"nodes_explored={result.nodes_explored} gap={result.mip_gap:.4%} runtime={result.runtime:.2f}s")
        if result.solution is not None:
            ok, reason = result.solution.validate(inst)
            print("solution valid" if ok else f"solution INVALID: {reason}")
            for idx, route in enumerate(result.solution.routes):
                print(f"  route {idx}: {route.nodes} (load={route.load(inst):.1f}, dist={route.distance(inst):.2f})")
    else:
        lr = lagrangian_relaxation(inst, time_limit=args.time_limit, verbose=args.verbose)
        gap = (lr.upper_bound - lr.best_lower_bound) / lr.upper_bound if lr.upper_bound else float("nan")
        print(f"lower_bound={lr.best_lower_bound:.2f} root_lp_bound={lr.root_lp_bound:.2f} "
              f"upper_bound={lr.upper_bound:.2f} gap={gap:.4%} "
              f"vehicles={lr.num_vehicles} iterations={lr.iterations_run} runtime={lr.runtime:.2f}s "
              f"subproblem_exact={lr.subproblem_exact_throughout} "
              f"best_lb_iteration={lr.best_lb_iteration} best_ub_iteration={lr.best_ub_iteration}")
        if lr.best_solution is not None:
            ok, reason = lr.best_solution.validate(inst)
            print("solution valid" if ok else f"solution INVALID: {reason}")
            for idx, route in enumerate(lr.best_solution.routes):
                print(f"  route {idx}: {route.nodes} (load={route.load(inst):.1f}, dist={route.distance(inst):.2f})")


if __name__ == "__main__":
    main()
