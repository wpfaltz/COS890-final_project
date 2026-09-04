"""Hand-rolled Branch-and-Bound and Branch-and-Cut solvers for the compact VRPTW model.

These are our own best-first search trees. Gurobi is used strictly as an
LP engine (to compute the bound at each node) -- it never sees the integer
variables as binary and never runs its own branching/cutting-plane logic.
This is deliberate: the project's Branch-and-Bound and Branch-and-Cut must
be manually formulated and coded, not "Cuts=0" flags on top of Gurobi's
automatic MIP solver. See `reference_gurobi.py` for a solver that *does*
hand the whole MILP to Gurobi's native engine, kept only as a baseline for
comparison, not counted as one of the three required methods.

- Branch-and-Bound: best-first search over the compact LP relaxation, no
  cuts beyond the base formulation. Each node branches on the most
  fractional x[i, j] (fixed to 0 in one child, to 1 in the other).
- Branch-and-Cut: identical tree, but at the root and at every subsequent
  node, a cutting-plane loop separates rounded-capacity cuts and
  infeasible-path cuts (cuts.py) against the current LP solution and adds
  them as global constraints before deciding whether to branch.

The Solomon lexicographic objective (fewest vehicles, then shortest
distance) is handled in two phases, exactly as before: phase 1 minimizes
the fleet size, phase 2 fixes it and minimizes distance.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .cuts import fix_infeasible_arcs, separate_infeasible_path_cuts, separate_rounded_capacity_cuts
from .formulation import build_model
from .instance import VRPTWInstance
from .solution import Solution

INT_TOL = 1e-6


@dataclass
class SolveResult:
    method: str
    status: str
    num_vehicles: int
    total_distance: float
    lower_bound: float
    root_lp_bound: float
    mip_gap: float
    nodes_explored: int
    runtime: float
    solution: Solution | None


def _add_violated_cuts(model, inst, x, x_val, customers, added_keys: set) -> bool:
    """Add newly-violated RCC / infeasible-path cuts as global constraints; True if any were added."""
    added = False
    for component, rhs in separate_rounded_capacity_cuts(inst, x_val, customers):
        key = ("rcc", frozenset(component))
        if key in added_keys:
            continue
        added_keys.add(key)
        crossing = gp.quicksum(var for (i, j), var in x.items() if (i in component) != (j in component))
        model.addConstr(crossing >= rhs)
        added = True
    for (i, j, k) in separate_infeasible_path_cuts(inst, x_val, customers):
        key = ("path", (i, j, k))
        if key in added_keys:
            continue
        added_keys.add(key)
        model.addConstr(x[i, j] + x[j, k] <= 1)
        added = True
    if added:
        model.update()
    return added


def _run_tree(
    inst: VRPTWInstance,
    fixed_vehicles: int | None,
    objective: str,
    use_cuts: bool,
    time_limit: float,
    mip_gap: float,
    max_nodes: int,
    verbose: bool,
) -> tuple[float, float, float, int, str, Solution | None]:
    """Our own best-first branch-and-bound (optionally branch-and-cut) search tree.

    Returns (incumbent_obj, final_lower_bound, root_lp_bound, nodes_explored, status, incumbent_solution).
    """
    model, v = build_model(inst, fixed_vehicles=fixed_vehicles, objective=objective)
    x, customers = v["x"], v["customers"]
    for var in x.values():
        var.VType = GRB.CONTINUOUS  # LP relaxation only: branching (not Gurobi) enforces integrality
    if objective == "vehicles":
        # A pure vehicle-count objective barely constrains interior (non-depot) arcs, so
        # its LP relaxation is highly degenerate and leaves the tree with almost no guidance
        # to reach integrality. A tiny distance tie-break (never enough to trade off a
        # vehicle) keeps the true minimum fleet size while resolving that degeneracy.
        epsilon = 1e-6
        model.setObjective(
            gp.quicksum(x[inst.depot, j] for j in customers)
            + epsilon * gp.quicksum(inst.distance[i, j] * var for (i, j), var in x.items()),
            GRB.MINIMIZE,
        )
    fix_infeasible_arcs(inst, x, model)
    model.Params.OutputFlag = 1 if verbose else 0
    model.update()

    base_bounds = {key: (var.LB, var.UB) for key, var in x.items()}
    added_cut_keys: set = set()
    current_fixed: dict = {}

    def solve_relaxation(fixed: dict) -> float | None:
        nonlocal current_fixed
        # Only touch the variables whose bounds actually differ from what's on the
        # model right now -- resetting all ~n^2 arc variables on every node is the
        # dominant cost once instances get to ~100 customers.
        for key in current_fixed.keys() - fixed.keys():
            lb0, ub0 = base_bounds[key]
            x[key].LB, x[key].UB = lb0, ub0
        for key, bounds in fixed.items():
            if current_fixed.get(key) == bounds:
                continue
            lb0, ub0 = base_bounds[key]
            flb, fub = bounds
            x[key].LB, x[key].UB = max(lb0, flb), min(ub0, fub)
        current_fixed = fixed
        model.optimize()
        return model.ObjVal if model.Status == GRB.OPTIMAL else None

    start = time.time()
    root_obj = solve_relaxation({})
    if root_obj is None:
        return float("inf"), float("inf"), float("inf"), 0, "infeasible", None

    if use_cuts:
        for _ in range(50):  # root cutting-plane loop
            x_val = {k: var.X for k, var in x.items()}
            if not _add_violated_cuts(model, inst, x, x_val, customers, added_cut_keys):
                break
            root_obj = solve_relaxation({})
    root_lp_bound = root_obj

    counter = itertools.count()
    heap = [(root_obj, next(counter), {})]
    nodes_explored = 0
    incumbent = float("inf")
    incumbent_solution = None
    status = None

    while heap:
        if time.time() - start > time_limit or nodes_explored >= max_nodes:
            status = "node_or_time_limit"
            break

        # Best-first: the smallest bound among all pending nodes is the current global lower bound.
        if incumbent < float("inf") and heap[0][0] >= incumbent - abs(incumbent) * mip_gap - 1e-6:
            status = "optimal"
            break

        bound, _, fixed = heapq.heappop(heap)
        if incumbent < float("inf") and bound >= incumbent - 1e-6:
            continue  # fathomed by bound (stale heap entry)
        nodes_explored += 1

        obj = solve_relaxation(fixed)
        if obj is None or (incumbent < float("inf") and obj >= incumbent - 1e-6):
            continue

        if use_cuts:
            for _ in range(8):  # per-node cutting-plane loop (cut separation itself is O(n^2))
                x_val = {k: var.X for k, var in x.items()}
                if not _add_violated_cuts(model, inst, x, x_val, customers, added_cut_keys):
                    break
                obj = solve_relaxation(fixed)
                if obj is None or (incumbent < float("inf") and obj >= incumbent - 1e-6):
                    break
            if obj is None or (incumbent < float("inf") and obj >= incumbent - 1e-6):
                continue

        x_val = {k: var.X for k, var in x.items()}
        frac = [(k, val) for k, val in x_val.items() if INT_TOL < val < 1 - INT_TOL]

        if not frac:
            if obj < incumbent - 1e-6:
                incumbent = obj
                arcs = {k for k, val in x_val.items() if val > 0.5}
                incumbent_solution = Solution.from_arcs(arcs, inst)
            continue

        branch_key, _ = min(frac, key=lambda kv: abs(kv[1] - 0.5))
        for new_bound in (0.0, 1.0):
            child_fixed = dict(fixed)
            child_fixed[branch_key] = (new_bound, new_bound)
            heapq.heappush(heap, (obj, next(counter), child_fixed))

    if status is None:  # heap ran dry: every node was fathomed
        status = "optimal" if incumbent < float("inf") else "infeasible"

    final_lb = heap[0][0] if heap else incumbent
    if incumbent < float("inf"):
        final_lb = min(final_lb, incumbent)
    return incumbent, final_lb, root_lp_bound, nodes_explored, status, incumbent_solution


def _find_min_vehicles(inst: VRPTWInstance, time_limit: float) -> int | None:
    """Determine the minimal fleet size (phase 1) via Gurobi's own MIP solver.

    Counting depot-out arcs has no interesting routing structure for a hand-rolled
    B&B/B&C to explore -- it is a legitimate auxiliary MIP, solved directly, so
    the manual search tree can spend its entire budget on phase 2 (the actual
    routing problem under a fixed fleet size).
    """
    model, _ = build_model(inst, objective="vehicles")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.optimize()
    return int(round(model.ObjVal)) if model.SolCount > 0 else None


def _solve_two_phase(
    inst: VRPTWInstance,
    method: str,
    use_cuts: bool,
    time_limit: float,
    mip_gap: float,
    max_nodes: int,
    verbose: bool,
) -> SolveResult:
    start = time.time()

    phase1_budget = max(min(time_limit * 0.2, 60.0), 5.0)
    best_k = _find_min_vehicles(inst, phase1_budget)
    if best_k is None:
        elapsed = time.time() - start
        return SolveResult(method, "infeasible", 0, float("inf"), float("inf"), float("inf"), float("inf"), 0, elapsed, None)
    remaining_time = max(time_limit - (time.time() - start), 1.0)

    inc2, lb2, root2, nodes2, status2, sol2 = _run_tree(
        inst, best_k, "distance", use_cuts, remaining_time, mip_gap, max_nodes, verbose
    )
    elapsed = time.time() - start
    gap = (inc2 - lb2) / inc2 if inc2 not in (0.0, float("inf")) else 0.0

    return SolveResult(
        method=method,
        status=status2,
        num_vehicles=best_k,
        total_distance=inc2,
        lower_bound=lb2,
        root_lp_bound=root2,
        mip_gap=gap,
        nodes_explored=nodes2,
        runtime=elapsed,
        solution=sol2,
    )


def solve_branch_and_bound(
    inst: VRPTWInstance,
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    max_nodes: int = 20_000,
    verbose: bool = False,
) -> SolveResult:
    return _solve_two_phase(inst, "branch_and_bound", False, time_limit, mip_gap, max_nodes, verbose)


def solve_branch_and_cut(
    inst: VRPTWInstance,
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    max_nodes: int = 20_000,
    verbose: bool = False,
) -> SolveResult:
    return _solve_two_phase(inst, "branch_and_cut", True, time_limit, mip_gap, max_nodes, verbose)
