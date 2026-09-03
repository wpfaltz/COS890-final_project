"""Branch-and-Bound and Branch-and-Cut solvers for the compact VRPTW model.

Both methods solve the same compact MILP (formulation.py) with Gurobi's
native search tree; they differ only in which cuts are allowed to shape it:

- Branch-and-Bound: Gurobi's own cutting planes and heuristics are switched
  off (Cuts=0, Heuristics=0), so the tree is driven purely by the LP
  relaxation bound at each node plus branching on fractional variables.
- Branch-and-Cut: Gurobi's own cuts stay on, and on top of them we
  separate rounded-capacity cuts and infeasible-path cuts (cuts.py) at
  fractional nodes via a callback, strengthening the LP bound.

The Solomon lexicographic objective (fewest vehicles, then shortest
distance) is handled by solving in two phases: minimize the fleet size
first, then re-solve with the fleet size fixed to minimize distance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .cuts import fix_infeasible_arcs, separate_infeasible_path_cuts, separate_rounded_capacity_cuts
from .formulation import build_model
from .instance import VRPTWInstance
from .solution import Solution


@dataclass
class SolveResult:
    method: str
    status: str
    num_vehicles: int
    total_distance: float
    lower_bound: float
    mip_gap: float
    runtime: float
    solution: Solution | None


def _make_cut_callback(inst: VRPTWInstance, x: dict, customers: list[int]):
    def callback(model, where):
        if where != GRB.Callback.MIPNODE:
            return
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
            return
        x_val = {key: model.cbGetNodeRel(var) for key, var in x.items()}

        for component, rhs in separate_rounded_capacity_cuts(inst, x_val, customers):
            crossing = gp.quicksum(
                var for (i, j), var in x.items() if (i in component) != (j in component)
            )
            model.cbCut(crossing >= rhs)

        for (i, j, k) in separate_infeasible_path_cuts(inst, x_val, customers):
            model.cbCut(x[i, j] + x[j, k] <= 1)

    return callback


def _solve_phase(
    inst: VRPTWInstance,
    fixed_vehicles: int | None,
    objective: str,
    use_cuts: bool,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
) -> tuple[gp.Model, dict]:
    model, v = build_model(inst, fixed_vehicles=fixed_vehicles, objective=objective)
    fix_infeasible_arcs(inst, v["x"], model)

    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    if use_cuts:
        model.Params.Cuts = -1  # Gurobi default: automatic cutting planes.
        callback = _make_cut_callback(inst, v["x"], v["customers"])
        model.optimize(callback)
    else:
        model.Params.Cuts = 0
        model.Params.Heuristics = 0
        model.optimize()

    return model, v


def _extract_solution(model: gp.Model, v: dict, inst: VRPTWInstance) -> Solution | None:
    if model.SolCount == 0:
        return None
    arcs = {(i, j) for (i, j), var in v["x"].items() if var.X > 0.5}
    return Solution.from_arcs(arcs, inst)


def _solve(
    inst: VRPTWInstance,
    method: str,
    use_cuts: bool,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
) -> SolveResult:
    start = time.time()

    # Phase 1: minimize fleet size.
    model1, v1 = _solve_phase(inst, None, "vehicles", use_cuts, time_limit, mip_gap, verbose)
    if model1.SolCount == 0:
        elapsed = time.time() - start
        return SolveResult(method, str(model1.Status), 0, float("inf"), model1.ObjBound, float("inf"), elapsed, None)
    best_k = int(round(model1.ObjVal))
    remaining_time = max(time_limit - (time.time() - start), 1.0)

    # Phase 2: fix fleet size, minimize distance.
    model2, v2 = _solve_phase(inst, best_k, "distance", use_cuts, remaining_time, mip_gap, verbose)
    elapsed = time.time() - start

    solution = _extract_solution(model2, v2, inst)
    status = "optimal" if model2.Status == GRB.OPTIMAL else "time_limit" if model2.Status == GRB.TIME_LIMIT else str(model2.Status)
    total_distance = model2.ObjVal if model2.SolCount > 0 else float("inf")
    mip_gap_final = model2.MIPGap if model2.SolCount > 0 else float("inf")

    return SolveResult(
        method=method,
        status=status,
        num_vehicles=best_k,
        total_distance=total_distance,
        lower_bound=model2.ObjBound,
        mip_gap=mip_gap_final,
        runtime=elapsed,
        solution=solution,
    )


def solve_branch_and_bound(
    inst: VRPTWInstance, time_limit: float = 300.0, mip_gap: float = 0.0, verbose: bool = False
) -> SolveResult:
    return _solve(inst, "branch_and_bound", use_cuts=False, time_limit=time_limit, mip_gap=mip_gap, verbose=verbose)


def solve_branch_and_cut(
    inst: VRPTWInstance, time_limit: float = 300.0, mip_gap: float = 0.0, verbose: bool = False
) -> SolveResult:
    return _solve(inst, "branch_and_cut", use_cuts=True, time_limit=time_limit, mip_gap=mip_gap, verbose=verbose)
