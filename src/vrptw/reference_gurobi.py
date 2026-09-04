"""Baseline solver that hands the whole compact MILP to Gurobi's own,
fully-automatic branch-and-cut engine (its own cuts, heuristics, node
selection -- everything).

This is NOT one of the three required methods. It exists only so the
hand-rolled solvers in solvers.py (our own search tree, cuts and
branching) have something to sanity-check their optimal values against,
and so the presentation can show, purely for comparison, how much better
a mature commercial solver does out of the box.
"""
from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from .cuts import fix_infeasible_arcs
from .formulation import build_model
from .instance import VRPTWInstance
from .solution import Solution
from .solvers import SolveResult


def _root_lp_bound(model: gp.Model) -> float:
    relaxed = model.relax()
    relaxed.Params.OutputFlag = 0
    relaxed.optimize()
    return relaxed.ObjVal if relaxed.Status == GRB.OPTIMAL else float("nan")


def _solve_phase(
    inst: VRPTWInstance, fixed_vehicles: int | None, objective: str, time_limit: float, mip_gap: float, verbose: bool
) -> tuple[gp.Model, dict, float]:
    model, v = build_model(inst, fixed_vehicles=fixed_vehicles, objective=objective)
    fix_infeasible_arcs(inst, v["x"], model)
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    root_bound = _root_lp_bound(model)
    model.optimize()
    return model, v, root_bound


def solve_gurobi_reference(
    inst: VRPTWInstance, time_limit: float = 300.0, mip_gap: float = 0.0, verbose: bool = False
) -> SolveResult:
    start = time.time()

    # Cap phase 1 (minimize fleet size) so it can't eat the whole budget and starve
    # phase 2 -- proving the minimal fleet size is itself a full VRPTW feasibility
    # problem under this compact formulation and can be just as hard as the routing
    # problem, so an uncapped phase 1 can time out having only found (not proven) an
    # upper bound on the fleet size, leaving phase 2 with no time and an unreliable
    # fixed vehicle count. Slightly more generous than solvers._solve_two_phase's
    # 20%/60s/5s split since Gurobi's own engine (vs. our hand-rolled tree) can make
    # good use of the extra time on this sub-MIP.
    phase1_budget = max(min(time_limit * 0.2, 150.0), 5.0)

    model1, v1, root1 = _solve_phase(inst, None, "vehicles", phase1_budget, mip_gap, verbose)
    if model1.SolCount == 0:
        elapsed = time.time() - start
        return SolveResult(
            "gurobi_reference", str(model1.Status), 0, float("inf"), model1.ObjBound, root1,
            float("inf"), int(model1.NodeCount), elapsed, None,
        )
    best_k = int(round(model1.ObjVal))
    vehicles_proven_optimal = model1.Status == GRB.OPTIMAL
    remaining = max(time_limit - (time.time() - start), 1.0)

    model2, v2, root2 = _solve_phase(inst, best_k, "distance", remaining, mip_gap, verbose)
    elapsed = time.time() - start

    solution = None
    if model2.SolCount > 0:
        arcs = {(i, j) for (i, j), var in v2["x"].items() if var.X > 0.5}
        solution = Solution.from_arcs(arcs, inst)
    if model2.Status == GRB.OPTIMAL and vehicles_proven_optimal:
        status = "optimal"
    elif not vehicles_proven_optimal:
        # phase 2 may have solved to optimality, but for an unproven (possibly too
        # large) fleet size, so the overall result isn't a proven global optimum.
        status = "fleet_size_not_proven_optimal"
    else:
        status = "time_limit" if model2.Status == GRB.TIME_LIMIT else str(model2.Status)
    total_distance = model2.ObjVal if model2.SolCount > 0 else float("inf")
    gap = model2.MIPGap if model2.SolCount > 0 else float("inf")

    return SolveResult(
        method="gurobi_reference",
        status=status,
        num_vehicles=best_k,
        total_distance=total_distance,
        lower_bound=model2.ObjBound,
        root_lp_bound=root2,
        mip_gap=gap,
        nodes_explored=int(model1.NodeCount + model2.NodeCount),
        runtime=elapsed,
        solution=solution,
    )
