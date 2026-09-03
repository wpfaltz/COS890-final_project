"""Compact arc-flow (2-index) MILP formulation for the VRPTW, built with gurobipy.

Variables
---------
x[i, j]  in {0,1}  for every ordered pair of distinct nodes i, j (0 = depot)
w[i]     >= 0      service start time at customer i
u[i]     >= 0      cumulative load delivered up to and including customer i

Constraints
-----------
- degree constraints: every customer is left and entered exactly once
- depot degree: at most `max_vehicles` routes leave the depot, and the number
  of routes leaving equals the number returning
- time-window linking (also eliminates subtours, since travel times are > 0):
      w[j] >= w[i] + service[i] + t[i, j] - M_ij * (1 - x[i, j])
- capacity linking (MTZ-style):
      u[j] >= u[i] + demand[j] - Q * (1 - x[i, j])

No lazy subtour-elimination constraints are required for correctness: the
big-M time-linking constraints already forbid any cycle that does not pass
through the depot, since travel times are strictly positive. Rounded
capacity cuts / infeasible-path cuts (see cuts.py) are added on top purely
to strengthen the LP relaxation (branch-and-cut), not for feasibility.
"""
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .instance import VRPTWInstance


def build_model(
    inst: VRPTWInstance,
    fixed_vehicles: int | None = None,
    objective: str = "distance",
) -> tuple[gp.Model, dict]:
    """Build the compact VRPTW MILP.

    Args:
        inst: the VRPTW instance.
        fixed_vehicles: if given, forces the number of routes leaving the
            depot to be exactly this value (used in phase 2 of the
            lexicographic solve, after phase 1 found the minimal fleet size).
        objective: "distance" minimizes total travelled distance (phase 2 of
            the lexicographic Solomon objective), "vehicles" minimizes the
            number of routes leaving the depot (phase 1).

    Returns:
        (model, vars) where vars = {"x": x, "w": w, "u": u}.
    """
    n = inst.n_nodes
    nodes = range(n)
    customers = list(inst.customers)
    depot = inst.depot
    Q = inst.vehicle_capacity
    max_vehicles = min(inst.max_vehicles, len(customers))

    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    model = gp.Model(f"VRPTW_{inst.name}")
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    w = model.addVars(customers, lb=[inst.ready_time[i] for i in customers],
                       ub=[inst.due_date[i] for i in customers], name="w")
    u = model.addVars(customers, lb=[inst.demand[i] for i in customers], ub=Q, name="u")

    if objective == "distance":
        model.setObjective(gp.quicksum(inst.distance[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE)
    elif objective == "vehicles":
        model.setObjective(gp.quicksum(x[depot, j] for j in customers), GRB.MINIMIZE)
    else:
        raise ValueError(f"unknown objective '{objective}', expected 'distance' or 'vehicles'")

    # Degree constraints: every customer visited exactly once.
    model.addConstrs((gp.quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers), name="out_deg")
    model.addConstrs((gp.quicksum(x[j, i] for j in nodes if j != i) == 1 for i in customers), name="in_deg")

    # Depot degree: bounded (or fixed) fleet size, routes that leave must return.
    depot_out = gp.quicksum(x[depot, j] for j in customers)
    depot_in = gp.quicksum(x[j, depot] for j in customers)
    if fixed_vehicles is not None:
        model.addConstr(depot_out == fixed_vehicles, name="fleet_size")
    else:
        model.addConstr(depot_out <= max_vehicles, name="fleet_size_ub")
    model.addConstr(depot_out == depot_in, name="depot_flow_conservation")

    # Time-window linking (also rules out subtours not containing the depot).
    for i, j in arcs:
        if j == depot:
            continue
        arrival_i = 0.0 if i == depot else inst.ready_time[i]
        service_i = 0.0 if i == depot else inst.service_time[i]
        due_i = inst.due_date[depot] if i == depot else inst.due_date[i]
        big_m = due_i + service_i + inst.travel_time(i, j) - inst.ready_time[j]
        big_m = max(big_m, 0.0)
        w_i = 0.0 if i == depot else w[i]
        model.addConstr(
            w[j] >= w_i + service_i + inst.travel_time(i, j) - big_m * (1 - x[i, j]),
            name=f"tw_link[{i},{j}]",
        )

    # Capacity linking (MTZ-style cumulative load).
    for i, j in arcs:
        if j == depot:
            continue
        demand_i = 0.0 if i == depot else inst.demand[i]
        big_m = Q + inst.demand[j]
        u_i = 0.0 if i == depot else u[i]
        model.addConstr(
            u[j] >= u_i + inst.demand[j] - big_m * (1 - x[i, j]),
            name=f"cap_link[{i},{j}]",
        )

    model.update()
    return model, {"x": x, "w": w, "u": u, "arcs": arcs, "customers": customers, "depot": depot}
