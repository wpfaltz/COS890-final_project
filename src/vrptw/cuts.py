"""Cut separation routines used by the branch-and-cut solver.

These add valid inequalities on top of the compact formulation in
formulation.py purely to strengthen the LP relaxation bound (the compact
model is already correct/feasible without them, thanks to the time-window
linking constraints).
"""
from __future__ import annotations

import math

import networkx as nx

from .instance import VRPTWInstance


def fix_infeasible_arcs(inst: VRPTWInstance, x: dict, model) -> int:
    """Preprocessing: fix x[i, j] = 0 for arcs that can never be used feasibly.

    An arc (i, j) is infeasible if leaving i at the earliest possible time
    and travelling straight to j would still arrive after j's due date.
    Returns the number of arcs fixed.
    """
    depot = inst.depot
    fixed = 0
    for (i, j), var in list(x.items()):
        if j == depot:
            continue
        earliest_i = 0.0 if i == depot else inst.ready_time[i]
        service_i = 0.0 if i == depot else inst.service_time[i]
        earliest_arrival_j = earliest_i + service_i + inst.travel_time(i, j)
        if earliest_arrival_j > inst.due_date[j] + 1e-9:
            var.ub = 0
            fixed += 1
    return fixed


def separate_rounded_capacity_cuts(inst: VRPTWInstance, x_val: dict, customers: list[int]) -> list[tuple[set, float]]:
    """Heuristic separation of rounded capacity cuts (RCC).

    Builds the undirected support graph restricted to arcs with positive
    fractional flow and looks at its connected components among customers.
    Any component S whose total demand exceeds one vehicle's capacity needs
    at least ceil(d(S)/Q) vehicles crossing its boundary, giving the cut:
        sum_{i in S, j not in S} x[i, j] >= ceil(d(S) / Q)

    Returns a list of (S, rhs) violated (or component-based) candidate cuts.
    """
    Q = inst.vehicle_capacity
    graph = nx.Graph()
    graph.add_nodes_from(customers)
    for (i, j), val in x_val.items():
        if i in customers and j in customers and val > 1e-6:
            w = graph.get_edge_data(i, j, default={"weight": 0.0})["weight"] + val
            graph.add_edge(i, j, weight=w)

    cuts = []
    for component in nx.connected_components(graph):
        demand = sum(inst.demand[c] for c in component)
        if demand <= Q + 1e-9:
            continue
        rhs = math.ceil(demand / Q - 1e-9)
        crossing = sum(
            val for (i, j), val in x_val.items()
            if (i in component) != (j in component)
        )
        if crossing < rhs - 1e-6:
            cuts.append((set(component), rhs))
    return cuts


def separate_infeasible_path_cuts(inst: VRPTWInstance, x_val: dict, customers: list[int]) -> list[tuple[int, int, int]]:
    """Separate violated infeasible 2-arc path cuts.

    For a triple (i, j, k) such that visiting i then j then k is infeasible
    w.r.t. time windows alone (independent of what happens before i), the
    inequality x[i, j] + x[j, k] <= 1 is valid. Returns triples whose
    current fractional value violates this inequality.
    """
    violated = []
    for j in customers:
        preds = [i for i in customers if i != j and x_val.get((i, j), 0.0) > 1e-6]
        succs = [k for k in customers if k != j and x_val.get((j, k), 0.0) > 1e-6]
        for i in preds:
            arrive_j = max(inst.ready_time[i] + inst.service_time[i] + inst.travel_time(i, j), inst.ready_time[j])
            if arrive_j > inst.due_date[j] + 1e-9:
                continue  # already excluded by the compact model's own constraints
            depart_j = arrive_j + inst.service_time[j]
            for k in succs:
                if k == i:
                    continue
                arrive_k = depart_j + inst.travel_time(j, k)
                if arrive_k > inst.due_date[k] + 1e-9:
                    if x_val[(i, j)] + x_val[(j, k)] > 1 + 1e-6:
                        violated.append((i, j, k))
    return violated
