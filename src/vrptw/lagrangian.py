"""Lagrangian relaxation for the VRPTW via subgradient optimization.

Relaxed constraint
-------------------
We dualize the "customer i is served exactly once" constraint. In the
Dantzig-Wolfe / set-partitioning view of the VRPTW (each decision variable
is a feasible elementary route), this constraint couples the routes
together. Once it is priced into the objective with a multiplier lambda_i
per customer, the relaxed problem decomposes into `num_vehicles`
independent copies of the same subproblem:

    find the minimum reduced-cost elementary route (depot -> depot)
    respecting time windows and vehicle capacity,
    where reduced_cost(i, j) = c_ij - lambda_j  (lambda charged once per
    customer visited, attributed to the arc that arrives at it).

Since the `num_vehicles` slots are otherwise independent, the optimal
value of the relaxed problem is simply `num_vehicles` copies of the single
best route found (or 0 if its reduced cost is non-negative). The pricing
subproblem itself is a Shortest Path Problem with Resource Constraints
(SPPRC), solved here with a label-setting DP: resources (time, load) only
increase along any used arc (travel/service times and demands are
non-negative), so labels can be processed in non-decreasing order of time
without needing non-negative reduced costs (unlike plain Dijkstra, which
the negative reduced costs here would break).

ng-route relaxation
--------------------
Forbidding a route from ever revisiting a customer (full elementarity)
requires a dominance rule keyed by the *global* set of visited customers,
whose state space explodes combinatorially for instances with many
mutually time-compatible customers (typically the clustered `c1xx`/`c2xx`
Solomon families). We use the standard ng-route relaxation (Baldacci,
Mingozzi & Roberti) instead: each customer j only "remembers" a small
neighbourhood ng(j) (its `ng_k` nearest customers); a label may revisit a
customer once every member of its ng-set has since been left behind. This
keeps the per-node dominance state small (2^ng_k at most) regardless of
instance size, and is still a valid relaxation of the elementary shortest
path problem (it only ever allows *more* routes than strict elementarity),
so an exactly-solved ng-route subproblem still yields a certified, if
somewhat weaker, Lagrangian lower bound.

Caveat: even with ng-route pruning, the label set can still grow quickly
for larger/harder instances. There is no cap on the number of labels
explored -- the subproblem always runs to true optimality, bounded only
by the overall wall-clock `time_limit` of the outer subgradient loop; if
that deadline is hit first, the returned bound is only a heuristic
approximation of the true subproblem optimum (documented via
`SubproblemResult.exact`).
"""
from __future__ import annotations

import heapq
import itertools
import time as _time
from dataclasses import dataclass, field

from gurobipy import GRB

from .formulation import build_model
from .instance import VRPTWInstance
from .solution import Route, Solution


@dataclass
class SubproblemResult:
    cost: float
    path: list[int] | None  # None means "don't use this vehicle" (reduced cost >= 0)
    exact: bool
    completed_routes: list[tuple[float, list[int]]] = field(default_factory=list)


def _build_ng_masks(inst: VRPTWInstance, customers: list[int], bit: dict[int, int], k: int) -> dict[int, int]:
    """ng(j) = j plus its k nearest customers by distance, encoded as a bitmask."""
    masks = {}
    for j in customers:
        nearest = sorted((c for c in customers if c != j), key=lambda c: inst.distance[j, c])[:k]
        mask = bit[j]
        for c in nearest:
            mask |= bit[c]
        masks[j] = mask
    return masks


def _solve_espprc(
    inst: VRPTWInstance,
    reduced_cost,
    bit: dict[int, int],
    ng_mask: dict[int, int],
    deadline: float | None = None,
    max_completed_routes: int = 50,
) -> SubproblemResult:
    depot = inst.depot
    customers = list(bit.keys())
    Q = inst.vehicle_capacity

    # label = (time, node, cost, real_cost, load, visited_mask, path)
    counter = itertools.count()
    start = (0.0, next(counter), depot, 0.0, 0.0, 0.0, 0, (depot,))
    heap = [start]
    labels_at_node: dict[int, list[tuple[float, float, float, int]]] = {}

    best_cost, best_path = 0.0, None
    completed_routes: list[tuple[float, list[int]]] = []
    explored = 0
    exact = True

    while heap:
        if deadline is not None and explored % 2000 == 0 and _time.time() > deadline:
            exact = False
            break
        arrival, _, node, cost, real_cost, load, visited, path = heapq.heappop(heap)
        explored += 1

        bucket = labels_at_node.setdefault(node, [])
        dominated = False
        for (t2, c2, l2, v2) in bucket:
            if c2 <= cost and t2 <= arrival and l2 <= load and (v2 & visited) == v2:
                dominated = True
                break
        if dominated:
            continue
        # Keep the bucket as a Pareto frontier: drop entries the new label dominates.
        labels_at_node[node] = [
            (t2, c2, l2, v2) for (t2, c2, l2, v2) in bucket
            if not (cost <= c2 and arrival <= t2 and load <= l2 and (visited & v2) == visited)
        ]
        labels_at_node[node].append((arrival, cost, load, visited))

        if node == depot and len(path) > 1:
            if cost < -1e-9 and len(completed_routes) < max_completed_routes:
                completed_routes.append((real_cost, list(path)))
            if cost < best_cost - 1e-9:
                best_cost, best_path = cost, list(path)
            continue

        cur_service = 0.0 if node == depot else inst.service_time[node]
        for j in customers:
            if visited & bit[j]:
                continue  # j (or a member of its ng-set) was visited too recently
            new_load = load + inst.demand[j]
            if new_load > Q + 1e-9:
                continue
            new_arrival = max(arrival + cur_service + inst.travel_time(node, j), inst.ready_time[j])
            if new_arrival > inst.due_date[j] + 1e-9:
                continue
            new_cost = cost + reduced_cost[node, j]
            new_real_cost = real_cost + inst.distance[node, j]
            new_visited = (visited & ng_mask[j]) | bit[j]
            heapq.heappush(
                heap,
                (new_arrival, next(counter), j, new_cost, new_real_cost, new_load, new_visited, path + (j,)),
            )

        if len(path) > 1:  # allow returning to depot only after visiting at least one customer
            new_arrival = max(arrival + cur_service + inst.travel_time(node, depot), inst.ready_time[depot])
            if new_arrival <= inst.due_date[depot] + 1e-9:
                new_cost = cost + reduced_cost[node, depot]
                new_real_cost = real_cost + inst.distance[node, depot]
                heapq.heappush(
                    heap,
                    (new_arrival, next(counter), depot, new_cost, new_real_cost, load, visited, path + (depot,)),
                )

    return SubproblemResult(cost=best_cost, path=best_path, exact=exact, completed_routes=completed_routes)


def _estimate_min_vehicles(inst: VRPTWInstance, time_limit: float = 20.0) -> int:
    """Use Gurobi to quickly find the minimal fleet size, giving a tight K for the
    Lagrangian subproblem (using inst.max_vehicles instead would make every route
    slot 'free', producing a very loose, uninformative bound).
    """
    model, v = build_model(inst, objective="vehicles")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.optimize()
    return int(round(model.ObjVal)) if model.SolCount > 0 else min(inst.max_vehicles, inst.n_customers)


def _compute_root_lp_bound(inst: VRPTWInstance, num_vehicles: int, time_limit: float = 30.0) -> float:
    """Plain LP relaxation of the compact formulation (fleet size fixed to the value
    used by the Lagrangian subproblem), solved once with no branching or cuts. This is
    NOT part of the Lagrangian method itself -- it is a cheap complementary bound
    reported alongside it, directly comparable to the `root_lp_bound` reported by the
    B&B/B&C trees.
    """
    model, v = build_model(inst, fixed_vehicles=num_vehicles, objective="distance")
    for var in v["x"].values():
        var.VType = GRB.CONTINUOUS
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.optimize()
    return model.ObjVal if model.Status == GRB.OPTIMAL else float("-inf")


def _greedy_solution(inst: VRPTWInstance, customers: list[int] | None = None) -> Solution:
    """Simple nearest-feasible-neighbour construction, used to seed/repair an upper bound."""
    unvisited = set(inst.customers if customers is None else customers)
    routes: list[Route] = []
    while unvisited:
        nodes = [inst.depot]
        load = 0.0
        t = 0.0
        cur = inst.depot
        while True:
            candidates = []
            for c in unvisited:
                new_load = load + inst.demand[c]
                if new_load > inst.vehicle_capacity + 1e-9:
                    continue
                arrival = max(t + inst.travel_time(cur, c), inst.ready_time[c])
                if arrival > inst.due_date[c] + 1e-9:
                    continue
                back = arrival + inst.service_time[c] + inst.travel_time(c, inst.depot)
                if back > inst.due_date[inst.depot] + 1e-9:
                    continue
                candidates.append((inst.distance[cur, c], c, arrival))
            if not candidates:
                break
            _, c, arrival = min(candidates)
            nodes.append(c)
            load += inst.demand[c]
            t = arrival + inst.service_time[c]
            cur = c
            unvisited.remove(c)
        nodes.append(inst.depot)
        if len(nodes) > 2:
            routes.append(Route(nodes=nodes))
        else:
            # Should not normally happen for standard Solomon instances; guard against infinite loop.
            raise RuntimeError("greedy construction got stuck: an unvisited customer has no feasible route")
    return Solution(routes=routes)


def _assemble_solution_from_pool(inst: VRPTWInstance, pool: dict[frozenset, tuple[float, list[int]]]) -> Solution:
    """Lagrangian heuristic: greedily assemble a full solution out of the elementary
    routes ("columns") found as attractive by the pricing subproblem across
    iterations, covering any customers left over with the nearest-neighbour
    fallback. This is what actually lets the reported upper bound improve as the
    subgradient method progresses, instead of being frozen at the first guess.
    """
    candidates = sorted(pool.values(), key=lambda pc: pc[0] / max(1, len(pc[1]) - 2))
    covered: set[int] = set()
    routes: list[Route] = []
    for cost, path in candidates:
        custs = path[1:-1]
        if any(c in covered for c in custs):
            continue
        routes.append(Route(nodes=list(path)))
        covered.update(custs)

    remaining = [c for c in inst.customers if c not in covered]
    if remaining:
        routes.extend(_greedy_solution(inst, customers=remaining).routes)
    return Solution(routes=routes)


@dataclass
class LagrangianResult:
    best_lower_bound: float
    root_lp_bound: float
    num_vehicles: int
    multipliers: dict
    history: list[float] = field(default_factory=list)
    upper_bound: float | None = None
    best_solution: Solution | None = None
    runtime: float = 0.0
    iterations_run: int = 0
    subproblem_exact_throughout: bool = True
    best_lb_iteration: int = 0
    best_ub_iteration: int = 0


def lagrangian_relaxation(
    inst: VRPTWInstance,
    num_vehicles: int | None = None,
    max_iterations: int | None = None,
    initial_alpha: float = 2.0,
    upper_bound: float | None = None,
    ng_k: int = 8,
    time_limit: float = 120.0,
    verbose: bool = False,
) -> LagrangianResult:
    start_time = _time.time()
    customers = list(inst.customers)
    K = num_vehicles if num_vehicles is not None else _estimate_min_vehicles(inst)
    root_lp_bound = _compute_root_lp_bound(inst, K)

    bit = {c: 1 << idx for idx, c in enumerate(customers)}
    ng_mask = _build_ng_masks(inst, customers, bit, k=ng_k)

    best_solution = _greedy_solution(inst)
    if upper_bound is None:
        upper_bound = best_solution.total_distance(inst)
    else:
        best_solution = None  # caller supplied their own bound; we have no matching solution for it

    column_pool: dict[frozenset, tuple[float, list[int]]] = {}

    lam = {c: 0.0 for c in customers}
    alpha = initial_alpha
    best_lb = 0.0  # trivially valid: all distances are non-negative, so the true optimum is always >= 0
    history: list[float] = []
    stall_count = 0
    exact_throughout = True
    best_lb_iteration = 0
    best_ub_iteration = 0
    it = 0
    deadline = start_time + time_limit  # sole stopping criterion for the subproblem -- no node/label cap

    while max_iterations is None or it < max_iterations:
        elapsed = _time.time() - start_time
        if elapsed > time_limit:
            break
        it += 1

        reduced_cost = {}
        for i in range(inst.n_nodes):
            for j in range(inst.n_nodes):
                if i == j:
                    continue
                reduced_cost[i, j] = inst.distance[i, j] - (lam[j] if j in lam else 0.0)

        sub = _solve_espprc(inst, reduced_cost, bit, ng_mask, deadline=deadline)
        exact_throughout &= sub.exact

        for real_cost, path in sub.completed_routes:
            key = frozenset(path[1:-1])
            if key not in column_pool or real_cost < column_pool[key][0]:
                column_pool[key] = (real_cost, path)

        route_reduced_cost = min(sub.cost, 0.0)
        used_slots = K if sub.cost < -1e-9 else 0
        lb = sum(lam.values()) + used_slots * route_reduced_cost
        history.append(lb)

        # A label search cut off before exhaustion can under-explore and report
        # a reduced cost that is not truly minimal, making `lb` an *invalid*
        # (possibly too high) bound. Only let exactly-solved iterations move
        # best_lb, so the reported bound always stays a certified lower bound.
        improved = False
        if sub.exact:
            improved = lb > best_lb + 1e-6
            if improved:
                best_lb_iteration = it
            best_lb = max(best_lb, lb)

        # Every few iterations, try to turn the pricing problem's by-product
        # columns into a full, better feasible solution (the Lagrangian heuristic).
        if column_pool and it % 5 == 0:
            candidate = _assemble_solution_from_pool(inst, column_pool)
            ok, _ = candidate.validate(inst)
            if ok and candidate.total_distance(inst) < upper_bound - 1e-6:
                upper_bound = candidate.total_distance(inst)
                best_solution = candidate
                best_ub_iteration = it

        # Subgradient of the relaxed constraint "customer i visited exactly once".
        visited_in_best = set(sub.path[1:-1]) if (sub.path and used_slots > 0) else set()
        subgrad = {c: 1.0 - (used_slots if c in visited_in_best else 0.0) for c in customers}
        norm_sq = sum(g * g for g in subgrad.values())

        if norm_sq < 1e-9:
            if verbose:
                print(f"iter {it}: subgradient ~0, stopping (lb={lb:.2f})")
            break

        step = alpha * (upper_bound - best_lb) / norm_sq
        for c in customers:
            lam[c] += step * subgrad[c]

        # Track stagnation of the best bound found so far (monotonic), not the
        # raw per-iteration lb (which oscillates and would never signal a stall).
        if improved:
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= 10:
            alpha /= 2.0
            stall_count = 0

        if verbose:
            print(f"iter {it}: lb={lb:.2f} best_lb={best_lb:.2f} ub={upper_bound:.2f} "
                  f"alpha={alpha:.4f} step={step:.4f}")

        if alpha < 1e-6:
            break

    if column_pool:
        candidate = _assemble_solution_from_pool(inst, column_pool)
        ok, _ = candidate.validate(inst)
        if ok and candidate.total_distance(inst) < upper_bound - 1e-6:
            upper_bound = candidate.total_distance(inst)
            best_solution = candidate
            best_ub_iteration = it

    return LagrangianResult(
        best_lower_bound=best_lb,
        root_lp_bound=root_lp_bound,
        num_vehicles=K,
        multipliers=lam,
        history=history,
        upper_bound=upper_bound,
        best_solution=best_solution,
        runtime=_time.time() - start_time,
        iterations_run=it,
        subproblem_exact_throughout=exact_throughout,
        best_lb_iteration=best_lb_iteration,
        best_ub_iteration=best_ub_iteration,
    )
