"""Route / solution representation, feasibility checks and cost evaluation."""
from __future__ import annotations

from dataclasses import dataclass

from .instance import VRPTWInstance


@dataclass
class Route:
    """A single vehicle route as a sequence of nodes, starting and ending at the depot."""

    nodes: list[int]  # e.g. [0, 3, 7, 0]

    @property
    def customers(self) -> list[int]:
        return self.nodes[1:-1]

    def distance(self, inst: VRPTWInstance) -> float:
        return sum(inst.distance[a, b] for a, b in zip(self.nodes, self.nodes[1:]))

    def load(self, inst: VRPTWInstance) -> float:
        return sum(inst.demand[c] for c in self.customers)

    def is_feasible(self, inst: VRPTWInstance) -> tuple[bool, str]:
        """Check capacity and time-window feasibility, returning (ok, reason_if_not)."""
        if self.load(inst) > inst.vehicle_capacity + 1e-6:
            return False, f"capacity exceeded: load={self.load(inst)} > Q={inst.vehicle_capacity}"

        t = 0.0
        for a, b in zip(self.nodes, self.nodes[1:]):
            arrival = t + inst.travel_time(a, b)
            start_service = max(arrival, inst.ready_time[b])
            if start_service > inst.due_date[b] + 1e-6:
                return False, f"time window violated at node {b}: arrival={start_service} > due={inst.due_date[b]}"
            t = start_service + inst.service_time[b]
        return True, ""


@dataclass
class Solution:
    """A full VRPTW solution: a set of routes covering every customer exactly once."""

    routes: list[Route]

    @property
    def num_vehicles(self) -> int:
        return len(self.routes)

    def total_distance(self, inst: VRPTWInstance) -> float:
        return sum(r.distance(inst) for r in self.routes)

    def cost_tuple(self, inst: VRPTWInstance) -> tuple[int, float]:
        """Lexicographic objective used by the Solomon benchmark: (#vehicles, distance)."""
        return (self.num_vehicles, self.total_distance(inst))

    def validate(self, inst: VRPTWInstance) -> tuple[bool, str]:
        visited = [c for r in self.routes for c in r.customers]
        if sorted(visited) != list(inst.customers):
            return False, "not every customer is visited exactly once"
        for idx, r in enumerate(self.routes):
            ok, reason = r.is_feasible(inst)
            if not ok:
                return False, f"route {idx} infeasible: {reason}"
        return True, ""

    @staticmethod
    def from_arcs(arcs: set[tuple[int, int]], inst: VRPTWInstance) -> "Solution":
        """Rebuild routes from a set of selected arcs (i, j) with i != j, depot = 0."""
        out_next: dict[int, int] = {i: j for i, j in arcs if i == inst.depot or i in inst.customers}
        routes = []
        starts = sorted(j for (i, j) in arcs if i == inst.depot)
        for start in starts:
            nodes = [inst.depot, start]
            cur = start
            while cur != inst.depot:
                cur = out_next[cur]
                nodes.append(cur)
            routes.append(Route(nodes=nodes))
        return Solution(routes=routes)
