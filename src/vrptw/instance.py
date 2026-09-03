"""Parsing and data model for VRPTW instances in Solomon format."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VRPTWInstance:
    """A VRPTW instance following the classic Solomon benchmark conventions.

    Node 0 is always the depot. Nodes 1..n are customers.
    Travel time between any pair of nodes equals their Euclidean distance
    (the Solomon convention), so `distance` and `travel_time` share the same matrix.
    """

    name: str
    vehicle_capacity: float
    max_vehicles: int
    x: np.ndarray
    y: np.ndarray
    demand: np.ndarray
    ready_time: np.ndarray
    due_date: np.ndarray
    service_time: np.ndarray
    distance: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.x)
        dx = self.x.reshape(n, 1) - self.x.reshape(1, n)
        dy = self.y.reshape(n, 1) - self.y.reshape(1, n)
        self.distance = np.sqrt(dx**2 + dy**2)

    @property
    def n_nodes(self) -> int:
        return len(self.x)

    @property
    def n_customers(self) -> int:
        return self.n_nodes - 1

    @property
    def customers(self) -> range:
        return range(1, self.n_nodes)

    @property
    def depot(self) -> int:
        return 0

    def travel_time(self, i: int, j: int) -> float:
        return self.distance[i, j]


def load_solomon_instance(path: str | Path, num_customers: int | None = None) -> VRPTWInstance:
    """Load a Solomon-format instance file.

    Args:
        path: path to the .txt instance file.
        num_customers: if given, truncate the instance to the first N customers
            (standard way of deriving the 25/50-customer benchmark subsets).
    """
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]

    name = lines[0].strip()

    vehicle_header_idx = next(i for i, line in enumerate(lines) if line.strip() == "VEHICLE")
    number, capacity = lines[vehicle_header_idx + 2].split()
    max_vehicles = int(number)
    vehicle_capacity = float(capacity)

    customer_header_idx = next(i for i, line in enumerate(lines) if line.strip() == "CUSTOMER")
    # Skip "CUSTOMER", the column-names line and the blank line that follows it.
    data_start = customer_header_idx + 3

    rows = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        rows.append([float(v) for v in line.split()])

    if num_customers is not None:
        # row 0 is the depot, keep it plus the first `num_customers` customers.
        rows = rows[: num_customers + 1]

    arr = np.array(rows)
    return VRPTWInstance(
        name=name,
        vehicle_capacity=vehicle_capacity,
        max_vehicles=max_vehicles,
        x=arr[:, 1],
        y=arr[:, 2],
        demand=arr[:, 3],
        ready_time=arr[:, 4],
        due_date=arr[:, 5],
        service_time=arr[:, 6],
    )
