from vrptw.instance import load_solomon_instance
from vrptw.solution import Route, Solution


def test_feasible_route_on_toy_instance():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=5)
    # Trivial single-customer routes are always feasible for c101.
    for c in inst.customers:
        route = Route(nodes=[inst.depot, c, inst.depot])
        ok, reason = route.is_feasible(inst)
        assert ok, reason


def test_capacity_violation_detected():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=25)
    overloaded = [c for c in inst.customers if inst.demand[c] > 0]
    # Chain every positive-demand customer into a single route to blow past capacity.
    route = Route(nodes=[inst.depot] + overloaded + [inst.depot])
    ok, reason = route.is_feasible(inst)
    assert not ok
    assert "capacity" in reason


def test_solution_validate_detects_missing_customer():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=3)
    sol = Solution(routes=[Route(nodes=[0, 1, 0]), Route(nodes=[0, 2, 0])])
    ok, reason = sol.validate(inst)
    assert not ok


def test_from_arcs_roundtrip():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=5)
    arcs = {(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 0)}
    sol = Solution.from_arcs(arcs, inst)
    # Only checks structural reconstruction (routes/order); the arcs above are
    # arbitrary and not guaranteed to respect time windows.
    assert sol.num_vehicles == 2
    assert sorted(c for r in sol.routes for c in r.customers) == list(inst.customers)
