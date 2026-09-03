from vrptw.instance import load_solomon_instance
from vrptw.lagrangian import lagrangian_relaxation
from vrptw.solvers import solve_branch_and_cut


def test_lagrangian_bound_is_valid_lower_bound():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=10)
    exact = solve_branch_and_cut(inst, time_limit=60)
    lr = lagrangian_relaxation(inst, num_vehicles=exact.num_vehicles, max_iterations=60, time_limit=60)
    # A valid Lagrangian bound never exceeds the true optimum (up to solver/DP tolerance).
    assert lr.best_lower_bound <= exact.total_distance + 1e-3


def test_lagrangian_runs_on_default_settings():
    inst = load_solomon_instance("data/solomon/raw/c101.txt", num_customers=10)
    lr = lagrangian_relaxation(inst, max_iterations=20, time_limit=30)
    assert lr.iterations_run > 0
    assert lr.best_lower_bound > float("-inf")
