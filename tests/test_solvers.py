import pytest

from vrptw.instance import load_solomon_instance
from vrptw.solvers import solve_branch_and_bound, solve_branch_and_cut


@pytest.fixture(scope="module")
def small_instance():
    return load_solomon_instance("data/solomon/raw/c101.txt", num_customers=10)


def test_branch_and_cut_finds_feasible_optimum(small_instance):
    result = solve_branch_and_cut(small_instance, time_limit=60)
    assert result.status == "optimal"
    assert result.solution is not None
    ok, reason = result.solution.validate(small_instance)
    assert ok, reason
    assert result.solution.num_vehicles == result.num_vehicles
    assert result.solution.total_distance(small_instance) == pytest.approx(result.total_distance, abs=1e-4)


def test_branch_and_bound_matches_branch_and_cut(small_instance):
    bb = solve_branch_and_bound(small_instance, time_limit=60)
    bc = solve_branch_and_cut(small_instance, time_limit=60)
    assert bb.status == "optimal"
    assert bb.num_vehicles == bc.num_vehicles
    assert bb.total_distance == pytest.approx(bc.total_distance, abs=1e-4)
