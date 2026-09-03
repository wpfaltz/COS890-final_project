import numpy as np
import pytest

from vrptw.instance import load_solomon_instance

INSTANCE_PATH = "data/solomon/raw/c101.txt"


def test_parses_header_and_shape():
    inst = load_solomon_instance(INSTANCE_PATH)
    assert inst.name == "C101"
    assert inst.vehicle_capacity == 200.0
    assert inst.max_vehicles == 25
    assert inst.n_nodes == 101
    assert inst.n_customers == 100
    assert inst.depot == 0


def test_depot_row_matches_file():
    inst = load_solomon_instance(INSTANCE_PATH)
    assert inst.x[0] == 40
    assert inst.y[0] == 50
    assert inst.demand[0] == 0
    assert inst.due_date[0] == 1236


def test_truncates_to_num_customers():
    inst = load_solomon_instance(INSTANCE_PATH, num_customers=25)
    assert inst.n_nodes == 26
    assert inst.n_customers == 25


def test_distance_matrix_symmetric_and_zero_diagonal():
    inst = load_solomon_instance(INSTANCE_PATH, num_customers=10)
    assert np.allclose(inst.distance, inst.distance.T)
    assert np.allclose(np.diag(inst.distance), 0.0)
    expected_01 = np.hypot(inst.x[0] - inst.x[1], inst.y[0] - inst.y[1])
    assert inst.distance[0, 1] == pytest.approx(expected_01)
