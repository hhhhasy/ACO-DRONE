"""Tiny deterministic problem instances used only by unit tests."""
from __future__ import annotations

import os
import sys

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import numpy as np

from aco_solver import ProblemData, _distance_matrices
from drone_physics import (
    DRONE_BATTERY_CAPACITY_WH,
    DRONE_PAYLOAD_CAPACITY_KG,
    DRONE_SPEED_M_S,
    TRUCK_SPEED_M_S,
)


def make_problem(n: int = 7, k: int = 2, d: int = 2, seed: int = 0) -> ProblemData:
    """A small physicalized instance with no distribution-network intent."""
    rng = np.random.RandomState(seed)
    depot = (0.0, 0.0)
    coords = [
        (float(250.0 + 900.0 * rng.rand()),
         float(250.0 + 900.0 * rng.rand()))
        for _ in range(n)
    ]
    demands = [0.0] + [float(round(0.25 + 0.6 * rng.rand(), 3)) for _ in range(n)]
    veh, dro = _distance_matrices(depot, coords)
    return ProblemData(
        num_customers=n,
        num_vehicles=k,
        num_drones_per_vehicle=d,
        vehicle_speed=TRUCK_SPEED_M_S,
        drone_speed=DRONE_SPEED_M_S,
        drone_payload_capacity=DRONE_PAYLOAD_CAPACITY_KG,
        drone_battery_capacity_wh=DRONE_BATTERY_CAPACITY_WH,
        depot_coords=depot,
        customer_coords=coords,
        customer_demands=demands,
        veh_dist_matrix=veh,
        dro_dist_matrix=dro,
        source_file="synthetic",
    )


def collect_served_customers(solution) -> list:
    served = []
    for route in solution.vehicle_routes:
        served.extend(int(c) for c in route[1:-1])
    for sorties in solution.drone_routes:
        for _label, _launch, deliveries, _recovery in sorties:
            served.extend(int(c) for c in deliveries)
    return served
