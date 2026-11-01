# Drop-in ACO+Tabu solver, now using the 100% dual-sided workload-aware
# service construction (strategy B) by default.
# Intended filename: aco_solver.py. Existing ACOConfig, load_problem,
# AntColonySolver, solve(), and output helpers remain callable as before.
# No changes to the experiment runner are required.
"""Literature-informed Ant Colony System for multi-truck / multi-drone delivery.

Algorithmic layer
-----------------
This is an independent ACO/ACS solver.  It does NOT import the project's ALNS
or GA implementation.  It combines four ideas commonly used in routing ACO:

1. Ant Colony System (ACS) pseudo-random proportional transition and local /
   global pheromone updates (Dorigo & Gambardella, 1997).
2. Dynamic traveller selection for min-max mTSP: the currently least-loaded
   truck is expanded first, which directly targets workload balance.
3. A soft angular-sector bias for assigning customers to trucks.  The bias is
   deliberately soft, so ants can cross sector boundaries when pheromone and
   distance evidence support it.
4. Separate pheromone layers for route edges, truck assignment, service mode
   (truck/drone), and physical-drone choice.
5. Optional short-term tabu memory over complete constructive codes with a
   bounded resampling policy, plus an independent bounded exact-decoding cache.
   Disable tabu_enabled for an original-ACS ablation (and set
   evaluation_cache_size=0 for strict original execution semantics).

Service allocation default
--------------------------
Every ant uses the dual-sided workload-aware construction (strategy B); this is
the ``fixed100`` setting of the ablation study.  Existing runners that do not
touch service-policy options therefore receive 100% strategy B automatically.
The legacy construction and the dynamic workload mixture remain available by
explicit ACOConfig overrides, so the public API is unchanged.

Physical / feasibility layer
----------------------------
All final fitness and feasibility decisions go through the supplied
``evaluator.evaluate_solution`` and ``drone_physics`` modules.  Therefore this
solver preserves the original assumptions exactly:

* physicalized coordinates in metres and demands in kilograms;
* truck Manhattan distance, drone Euclidean distance;
* truck/drone speeds from the shared physics module;
* 3 kg payload capacity;
* 200 Wh battery for every physical drone, with no recharge;
* payload-dependent flight energy and hover energy;
* truck-drone synchronization;
* no maximum-drone-distance constraint.

The decoder is a feasibility mapping, not an ALNS neighbourhood.  Drone-served
customers are packed in ant order and assigned chronological launch/recovery
windows on the owning truck route.  It may demote a drone customer back to the
truck only when required to recover feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import json
import math
import random
import time
import itertools
from collections import OrderedDict, deque

import numpy as np

from drone_physics import (
    DRONE_BATTERY_CAPACITY_WH,
    DRONE_PAYLOAD_CAPACITY_KG,
    DRONE_SPEED_M_S,
    TRUCK_SPEED_M_S,
    energy_optimal_sortie,
    evaluate_sortie_flight,
    hover_energy_wh,
)
from fast_window_physics import prepare_fixed_order_flight
from evaluator import SolutionEvaluation, evaluate_solution

EPS = 1e-9


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
@dataclass
class ProblemData:
    num_customers: int
    num_vehicles: int
    num_drones_per_vehicle: int
    vehicle_speed: float
    drone_speed: float
    drone_payload_capacity: float
    drone_battery_capacity_wh: float
    depot_coords: Tuple[float, float]
    customer_coords: List[Tuple[float, float]]
    customer_demands: List[float]  # 1-based; depot demand is zero
    veh_dist_matrix: np.ndarray
    dro_dist_matrix: np.ndarray
    source_file: str = ""


def _distance_matrices(depot, customers):
    coords = np.asarray([tuple(depot)] + [tuple(x) for x in customers], dtype=float)
    delta = coords[:, None, :] - coords[None, :, :]
    truck = np.abs(delta).sum(axis=2)                 # project definition
    drone = np.sqrt(np.square(delta).sum(axis=2))    # project definition
    return truck, drone


def load_problem(json_path: str) -> ProblemData:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not bool(raw.get("physicalized", False)):
        raise ValueError("dataset must already be physicalized (m / kg)")
    locs = raw["locations"]
    # `locations` is the authoritative customer list -- that is how the rest of
    # the project defines an instance (``load_data.build_model_kwargs`` sets
    # num_customers from len(customer_coords) and never reads
    # ``number_of_nodes``).  That field is inconsistent across the benchmark
    # files: in most it counts customers, in 9 of the 80 大规模 files
    # (M-n101/M-n121/M-n151/M-n200, P-n16/P-n19/P-n20/P-n21/P-n22) it counts the
    # depot as well.  Requiring strict equality made those 9 unloadable, so we
    # take the length and only cross-check that the declared value is one of the
    # two conventions actually seen.
    n = len(locs)
    declared = int(raw["number_of_nodes"])
    if declared not in (n, n + 1):
        raise ValueError(
            "number_of_nodes=%d does not match locations length=%d"
            % (declared, n)
        )

    depot = (float(raw["depot"]["x"]), float(raw["depot"]["y"]))
    coords = [(float(x["x"]), float(x["y"])) for x in locs]
    demands = [0.0] + [float(x["demand"]) for x in locs]
    veh, dro = _distance_matrices(depot, coords)

    # The benchmark has already been physicalized.  Do not rescale here.
    return ProblemData(
        num_customers=n,
        num_vehicles=int(raw["the_number_of_trucks"]),
        num_drones_per_vehicle=int(raw["the_number_of_dronescarried_by_each_truck"]),
        vehicle_speed=TRUCK_SPEED_M_S,
        drone_speed=DRONE_SPEED_M_S,
        drone_payload_capacity=DRONE_PAYLOAD_CAPACITY_KG,
        drone_battery_capacity_wh=DRONE_BATTERY_CAPACITY_WH,
        depot_coords=depot,
        customer_coords=coords,
        customer_demands=demands,
        veh_dist_matrix=veh,
        dro_dist_matrix=dro,
        source_file=json_path,
    )


# ---------------------------------------------------------------------------
# Solution representation
# ---------------------------------------------------------------------------
@dataclass
class AntCode:
    """Constructive ACO phenotype before exact evaluation."""

    master_routes: List[List[int]]
    service: List[int]       # customer -> 0 truck, 1 drone
    drone: List[int]         # customer -> physical drone index on owning truck


@dataclass
class ACOSolution:
    code: AntCode
    vehicle_routes: List[List[int]]
    drone_routes: List[List[List[object]]]
    evaluation: SolutionEvaluation

    @property
    def objective(self) -> float:
        return float(self.evaluation.objective)

    @property
    def feasible(self) -> bool:
        return bool(self.evaluation.feasible)

    def clone(self) -> "ACOSolution":
        # evaluation is immutable enough for our use; deep copy routes/code only.
        return ACOSolution(
            code=AntCode(
                master_routes=[list(r) for r in self.code.master_routes],
                service=list(self.code.service),
                drone=list(self.code.drone),
            ),
            vehicle_routes=[list(r) for r in self.vehicle_routes],
            drone_routes=copy.deepcopy(self.drone_routes),
            evaluation=self.evaluation,
        )


# ---------------------------------------------------------------------------
# Decoder: map ant decisions to an exact feasible truck-drone solution
# ---------------------------------------------------------------------------
class _OneVehicleView:
    def __init__(self, p: ProblemData):
        self.num_vehicles = 1
        self.num_customers = p.num_customers
        self.vehicle_speed = p.vehicle_speed
        self.drone_speed = p.drone_speed
        self.veh_dist_matrix = p.veh_dist_matrix
        self.dro_dist_matrix = p.dro_dist_matrix
        self.customer_demands = p.customer_demands
        self.drone_payload_capacity = p.drone_payload_capacity
        self.drone_battery_capacity_wh = p.drone_battery_capacity_wh


def _pack_drone_customers(problem: ProblemData, seq: Sequence[int]) -> List[List[int]]:
    """Next-fit payload packing in the ant's route order."""
    groups: List[List[int]] = []
    cur: List[int] = []
    load = 0.0
    for c0 in seq:
        c = int(c0)
        q = float(problem.customer_demands[c])
        if q > problem.drone_payload_capacity + EPS:
            return [[c]]
        if cur and load + q > problem.drone_payload_capacity + EPS:
            groups.append(cur)
            cur = []
            load = 0.0
        cur.append(c)
        load += q
    if cur:
        groups.append(cur)
    return groups


def _truck_prefix_times(problem: ProblemData, route: Sequence[int]) -> List[float]:
    prefix = [0.0] * len(route)
    for i in range(1, len(route)):
        prefix[i] = prefix[i - 1] + float(
            problem.veh_dist_matrix[int(route[i - 1]), int(route[i])]
        ) / problem.vehicle_speed
    return prefix


def _choose_windows_for_drone(
    problem: ProblemData,
    route: Sequence[int],
    groups: Sequence[Sequence[int]],
) -> Optional[List[Tuple[int, List[int], int]]]:
    """Assign chronological launch/recovery windows to one physical drone.

    The cheap first pass evaluates each legal window in the ant's customer
    order.  Exact payload-dependent energy-optimal ordering is then applied to
    only the best few candidates.  This keeps ACO construction tractable while
    the final evaluator remains exact.
    """
    if not groups:
        return []
    if len(route) < 2 or len(groups) > len(route) - 1:
        return None

    prefix = _truck_prefix_times(problem, route)
    previous_recovery_idx = 0
    used_wh = 0.0
    result: List[Tuple[int, List[int], int]] = []
    top_k = 3

    for g_idx, g0 in enumerate(groups):
        deliveries = [int(c) for c in g0]
        fixed_order_flight = prepare_fixed_order_flight(problem, deliveries) if deliveries else None
        payload = sum(float(problem.customer_demands[c]) for c in deliveries)
        if not deliveries or payload > problem.drone_payload_capacity + EPS:
            return None

        remaining_groups = len(groups) - g_idx - 1
        latest_recovery = (len(route) - 1) - remaining_groups
        candidates = []

        for li in range(previous_recovery_idx, len(route) - 1):
            if li >= latest_recovery:
                break
            for ri in range(li + 1, latest_recovery + 1):
                launch = int(route[li])
                recovery = int(route[ri])
                flight_t, flight_e = fixed_order_flight(launch, recovery)
                truck_segment_t = float(prefix[ri] - prefix[li])
                truck_wait = max(0.0, flight_t - truck_segment_t)
                hover_t = 0.0 if recovery == 0 else max(0.0, truck_segment_t - flight_t)
                total_e = flight_e + float(hover_energy_wh(hover_t, 0.0))
                # Primary objective proxy: avoid delaying the truck.  Then use
                # battery economically and prefer wider windows for parallelism.
                key = (truck_wait, total_e, flight_e, -(ri - li), li, ri)
                candidates.append((key, li, ri))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        shortlist = list(candidates[:top_k])
        widest = min(candidates, key=lambda x: (-(x[2] - x[1]), x[0]))
        if all((x[1], x[2]) != (widest[1], widest[2]) for x in shortlist):
            shortlist.append(widest)

        best = None
        for _proxy, li, ri in shortlist:
            launch = int(route[li])
            recovery = int(route[ri])
            flight = energy_optimal_sortie(
                launch,
                recovery,
                deliveries,
                problem.dro_dist_matrix,
                problem.customer_demands,
            )
            flight_t = float(flight["flight_time_s"])
            flight_e = float(flight["flight_energy_wh"])
            truck_segment_t = float(prefix[ri] - prefix[li])
            truck_wait = max(0.0, flight_t - truck_segment_t)
            hover_t = 0.0 if recovery == 0 else max(0.0, truck_segment_t - flight_t)
            total_e = flight_e + float(hover_energy_wh(hover_t, 0.0))
            if used_wh + total_e > problem.drone_battery_capacity_wh + EPS:
                continue
            key = (truck_wait, total_e, flight_e, -(ri - li), li, ri)
            if best is None or key < best[0]:
                best = (key, li, ri, flight, total_e)

        if best is None:
            return None
        _, li, ri, flight, total_e = best
        result.append((int(route[li]), [int(c) for c in flight["order"]], int(route[ri])))
        previous_recovery_idx = int(ri)
        used_wh += float(total_e)

    return result


def _decode_vehicle(
    problem: ProblemData,
    vehicle_idx: int,
    master: Sequence[int],
    service_gene: Sequence[int],
    drone_gene: Sequence[int],
) -> Tuple[List[int], List[List[object]]]:
    """Decode one truck's master sequence with feasibility repair."""
    D = problem.num_drones_per_vehicle
    local_service = list(service_gene)
    local_model = _OneVehicleView(problem)

    if D <= 0:
        route = [0] + [int(c) for c in master] + [0]
        return route, []

    for c0 in master:
        c = int(c0)
        if (
            local_service[c] == 1
            and float(problem.customer_demands[c]) > problem.drone_payload_capacity + EPS
        ):
            local_service[c] = 0

    max_repairs = max(3, len(master) + 3)
    for _ in range(max_repairs):
        truck_customers = [int(c) for c in master if local_service[int(c)] == 0]
        route = [0] + truck_customers + [0]
        by_drone: List[List[int]] = [[] for _ in range(D)]
        for c0 in master:
            c = int(c0)
            if local_service[c] == 1:
                by_drone[int(drone_gene[c]) % D].append(c)

        sorties: List[List[object]] = []
        failure_group: Optional[List[int]] = None
        for di, seq in enumerate(by_drone):
            if not seq:
                continue
            packed = _pack_drone_customers(problem, seq)
            if any(
                sum(float(problem.customer_demands[c]) for c in g)
                > problem.drone_payload_capacity + EPS
                for g in packed
            ):
                failure_group = max(packed, key=lambda g: sum(problem.customer_demands[c] for c in g))
                break
            windows = _choose_windows_for_drone(problem, route, packed)
            if windows is None:
                failure_group = max(
                    packed,
                    key=lambda g: (sum(problem.customer_demands[c] for c in g), len(g)),
                )
                break
            label = f"{vehicle_idx + 1}-D{di + 1}"
            for launch, deliveries, recovery in windows:
                sorties.append([label, launch, list(deliveries), recovery])

        if failure_group:
            # Deterministic repair only for infeasibility: return the hardest
            # customer in the failed drone group to the truck.
            c = max(
                failure_group,
                key=lambda x: (float(problem.customer_demands[int(x)]), int(x)),
            )
            local_service[int(c)] = 0
            continue

        # Evaluate this vehicle exactly; hover can still push cumulative SOC over
        # 200 Wh even if the flight-only window filter was acceptable.
        ev = evaluate_solution([route], [sorties], local_model, check_coverage=False)
        if ev.feasible:
            return route, sorties

        # Battery/synchronization repair: demote one customer from the physical
        # drone with the largest approximate flight energy.
        if sorties:
            by_label: Dict[str, List[Tuple[List[int], float]]] = {}
            for label, launch, deliveries, recovery in sorties:
                flight = energy_optimal_sortie(
                    int(launch), int(recovery), deliveries,
                    problem.dro_dist_matrix, problem.customer_demands,
                )
                by_label.setdefault(str(label), []).append(
                    ([int(c) for c in deliveries], float(flight["flight_energy_wh"]))
                )
            label = max(by_label, key=lambda lab: sum(x[1] for x in by_label[lab]))
            deliveries, _ = max(by_label[label], key=lambda x: x[1])
            if deliveries:
                c = max(deliveries, key=lambda x: (problem.customer_demands[int(x)], int(x)))
                local_service[int(c)] = 0
                continue
        break

    # Guaranteed fallback for this vehicle.
    return [0] + [int(c) for c in master] + [0], []


def decode(problem: ProblemData, code: AntCode) -> ACOSolution:
    vehicle_routes: List[List[int]] = []
    drone_routes: List[List[List[object]]] = []
    for v in range(problem.num_vehicles):
        route, sorties = _decode_vehicle(
            problem,
            v,
            code.master_routes[v],
            code.service,
            code.drone,
        )
        vehicle_routes.append(route)
        drone_routes.append(sorties)

    ev = evaluate_solution(vehicle_routes, drone_routes, problem, check_coverage=True)
    if not ev.feasible:
        # One final structural fallback cannot violate customer coverage because
        # master_routes are a partition of all customers.
        vehicle_routes = [[0] + list(map(int, r)) + [0] for r in code.master_routes]
        drone_routes = [[] for _ in range(problem.num_vehicles)]
        ev = evaluate_solution(vehicle_routes, drone_routes, problem, check_coverage=True)
        if not ev.feasible:
            raise RuntimeError("ACO all-truck fallback infeasible: " + ev.reason)

    return ACOSolution(code=code, vehicle_routes=vehicle_routes, drone_routes=drone_routes, evaluation=ev)


# ---------------------------------------------------------------------------
# ACS
# ---------------------------------------------------------------------------
@dataclass
class ACOConfig:
    ants: int = 20
    iterations: int = 80
    alpha: float = 1.0
    beta: float = 2.0
    assignment_alpha: float = 1.0
    service_alpha: float = 1.0
    service_beta: float = 1.5
    q0: float = 0.85
    rho: float = 0.10
    xi: float = 0.10
    sector_bias: float = 2.0
    candidate_list_size: int = 18
    restart_after: int = 45
    seed: int = 13
    verbose_every: int = 10
    # Wall-clock budget in seconds for the whole ``solve()`` call.  0.0 keeps the
    # old behaviour: the run stops on ``iterations`` alone.  Added so the ACS can
    # be compared with ALNS/VND/VNS/GA under an identical time budget.
    time_limit_seconds: float = 0.0
    # Short-term complete-ant memory.  Set False to recover the original ACS
    # construction/evaluation path for fair ablation experiments.
    tabu_enabled: bool = True
    tabu_tenure: int = 40                 # accepted ants before entry expires
    tabu_retry_limit: int = 2            # at most 2 extra constructions per ant
    # Deterministic LRU cache of EXACT decoded solutions (0 disables cache).
    # Memory is local to one solver/instance; there are no cross-run cache hits.
    evaluation_cache_size: int = 1024
    # Default service policy: 100% of ants use the dual-sided workload-aware
    # construction (the same setting as the ablation study's ``fixed100``).
    # Existing callers need no changes and automatically receive this default.
    # The dynamic workload controller remains opt-in: pass
    # service_policy_mode="workload" and set this fraction to the desired base
    # (0.50 reproduces the previous adaptive behaviour).
    # To reproduce the old legacy construction, pass
    # service_policy_mode="fixed", adaptive_service_fraction=0.0.
    adaptive_service_fraction: float = 1.0
    # A conservative approximation of multi-customer drone sorties. It is not
    # used for feasibility or the true objective: exact decode remains in charge.
    service_drone_queue_factor: float = 0.70
    service_proxy_strength: float = 1.0
    # "fixed" + fraction 1.0 is the default (100% strategy B, no random draw).
    # "workload" is retained for the original adaptive controller, opt-in only.
    service_policy_mode: str = "fixed"  # "fixed" | "workload"
    service_auto_min_probability: float = 0.20
    service_auto_max_probability: float = 0.80
    service_auto_workload_reference: float = 0.50
    service_auto_workload_gain: float = 0.45
    service_auto_truck_gain: float = 0.30
    service_auto_drone_share_gain: float = 0.10


class AntColonySolver:
    def __init__(self, problem: ProblemData, config: Optional[ACOConfig] = None):
        self.p = problem
        self.cfg = config or ACOConfig()
        if self.cfg.ants <= 0:
            raise ValueError("ants must be positive")
        if self.cfg.iterations <= 0 and self.cfg.time_limit_seconds <= 0:
            raise ValueError("iterations<=0 requires a positive time_limit_seconds")
        if not (0.0 <= self.cfg.q0 <= 1.0):
            raise ValueError("q0 must be in [0,1]")
        if not (0.0 < self.cfg.rho < 1.0 and 0.0 < self.cfg.xi < 1.0):
            raise ValueError("rho and xi must be in (0,1)")
        if self.cfg.tabu_tenure < 0 or self.cfg.tabu_retry_limit < 0:
            raise ValueError("tabu_tenure and tabu_retry_limit must be >= 0")
        if self.cfg.tabu_enabled and self.cfg.tabu_tenure == 0:
            raise ValueError("tabu_tenure must be positive when tabu is enabled")
        if self.cfg.evaluation_cache_size < 0:
            raise ValueError("evaluation_cache_size must be >= 0")
        if not 0.0 <= self.cfg.adaptive_service_fraction <= 1.0:
            raise ValueError("adaptive_service_fraction must be in [0, 1]")
        if not 0.1 <= self.cfg.service_drone_queue_factor <= 2.0:
            raise ValueError("service_drone_queue_factor must be in [0.1, 2]")
        if not 0.0 <= self.cfg.service_proxy_strength <= 5.0:
            raise ValueError("service_proxy_strength must be in [0, 5]")
        if self.cfg.service_policy_mode not in ("fixed", "workload"):
            raise ValueError("service_policy_mode must be fixed or workload")
        if not (0.0 <= self.cfg.service_auto_min_probability <=
                self.cfg.service_auto_max_probability <= 1.0):
            raise ValueError("0 <= auto min <= auto max <= 1 required")
        if not 0.0 <= self.cfg.service_auto_workload_reference <= 1.0:
            raise ValueError("auto workload reference must be in [0, 1]")
        for attr in ("service_auto_workload_gain", "service_auto_truck_gain",
                     "service_auto_drone_share_gain"):
            if getattr(self.cfg, attr) < 0.0:
                raise ValueError(f"{attr} must be >= 0")
        # Counters are observations, not new fitness or constraints.
        self.policy_probability_sum = 0.0
        self.policy_probability_samples = 0
        self.policy_probability_last = self.cfg.adaptive_service_fraction
        self.policy_workload_last = 0.0
        self.policy_busy_last = 0.0
        self.policy_energy_last = 0.0
        self.policy_truck_factor_last = 0.0
        self.adaptive_service_ants = 0
        self.legacy_service_ants = 0

        # Entries are exact constructive decisions; never use only an objective
        # value as a key (different valid schedules can share a makespan).
        self._evaluation_cache: OrderedDict[tuple, ACOSolution] = OrderedDict()
        self._tabu_expires: Dict[tuple, int] = {}
        self._tabu_expiry_queue = deque()
        self._tabu_clock = 0
        self._reset_search_counters()

        self.rng = random.Random(self.cfg.seed)
        x0, y0 = self.p.depot_coords
        self._customer_angles = [
            (math.atan2(y - y0, x - x0) % (2.0 * math.pi), c)
            for c, (x, y) in enumerate(self.p.customer_coords, start=1)
        ]
        self._nearest = self._build_candidate_lists()
        baseline_code = self._greedy_all_truck_code()
        self.baseline = decode(self.p, baseline_code)
        self.tau0 = 1.0 / max(1.0, self.p.num_customers * self.baseline.objective)
        n = self.p.num_customers
        k = self.p.num_vehicles
        d = max(1, self.p.num_drones_per_vehicle)
        self.tau_edge = np.full((n + 1, n + 1), self.tau0, dtype=float)
        self.tau_assign = np.full((k, n + 1), self.tau0, dtype=float)
        self.tau_service = np.full((n + 1, 2), self.tau0, dtype=float)
        self.tau_drone = np.full((n + 1, d), self.tau0, dtype=float)

        collaborative = decode(self.p, self._collaborative_seed_code(baseline_code.master_routes))
        self.global_best = min(
            (self.baseline, collaborative), key=lambda s: s.objective
        ).clone()
        # Warm starts are already exact-decoded and may be encountered later.
        self._cache_store(self._code_signature(self.baseline.code), self.baseline)
        self._cache_store(self._code_signature(collaborative.code), collaborative)
        self._warm_start_pheromone(self.global_best)
        self.history: List[Dict[str, float]] = []
        # Optional convergence hook and real constructed-ant counter.
        self.history_callback = None
        self.constructed_ants = 0

    def _build_candidate_lists(self) -> List[List[int]]:
        n = self.p.num_customers
        lists = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            others = [j for j in range(1, n + 1) if j != i]
            others.sort(key=lambda j: (float(self.p.veh_dist_matrix[i, j]), j))
            lists[i] = others
        return lists

    def _greedy_all_truck_code(self) -> AntCode:
        """Balanced nearest-neighbour seed used only for tau0/reference."""
        n, k = self.p.num_customers, self.p.num_vehicles
        unvisited = set(range(1, n + 1))
        routes = [[] for _ in range(k)]
        current = [0] * k
        path = [0.0] * k
        while unvisited:
            def proxy(v):
                return path[v] + float(self.p.veh_dist_matrix[current[v], 0])
            v = min(range(k), key=lambda x: (proxy(x), x))
            c = min(unvisited, key=lambda j: (float(self.p.veh_dist_matrix[current[v], j]), j))
            path[v] += float(self.p.veh_dist_matrix[current[v], c])
            current[v] = c
            routes[v].append(c)
            unvisited.remove(c)
        service = [0] * (n + 1)
        drone = [0] * (n + 1)
        return AntCode(routes, service, drone)

    def _collaborative_seed_code(self, routes: Sequence[Sequence[int]]) -> AntCode:
        """Deterministic truck-drone seed for pheromone initialization.

        This is a constructive ACO warm start, not a local-search routine.  On
        each balanced master route it ranks individually feasible customers by
        the same truck-detour/drone-time desirability used by ants and initially
        marks the better ~65% as drone candidates.  The exact decoder then
        repairs battery/window conflicts.
        """
        n = self.p.num_customers
        service = [0] * (n + 1)
        drone = [0] * (n + 1)
        D = self.p.num_drones_per_vehicle
        if D <= 0:
            return AntCode([list(map(int, r)) for r in routes], service, drone)
        for master in routes:
            ranked = []
            for idx, c0 in enumerate(master):
                c = int(c0)
                _eta_t, eta_d = self._service_heuristics(master, idx)
                if eta_d > 0.0:
                    ranked.append((float(eta_d), c))
            ranked.sort(reverse=True)
            take = max(0, int(round(0.65 * len(ranked))))
            loads = [0.0] * D
            for _score, c in ranked[:take]:
                service[c] = 1
                di = min(range(D), key=lambda x: (loads[x], x))
                drone[c] = di
                loads[di] += float(self.p.customer_demands[c])
        return AntCode([list(map(int, r)) for r in routes], service, drone)

    def _warm_start_pheromone(self, seed: ACOSolution) -> None:
        """Bias initial trails toward a feasible collaborative seed."""
        bonus = 1 * self.tau0
        for v, route in enumerate(seed.code.master_routes):
            prev = 0
            for c0 in route:
                c = int(c0)
                self.tau_edge[prev, c] += bonus
                self.tau_edge[c, prev] = self.tau_edge[prev, c]
                self.tau_assign[v, c] += bonus
                prev = c
            self.tau_edge[prev, 0] += bonus
            self.tau_edge[0, prev] = self.tau_edge[prev, 0]

        realized = [0] * (self.p.num_customers + 1)
        realized_drone = [0] * (self.p.num_customers + 1)
        for v, sorties in enumerate(seed.drone_routes):
            for label, _l, deliveries, _r in sorties:
                try:
                    di = max(0, int(str(label).rsplit("D", 1)[1]) - 1)
                except Exception:
                    di = 0
                for c0 in deliveries:
                    c = int(c0)
                    realized[c] = 1
                    realized_drone[c] = di
        for c in range(1, self.p.num_customers + 1):
            mode = int(realized[c])
            self.tau_service[c, mode] += bonus
            if mode == 1 and self.p.num_drones_per_vehicle > 0:
                self.tau_drone[c, realized_drone[c] % self.p.num_drones_per_vehicle] += bonus

    def _sector_membership(self) -> List[int]:
        """Balanced angular sectors with a random rotation per ant."""
        n, k = self.p.num_customers, self.p.num_vehicles
        offset = self.rng.random() * (2.0 * math.pi)
        rotated = sorted((((a - offset) % (2.0 * math.pi), c) for a, c in self._customer_angles))
        sector = [0] * (n + 1)
        for rank, (_a, c) in enumerate(rotated):
            sector[c] = min(k - 1, (rank * k) // n)
        # Randomly rotate truck labels so no truck is tied to one geometric sector.
        shift = self.rng.randrange(k) if k > 1 else 0
        for c in range(1, n + 1):
            sector[c] = (sector[c] + shift) % k
        return sector

    @staticmethod
    def _roulette(rng: random.Random, items: Sequence[int], weights: Sequence[float]) -> int:
        total = float(sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            return int(rng.choice(list(items)))
        x = rng.random() * total
        acc = 0.0
        for item, w in zip(items, weights):
            acc += float(w)
            if acc >= x:
                return int(item)
        return int(items[-1])

    def _acs_pick(self, items: Sequence[int], scores: Sequence[float]) -> int:
        if not items:
            raise ValueError("empty ACS choice")
        if self.rng.random() <= self.cfg.q0:
            m = max(float(x) for x in scores)
            tol = max(1e-300, abs(m) * 1e-12)
            best = [int(i) for i, s in zip(items, scores) if abs(float(s) - m) <= tol]
            return int(self.rng.choice(best))
        return self._roulette(self.rng, items, scores)

    def _local_update_edge(self, i: int, j: int) -> None:
        xi = self.cfg.xi
        value = (1.0 - xi) * self.tau_edge[i, j] + xi * self.tau0
        self.tau_edge[i, j] = value
        self.tau_edge[j, i] = value

    def _local_update_assignment(self, v: int, c: int) -> None:
        xi = self.cfg.xi
        self.tau_assign[v, c] = (1.0 - xi) * self.tau_assign[v, c] + xi * self.tau0

    def _construct_master_routes(self) -> List[List[int]]:
        n, k = self.p.num_customers, self.p.num_vehicles
        sector = self._sector_membership()
        unvisited = set(range(1, n + 1))
        routes = [[] for _ in range(k)]
        current = [0] * k
        travelled = [0.0] * k

        while unvisited:
            # Dynamic traveller selection: expand the current least-loaded route.
            proxies = [
                travelled[v] + float(self.p.veh_dist_matrix[current[v], 0])
                for v in range(k)
            ]
            min_proxy = min(proxies)
            near_min = [v for v, x in enumerate(proxies) if x <= min_proxy * 1.03 + 1e-9]
            v = int(self.rng.choice(near_min))
            i = int(current[v])

            limit = self.cfg.candidate_list_size
            candidates = []
            for c in self._nearest[i]:
                if c in unvisited:
                    candidates.append(c)
                    if limit > 0 and len(candidates) >= limit:
                        break
            if not candidates:
                candidates = sorted(unvisited)

            scores = []
            for c in candidates:
                dist = float(self.p.veh_dist_matrix[i, c])
                radial = float(self.p.veh_dist_matrix[c, 0])
                eta = 1.0 / max(1.0, dist + 0.15 * radial)
                s = (
                    self.tau_edge[i, c] ** self.cfg.alpha
                    * self.tau_assign[v, c] ** self.cfg.assignment_alpha
                    * eta ** self.cfg.beta
                    * (self.cfg.sector_bias if sector[c] == v else 1.0)
                )
                scores.append(float(s))

            c = self._acs_pick(candidates, scores)
            routes[v].append(c)
            travelled[v] += float(self.p.veh_dist_matrix[i, c])
            current[v] = c
            unvisited.remove(c)
            self._local_update_edge(i, c)
            self._local_update_assignment(v, c)

        # Close each master route in the route-pheromone layer.
        for v in range(k):
            if routes[v]:
                self._local_update_edge(routes[v][-1], 0)
        return routes

    def _service_heuristics(self, master: Sequence[int], idx: int) -> Tuple[float, float]:
        c = int(master[idx])
        demand = float(self.p.customer_demands[c])
        if self.p.num_drones_per_vehicle <= 0 or demand > self.p.drone_payload_capacity + EPS:
            return 1.0, 0.0
        prev = 0 if idx == 0 else int(master[idx - 1])
        nxt = 0 if idx + 1 == len(master) else int(master[idx + 1])
        truck_detour_s = max(
            0.0,
            (
                float(self.p.veh_dist_matrix[prev, c])
                + float(self.p.veh_dist_matrix[c, nxt])
                - float(self.p.veh_dist_matrix[prev, nxt])
            ) / self.p.vehicle_speed,
        )
        flight = evaluate_sortie_flight(
            prev, [c], nxt,
            self.p.dro_dist_matrix,
            self.p.customer_demands,
            order=[c],
            speed_m_s=self.p.drone_speed,
        )
        flight_e = float(flight["flight_energy_wh"])
        if flight_e > self.p.drone_battery_capacity_wh + EPS:
            return 1.0, 0.0
        flight_t = max(1.0, float(flight["flight_time_s"]))
        ratio = truck_detour_s / flight_t
        payload_headroom = max(0.0, (self.p.drone_payload_capacity - demand) / self.p.drone_payload_capacity)
        eta_truck = 1.0
        eta_drone = 1.0 + 2.0 * ratio + 0.6 * payload_headroom
        return eta_truck, eta_drone

    def _construct_service_balanced(self, routes: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
        """Dual-sided makespan-proxy policy, with no forced drone-share target.

        For each truck, maintain a virtual truck driving duration after earlier
        drone assignments, and a *soft* per-physical-drone work queue. For both
        candidate service modes, compare their estimated vehicle completion
        times. This permits truck service to win even when a drone sortie is
        individually feasible. It also permits high drone shares if they are
        predicted to reduce the critical work. No hard drone-ratio cap is used.

        The estimate is NOT a feasibility test. Only the unchanged decode() and
        exact evaluator decide whether the final delivery plan is admissible.
        """
        n = self.p.num_customers
        D = self.p.num_drones_per_vehicle
        service = [0] * (n + 1)
        drone = [0] * (n + 1)
        if D <= 0:
            return service, drone
        mat = self.p.veh_dist_matrix
        speed = self.p.vehicle_speed
        qfactor = self.cfg.service_drone_queue_factor
        strength = self.cfg.service_proxy_strength
        # Precompute initial all-truck route loads to attenuate service moves
        # on vehicles whose route is already light (bottleneck-awareness).
        initial_times = []
        for master in routes:
            nodes = [0] + list(master) + [0]
            initial_times.append(sum(float(mat[a, b]) / speed for a, b in zip(nodes, nodes[1:])))
        largest_initial = max(initial_times, default=1.0)
        for v, master in enumerate(routes):
            remaining = [0] + [int(c) for c in master] + [0]
            truck_t = initial_times[v]
            drone_queue = [0.0] * D
            drone_energy = [0.0] * D
            # Important: keep the original genes and physical-drone choices;
            # only the construction *heuristic* is changed.
            vehicle_importance = max(0.35, min(1.0, truck_t / max(1.0, largest_initial)))
            for c0 in master:
                c = int(c0)
                demand = float(self.p.customer_demands[c])
                pos = remaining.index(c)
                prev, nxt = int(remaining[pos-1]), int(remaining[pos+1])
                detour = max(0.0, (float(mat[prev,c]) + float(mat[c,nxt]) - float(mat[prev,nxt])) / speed)
                eligible = demand <= self.p.drone_payload_capacity + EPS
                flight_t = 0.0
                flight_e = 0.0
                if eligible:
                    flight = evaluate_sortie_flight(
                        prev, [c], nxt, self.p.dro_dist_matrix,
                        self.p.customer_demands, order=[c], speed_m_s=self.p.drone_speed)
                    flight_t = float(flight['flight_time_s'])
                    flight_e = float(flight['flight_energy_wh'])
                    eligible = flight_e <= self.p.drone_battery_capacity_wh + EPS
                if not eligible:
                    continue
                # Cheapest feasible queue is only a proxy: real chronological
                # launch/recovery and cumulative SOC are handled by decode().
                best_di = min(range(D), key=lambda di: (drone_queue[di], drone_energy[di], di))
                predicted_queue = list(drone_queue)
                predicted_queue[best_di] += qfactor * flight_t
                # Soft energy congestion avoids a hard cap that previously
                # damaged drone-heavy M instances: a full battery is *not*
                # checked here because actual packing/window energy differs.
                excess = max(0.0, (drone_energy[best_di] + flight_e)
                                  / self.p.drone_battery_capacity_wh - 0.85)
                projected_drone = max(predicted_queue) + excess * 0.15 * flight_t
                before = max(truck_t, max(drone_queue))
                after = max(0.0, truck_t - detour, projected_drone)
                # Positive gain favours drone; negative gain favours truck.
                # Always score *both* directions with reciprocal desirabilities.
                scale = max(90.0, initial_times[v] / max(3, len(master)))
                signed = max(-2.2, min(2.2, vehicle_importance * (before-after) / scale))
                log_ratio = strength * signed
                eta_truck = math.exp(-0.5 * log_ratio)
                eta_drone = math.exp(0.5 * log_ratio)
                scores = [
                    float(self.tau_service[c, 0] ** self.cfg.service_alpha * eta_truck ** self.cfg.service_beta),
                    float(self.tau_service[c, 1] ** self.cfg.service_alpha * eta_drone ** self.cfg.service_beta),
                ]
                mode = self._acs_pick([0, 1], scores)
                service[c] = mode
                xi = self.cfg.xi
                self.tau_service[c, mode] = (1.0-xi) * self.tau_service[c, mode] + xi*self.tau0
                if mode == 1:
                    d_candidates = list(range(D))
                    # Keep physical choices stochastic, but account for expected
                    # work queue rather than payload alone.
                    d_scores = []
                    for di in d_candidates:
                        backlog = drone_queue[di] / max(90.0, initial_times[v])
                        energy_load = drone_energy[di] / max(EPS, self.p.drone_battery_capacity_wh)
                        d_scores.append(float(self.tau_drone[c, di] /
                                             (1.0 + 2.0*backlog + 0.25*energy_load)))
                    di = self._acs_pick(d_candidates, d_scores)
                    drone[c] = int(di)
                    drone_queue[di] += qfactor * flight_t
                    drone_energy[di] += flight_e
                    self.tau_drone[c, di] = (1.0-xi)*self.tau_drone[c, di] + xi*self.tau0
                    truck_t = max(0.0, truck_t - detour)
                    remaining.pop(pos)
        return service, drone

    def _service_policy_probability(self) -> float:
        """Probability of using the new (dual-sided) service heuristic.

        In workload mode, measure the estimated busy duration of EACH physical
        drone from the currently best feasible solution. Flight plus hovering
        occupies that drone; its load is normalized by the owning truck's
        completion time to make the score comparable across instance sizes.

        The pressure signal is deliberately soft and mixed with truck count:
        - few trucks -> more probability of workload-aware service;
        - high per-drone workload, especially when batteries are stressed,
          -> more probability;
        - many trucks / low drone workload -> preserve legacy exploration.
        The controller never enforces a drone-customer quota and keeps both
        strategies alive. True feasibility/makespan still use exact decode().
        """
        cfg = self.cfg
        base = float(cfg.adaptive_service_fraction)
        if cfg.service_policy_mode == "fixed":
            self.policy_workload_last = 0.0
            self.policy_busy_last = 0.0
            self.policy_energy_last = 0.0
            self.policy_truck_factor_last = 0.0
            return base

        k = self.p.num_vehicles
        D = self.p.num_drones_per_vehicle
        if D <= 0:
            self.policy_workload_last = 0.0
            self.policy_busy_last = 0.0
            self.policy_energy_last = 0.0
            self.policy_truck_factor_last = 0.0
            return 0.0

        best = self.global_best
        loads = [[0.0] * D for _ in range(k)]
        for report in best.evaluation.sorties:
            v = int(report.vehicle_idx)
            # Labels are like '2-D1'; the number after D is local to the truck.
            label = str(report.drone_label)
            try:
                di = int(label.rsplit("D", 1)[1]) - 1
            except (ValueError, IndexError):
                continue
            if 0 <= v < k and 0 <= di < D:
                loads[v][di] += max(0.0, float(report.flight_time_s))
                loads[v][di] += max(0.0, float(report.hover_time_s))

        vehicle_times = best.evaluation.vehicle_completion_times
        max_per_truck = [
            min(1.0, max(loads[v]) / max(1.0, float(vehicle_times[v])))
            for v in range(k)
        ]
        # Busy time alone is not a sign of *harmful* congestion: a highly
        # productive drone-heavy schedule can have long busy durations. Also
        # observe cumulative battery stress. Near-saturated batteries make
        # additional drone tasks harder to assign and re-schedule.
        busy_pressure = (0.65 * max(max_per_truck) +
                         0.35 * sum(max_per_truck) / max(1, k))
        # Include unused physical drones with zero load, otherwise a schedule
        # using only one drone could falsely look as though the fleet is full.
        battery_stress = [0.0] * (k * D)
        for (v, label), state in getattr(best.evaluation, 'drones', {}).items():
            try:
                di = int(str(label).rsplit('D', 1)[1]) - 1
            except (ValueError, IndexError):
                continue
            if 0 <= v < k and 0 <= di < D:
                used_fraction = float(state.energy_wh) / max(
                    EPS, self.p.drone_battery_capacity_wh)
                battery_stress[v * D + di] = max(
                    0.0, min(1.0, (used_fraction - 0.60) / 0.40))
        energy_pressure = sum(battery_stress) / max(1, k * D)
        pressure = 0.50 * busy_pressure + 0.50 * energy_pressure
        drone_customers = sum(len(sortie[2]) for truck in best.drone_routes
                              for sortie in truck)
        drone_share = drone_customers / max(1, self.p.num_customers)
        # k=2 => +1/3; k=4 => -1/3. This is a conservative fleet-size prior,
        # NOT a rule that small truck fleets necessarily prefer fewer drones.
        truck_factor = max(-1.0, min(1.0, (3.0 - k) / 3.0))
        p = (base
             + cfg.service_auto_workload_gain *
               (pressure - cfg.service_auto_workload_reference)
             + cfg.service_auto_truck_gain * truck_factor
             + cfg.service_auto_drone_share_gain * (drone_share - 0.65))
        p = min(cfg.service_auto_max_probability,
                max(cfg.service_auto_min_probability, p))
        self.policy_workload_last = float(pressure)
        self.policy_busy_last = float(busy_pressure)
        self.policy_energy_last = float(energy_pressure)
        self.policy_truck_factor_last = float(truck_factor)
        return float(p)

    def service_policy_statistics(self) -> Dict[str, object]:
        return {
            "mode": self.cfg.service_policy_mode,
            "adaptive_ants": int(self.adaptive_service_ants),
            "legacy_ants": int(self.legacy_service_ants),
            "probability_mean": (self.policy_probability_sum /
                                  max(1, self.policy_probability_samples)),
            "probability_last": float(self.policy_probability_last),
            "combined_workload_pressure_last": float(self.policy_workload_last),
            "drone_busy_pressure_last": float(self.policy_busy_last),
            "drone_battery_pressure_last": float(self.policy_energy_last),
            "truck_count_factor_last": float(self.policy_truck_factor_last),
        }

    def _construct_service_and_drone(self, routes: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
        # Choose once per ant; probability changes only with the best known
        # decoded schedule, keeping the decision cheap and reproducible.
        p = self._service_policy_probability()
        self.policy_probability_last = p
        self.policy_probability_sum += p
        self.policy_probability_samples += 1
        if p > 0.0 and (p >= 1.0 or self.rng.random() < p):
            self.adaptive_service_ants += 1
            return self._construct_service_balanced(routes)
        self.legacy_service_ants += 1
        n = self.p.num_customers
        D = self.p.num_drones_per_vehicle
        service = [0] * (n + 1)
        drone = [0] * (n + 1)

        for v, master in enumerate(routes):
            tentative_load = [0.0] * max(1, D)
            for idx, c0 in enumerate(master):
                c = int(c0)
                eta_t, eta_d = self._service_heuristics(master, idx)
                if eta_d <= 0.0:
                    service[c] = 0
                    continue
                scores = [
                    float(self.tau_service[c, 0] ** self.cfg.service_alpha * eta_t ** self.cfg.service_beta),
                    float(self.tau_service[c, 1] ** self.cfg.service_alpha * eta_d ** self.cfg.service_beta),
                ]
                mode = self._acs_pick([0, 1], scores)
                service[c] = mode
                # Local pheromone update on the selected service state.
                xi = self.cfg.xi
                self.tau_service[c, mode] = (
                    (1.0 - xi) * self.tau_service[c, mode] + xi * self.tau0
                )

                if mode == 1 and D > 0:
                    d_candidates = list(range(D))
                    d_scores = []
                    q = float(self.p.customer_demands[c])
                    for di in d_candidates:
                        balance_eta = 1.0 / (1.0 + tentative_load[di])
                        d_scores.append(float(self.tau_drone[c, di] * balance_eta))
                    di = self._acs_pick(d_candidates, d_scores)
                    drone[c] = int(di)
                    tentative_load[di] += q
                    self.tau_drone[c, di] = (
                        (1.0 - xi) * self.tau_drone[c, di] + xi * self.tau0
                    )
        return service, drone

    @staticmethod
    def _code_signature(code: AntCode) -> tuple:
        """Exact genotype identity, including assigned physical drones.

        The drone gene is irrelevant when a customer is served by the truck;
        masking those entries prevents false misses without merging schedules
        whose decoded truck/drone decisions may differ.  Keep truck identities:
        although trucks are symmetric physically, their pheromone layers are not.
        """
        routes = tuple(tuple(int(c) for c in route) for route in code.master_routes)
        service = tuple(int(v) for v in code.service[1:])
        drone = tuple(
            int(code.drone[c]) if int(code.service[c]) == 1 else -1
            for c in range(1, len(code.service))
        )
        return routes, service, drone

    def _reset_search_counters(self) -> None:
        self.ant_proposals = 0
        self.tabu_hits = 0
        self.tabu_retries = 0
        self.tabu_forced_accepts = 0
        self.tabu_aspiration_accepts = 0
        self.evaluation_cache_hits = 0
        self.exact_decode_calls = 0

    def _cache_store(self, signature: tuple, sol: ACOSolution) -> None:
        if self.cfg.evaluation_cache_size <= 0:
            return
        self._evaluation_cache[signature] = sol.clone()
        self._evaluation_cache.move_to_end(signature)
        while len(self._evaluation_cache) > self.cfg.evaluation_cache_size:
            self._evaluation_cache.popitem(last=False)

    def _decode_or_reuse(self, code: AntCode, signature: tuple) -> ACOSolution:
        if self.cfg.evaluation_cache_size > 0:
            cached = self._evaluation_cache.get(signature)
            if cached is not None:
                self.evaluation_cache_hits += 1
                self._evaluation_cache.move_to_end(signature)
                result = cached.clone()
                # Preserve this ant's actual genes (truck-served customers can
                # have different, intentionally ignored physical-drone genes).
                result.code = code
                return result
        self.exact_decode_calls += 1
        sol = decode(self.p, code)
        self._cache_store(signature, sol)
        return sol

    def _expire_tabu(self) -> None:
        while (self._tabu_expiry_queue
               and self._tabu_expiry_queue[0][0] <= self._tabu_clock):
            until, signature = self._tabu_expiry_queue.popleft()
            if self._tabu_expires.get(signature) == until:
                del self._tabu_expires[signature]

    def _remember_tabu(self, signature: tuple) -> None:
        if not self.cfg.tabu_enabled:
            return
        self._tabu_clock += 1
        self._expire_tabu()
        until = self._tabu_clock + self.cfg.tabu_tenure
        self._tabu_expires[signature] = until
        self._tabu_expiry_queue.append((until, signature))

    def tabu_statistics(self) -> Dict[str, int]:
        """Counters since solve() began, excluding constructor warm starts."""
        return {
            "constructed_ants": int(self.constructed_ants),
            "ant_proposals": int(self.ant_proposals),
            "tabu_hits": int(self.tabu_hits),
            "tabu_retries": int(self.tabu_retries),
            "tabu_forced_accepts": int(self.tabu_forced_accepts),
            "tabu_aspiration_accepts": int(self.tabu_aspiration_accepts),
            "exact_decode_calls": int(self.exact_decode_calls),
            "evaluation_cache_hits": int(self.evaluation_cache_hits),
            "active_tabu_entries": int(len(self._tabu_expires)),
            "evaluation_cache_entries": int(len(self._evaluation_cache)),
        }

    def construct_ant(self, budget_start: Optional[float] = None) -> ACOSolution:
        """Produce exactly one ant for ACS pheromone/global-best semantics.

        Tabu is checked BEFORE decoding.  A tabu hit triggers a bounded new
        constructive attempt.  If attempts are exhausted (or the wall-clock
        budget has expired), the last proposal is admitted, preferably via
        cache.  This keeps a small/exhausted search space from deadlocking.
        An aspiration bypass protects an unexpectedly better cached candidate.

        ACS local pheromone updates during construction are retained, including
        for rejected proposals; this diversification is part of the tabu variant.
        """
        retry_limit = self.cfg.tabu_retry_limit if self.cfg.tabu_enabled else 0
        for attempt in range(retry_limit + 1):
            routes = self._construct_master_routes()
            service, drone = self._construct_service_and_drone(routes)
            code = AntCode(routes, service, drone)
            self.ant_proposals += 1
            signature = self._code_signature(code)

            if self.cfg.tabu_enabled:
                self._expire_tabu()
                if signature in self._tabu_expires:
                    self.tabu_hits += 1
                    cached = self._evaluation_cache.get(signature)
                    aspiration = (
                        cached is not None and cached.feasible
                        and cached.objective < self.global_best.objective - EPS
                    )
                    if aspiration:
                        self.tabu_aspiration_accepts += 1
                    elif (attempt < retry_limit and
                          (budget_start is None or not self._deadline_reached(budget_start))):
                        self.tabu_retries += 1
                        continue
                    else:
                        self.tabu_forced_accepts += 1

            solution = self._decode_or_reuse(code, signature)
            self.constructed_ants += 1
            self._remember_tabu(signature)
            return solution
        raise AssertionError("bounded tabu retries must return an ant")

    def _pheromone_restart(self) -> None:
        # Smooth trails toward tau0 instead of fully erasing search history.
        w = 0.65
        self.tau_edge[:] = w * self.tau_edge + (1.0 - w) * self.tau0
        self.tau_assign[:] = w * self.tau_assign + (1.0 - w) * self.tau0
        self.tau_service[:] = w * self.tau_service + (1.0 - w) * self.tau0
        self.tau_drone[:] = w * self.tau_drone + (1.0 - w) * self.tau0

    def _global_update(self, best: ACOSolution) -> None:
        rho = self.cfg.rho
        floor = self.tau0 * 0.02
        ceiling = self.tau0 * 500.0

        # Evaporation on all pheromone layers.
        self.tau_edge *= (1.0 - rho)
        self.tau_assign *= (1.0 - rho)
        self.tau_service *= (1.0 - rho)
        self.tau_drone *= (1.0 - rho)

        delta = 1.0 / max(best.objective, EPS)
        # Route/assignment reinforcement on master routes.  This preserves the
        # constructive decisions made by the ant, not decoder repair artifacts.
        for v, route in enumerate(best.code.master_routes):
            prev = 0
            for c0 in route:
                c = int(c0)
                self.tau_edge[prev, c] += rho * delta
                self.tau_edge[c, prev] = self.tau_edge[prev, c]
                self.tau_assign[v, c] += rho * delta
                prev = c
            self.tau_edge[prev, 0] += rho * delta
            self.tau_edge[0, prev] = self.tau_edge[prev, 0]

        # Reinforce the realized service mode rather than a repaired genetic
        # intention.  A customer present on a truck route is truck-served;
        # otherwise its drone label identifies the physical drone.
        realized_service = [0] * (self.p.num_customers + 1)
        realized_drone = [0] * (self.p.num_customers + 1)
        for v, route in enumerate(best.vehicle_routes):
            for c in route[1:-1]:
                realized_service[int(c)] = 0
            for sortie in best.drone_routes[v]:
                label, _l, deliveries, _r = sortie
                try:
                    di = max(0, int(str(label).rsplit("D", 1)[1]) - 1)
                except Exception:
                    di = 0
                for c in deliveries:
                    realized_service[int(c)] = 1
                    realized_drone[int(c)] = di
        for c in range(1, self.p.num_customers + 1):
            m = int(realized_service[c])
            self.tau_service[c, m] += rho * delta
            if m == 1 and self.p.num_drones_per_vehicle > 0:
                di = int(realized_drone[c]) % self.p.num_drones_per_vehicle
                self.tau_drone[c, di] += rho * delta

        np.clip(self.tau_edge, floor, ceiling, out=self.tau_edge)
        np.clip(self.tau_assign, floor, ceiling, out=self.tau_assign)
        np.clip(self.tau_service, floor, ceiling, out=self.tau_service)
        np.clip(self.tau_drone, floor, ceiling, out=self.tau_drone)

    def _emit_history(self, event_type: str, iteration: int) -> None:
        callback = getattr(self, "history_callback", None)
        if callback is None:
            return
        value = float(self.global_best.objective)
        if not math.isfinite(value):
            return
        callback({
            "best_makespan_s": value,
            "iteration_or_generation": int(iteration),
            "evaluations": int(self.constructed_ants),
            "event_type": str(event_type),
        })

    def _deadline_reached(self, start: float) -> bool:
        """True once the ``time_limit_seconds`` budget is spent (0.0 = no limit)."""
        return (
            self.cfg.time_limit_seconds > 0
            and time.perf_counter() - start >= self.cfg.time_limit_seconds
        )

    def solve(self) -> Tuple[ACOSolution, List[Dict[str, float]], float]:
        start = time.perf_counter()
        stagnation = 0
        self.time_limit_hit = False
        self.iterations_used = 0
        self.constructed_ants = 0
        self._reset_search_counters()
        self._tabu_clock = 0
        self._tabu_expires.clear()
        self._tabu_expiry_queue.clear()

        iteration_source = (
            itertools.count(1) if self.cfg.iterations <= 0
            else range(1, self.cfg.iterations + 1)
        )
        for it in iteration_source:
            # Wall-clock budget, checked once per iteration.  The previous
            # iteration was complete (its ants, pheromone update and history
            # record all went through), so stopping here is clean.
            if self._deadline_reached(start):
                self.time_limit_hit = True
                break
            self.iterations_used = it
            # Fine-grained budget checkpoint: one iteration on the largest
            # instance (M-n200) costs ~2.5 s, so checking only between
            # iterations overshot a 10 s budget by 25%.  Checking once per ant
            # bounds the overshoot by a single ant construction.
            colony = []
            for _ in range(self.cfg.ants):
                if self._deadline_reached(start):
                    self.time_limit_hit = True
                    break
                colony.append(self.construct_ant(budget_start=start))
            feasible = [s for s in colony if s.feasible and math.isfinite(s.objective)]
            if self.time_limit_hit:
                # Incomplete iteration: fold the best ant that was actually
                # built into the global best and stop.  The pheromone update and
                # the history record are deliberately skipped -- a partial
                # colony did not complete an iteration, so letting it lay
                # pheromone would change the algorithm's semantics.
                if feasible:
                    partial_best = min(feasible, key=lambda s: s.objective)
                    if partial_best.objective < self.global_best.objective - 1e-9:
                        self.global_best = partial_best.clone()
                        self._emit_history("improvement", self.iterations_used)
                break
            if not feasible:
                raise RuntimeError("ACO produced no feasible ants")
            iteration_best = min(feasible, key=lambda s: s.objective)
            improved = iteration_best.objective < self.global_best.objective - 1e-9
            if improved:
                self.global_best = iteration_best.clone()
                self._emit_history("improvement", it)
                stagnation = 0
            else:
                stagnation += 1

            # ACS convention: reinforce the global-best solution.
            self._global_update(self.global_best)
            if self.cfg.restart_after > 0 and stagnation >= self.cfg.restart_after:
                self._pheromone_restart()
                stagnation = 0

            mean_obj = float(sum(s.objective for s in feasible) / len(feasible))
            rec = {
                "iteration": float(it),
                "iteration_best": float(iteration_best.objective),
                "global_best": float(self.global_best.objective),
                "mean_feasible": mean_obj,
                "feasible_ants": float(len(feasible)),
                "ant_proposals": float(self.ant_proposals),
                "tabu_hits": float(self.tabu_hits),
                "tabu_retries": float(self.tabu_retries),
                "tabu_forced_accepts": float(self.tabu_forced_accepts),
                "exact_decode_calls": float(self.exact_decode_calls),
                "evaluation_cache_hits": float(self.evaluation_cache_hits),
            }
            self.history.append(rec)
            if self.cfg.verbose_every > 0 and (
                it == 1 or it % self.cfg.verbose_every == 0 or it == self.cfg.iterations
            ):
                print(
                    f"[ACO] iter={it:4d} "
                    f"iter_best={iteration_best.objective:.4f} "
                    f"global_best={self.global_best.objective:.4f} "
                    f"mean={mean_obj:.4f}"
                )

        runtime = time.perf_counter() - start
        # Re-run the exact evaluator before returning the published solution.
        exact = evaluate_solution(
            self.global_best.vehicle_routes,
            self.global_best.drone_routes,
            self.p,
            check_coverage=True,
        )
        if not exact.feasible:
            raise RuntimeError("final ACO solution failed exact evaluator: " + exact.reason)
        self.global_best.evaluation = exact
        return self.global_best, list(self.history), runtime


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _drone_index_paths(vehicle_routes, drone_routes):
    """ALNS-compatible display: launch/recovery are truck-route indices."""
    grouped: Dict[str, List[List[int]]] = {}
    for v, sorties in enumerate(drone_routes):
        route = [int(x) for x in vehicle_routes[v]]
        first = {}
        last = {}
        for idx, node in enumerate(route):
            if node not in first:
                first[node] = idx
            last[node] = idx
        for label, launch, deliveries, recovery in sorties:
            li = first.get(int(launch), int(launch))
            ri = last.get(int(recovery), int(recovery))
            grouped.setdefault(str(label), []).append(
                [int(li)] + [int(c) for c in deliveries] + [int(ri)]
            )
    return grouped


def format_solution_details(problem: ProblemData, solution: ACOSolution) -> str:
    ev = evaluate_solution(solution.vehicle_routes, solution.drone_routes, problem, True)
    lines = ["truck_route=", str([[int(x) for x in r] for r in solution.vehicle_routes])]
    grouped = _drone_index_paths(solution.vehicle_routes, solution.drone_routes)
    labels = []
    for v in range(problem.num_vehicles):
        for di in range(problem.num_drones_per_vehicle):
            labels.append(f"{v + 1}-D{di + 1}")
    for idx, label in enumerate(labels, 1):
        lines.append(f"drone_routes{idx}=")
        lines.append(str(grouped.get(label, [])))

    lines += [
        "",
        f"总时间成本: {ev.makespan:.2f}",
        "",
        "===== 无人机能耗与电池 (Wh, 单位: m/s/kg/Wh) =====",
        f"battery_capacity_wh={problem.drone_battery_capacity_wh:.1f}",
        f"payload_capacity_kg={problem.drone_payload_capacity:.2f}",
        f"total_flight_energy_wh={ev.total_flight_energy_wh:.4f}",
        f"total_hover_energy_wh={ev.total_hover_energy_wh:.4f}",
    ]
    for rep in ev.sorties:
        lines.append(
            "sortie vehicle={} drone={} launch={} rendezvous={} order={} "
            "start_payload_kg={:.4f} flight_time_s={:.2f} "
            "flight_energy_wh={:.4f} hover_time_s={:.2f} hover_energy_wh={:.4f} "
            "total_energy_wh={:.4f}".format(
                rep.vehicle_idx + 1,
                rep.drone_label,
                rep.launch,
                rep.rendezvous,
                rep.order,
                rep.start_payload_kg,
                rep.flight_time_s,
                rep.flight_energy_wh,
                rep.hover_time_s,
                rep.hover_energy_wh,
                rep.total_energy_wh,
            )
        )
    for (v, label), state in sorted(ev.drones.items()):
        lines.append(
            "drone vehicle={} label={} sorties={} used_wh={:.4f} remaining_energy_wh={:.4f}".format(
                int(v) + 1,
                label,
                len(state.sortie_keys),
                state.energy_wh,
                state.remaining_energy_wh,
            )
        )
    lines.append(f"makespan_s={ev.makespan:.4f}")
    lines.append("")
    lines.append(f"总时长: {ev.makespan:.2f}")
    return "\n".join(lines) + "\n"


def solution_to_dict(
    problem: ProblemData,
    solution: ACOSolution,
    history: Optional[Sequence[Dict[str, float]]] = None,
    runtime_seconds: Optional[float] = None,
    config: Optional[ACOConfig] = None,
) -> Dict[str, object]:
    ev = evaluate_solution(solution.vehicle_routes, solution.drone_routes, problem, True)
    return {
        "algorithm": "literature-informed-ant-colony-system",
        "source_file": problem.source_file,
        "feasible": bool(ev.feasible),
        "reason": str(ev.reason),
        "makespan_s": float(ev.makespan),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "problem": {
            "num_customers": problem.num_customers,
            "num_vehicles": problem.num_vehicles,
            "num_drones_per_vehicle": problem.num_drones_per_vehicle,
            "truck_speed_m_s": problem.vehicle_speed,
            "drone_speed_m_s": problem.drone_speed,
            "drone_payload_capacity_kg": problem.drone_payload_capacity,
            "drone_battery_capacity_wh": problem.drone_battery_capacity_wh,
            "drone_max_distance_used": False,
        },
        "config": None if config is None else vars(config),
        "master_routes": [[int(x) for x in r] for r in solution.code.master_routes],
        "vehicle_routes": [[int(x) for x in r] for r in solution.vehicle_routes],
        "drone_routes": copy.deepcopy(solution.drone_routes),
        "vehicle_completion_times_s": [float(x) for x in ev.vehicle_completion_times],
        "total_flight_energy_wh": float(ev.total_flight_energy_wh),
        "total_hover_energy_wh": float(ev.total_hover_energy_wh),
        "drones": {
            f"{int(v) + 1}:{label}": {
                "used_wh": float(state.energy_wh),
                "remaining_energy_wh": float(state.remaining_energy_wh),
                "sorties": len(state.sortie_keys),
            }
            for (v, label), state in sorted(ev.drones.items())
        },
        "history": list(history or []),
    }


def save_result_json(path: str, problem: ProblemData, solution: ACOSolution, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution_to_dict(problem, solution, **kwargs), f, ensure_ascii=False, indent=2)


def save_solution_txt(path: str, problem: ProblemData, solution: ACOSolution) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_solution_details(problem, solution))
