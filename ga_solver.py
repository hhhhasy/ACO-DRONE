"""Literature-informed GA V3 for multi-truck / multi-drone cooperative delivery.

This module is independent from the ALNS implementation.  It uses only the
project's shared physical model (``drone_physics``) and exact evaluator
(``evaluator``), so GA and ALNS are judged by the same physics/objective.

Chromosome
==========
1) ``route_string``: a giant customer order plus K-1 separator tokens.  The
   customer order is genetic; at decoding time a min-max dynamic-programming
   split places the separators so the reference truck tours are balanced.
2) ``service[c]``: 0 truck, 1 drone (type-aware customer gene).
3) ``drone[c]``: physical drone index on the owning truck.

Drone customers assigned to the same physical drone are packed into payload-
feasible sorties in chromosome order.  Launch/recovery are *not* tied to
adjacent truck anchors: the decoder searches chronologically valid windows over
the whole truck route, including depot->depot, and ranks them by added truck
waiting time then energy.  This permits long parallel truck/drone operations.
Exact payload, flight energy, hover energy, synchronization and cumulative
200 Wh SOC are finally checked by ``evaluator.evaluate_solution``.

Search operators remain a classic GA:
- tournament selection
- elitism
- type-aware order crossover + route-segment-preserving crossover
- adaptive relocation / swap / inversion / bottleneck-transfer mutation
- block-aware service mutation and physical-drone mutation
- stagnation-triggered diversity injection

No ALNS destroy/repair operator is imported or called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import random
import statistics
import time
import itertools

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
    customer_demands: List[float]  # 1-based, depot=0
    veh_dist_matrix: np.ndarray
    dro_dist_matrix: np.ndarray
    source_file: str = ""


def _distance_matrices(depot, customers):
    coords = np.asarray([tuple(depot)] + [tuple(x) for x in customers], dtype=float)
    delta = coords[:, None, :] - coords[None, :, :]
    # Keep exactly the distance definitions used by the supplied project model.
    truck = np.abs(delta).sum(axis=2)  # Manhattan
    drone = np.sqrt(np.square(delta).sum(axis=2))  # Euclidean
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
# Chromosome and decoded solution
# ---------------------------------------------------------------------------
@dataclass
class Chromosome:
    route_string: List[int]
    service: List[int]  # 1-based: 0 truck, 1 drone
    drone: List[int]    # 1-based: 0..D-1
    fitness: float = math.inf
    solution: Optional["DecodedSolution"] = None

    def clone(self):
        return Chromosome(
            route_string=list(self.route_string),
            service=list(self.service),
            drone=list(self.drone),
            fitness=float(self.fitness),
            solution=self.solution,
        )


@dataclass
class Sortie:
    truck: int
    drone: int
    drone_label: str
    launch: int
    deliveries: List[int]
    recovery: int
    launch_index: int
    recovery_index: int
    flight_time_s: float = 0.0
    flight_energy_wh: float = 0.0
    hover_time_s: float = 0.0
    hover_energy_wh: float = 0.0

    @property
    def total_energy_wh(self):
        return self.flight_energy_wh + self.hover_energy_wh


@dataclass
class RouteSolution:
    truck: int
    master_sequence: List[int]
    route: List[int]
    sorties: List[Sortie]
    completion_time_s: float = 0.0
    drone_energy_wh: List[float] = field(default_factory=list)


@dataclass
class DecodedSolution:
    routes: List[RouteSolution]
    feasible: bool
    reason: str
    makespan_s: float
    exact_evaluation: Optional[SolutionEvaluation] = field(default=None, repr=False)

    @property
    def vehicle_routes(self):
        return [list(r.route) for r in self.routes]

    @property
    def drone_routes(self):
        return [
            [[s.drone_label, s.launch, list(s.deliveries), s.recovery] for s in r.sorties]
            for r in self.routes
        ]


def separators(k: int) -> List[int]:
    return [-i for i in range(1, k)]


def split_route_string(route_string: Sequence[int], k: int) -> List[List[int]]:
    sep_set = set(separators(k))
    groups: List[List[int]] = [[]]
    for token in route_string:
        if token in sep_set:
            groups.append([])
        else:
            groups[-1].append(int(token))
    while len(groups) < k:
        groups.append([])
    return groups[:k]

def split_customer_order_minmax(
    problem: ProblemData,
    route_string: Sequence[int],
    k: Optional[int] = None,
    service: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Optimal contiguous min-max split of a giant-tour customer order.

    When ``service`` is supplied, segment cost is the reference truck-tour
    distance through only the truck-served customers in that segment.  This is
    the right surrogate for the current chromosome: it prevents a solution with
    very few but geographically disastrous truck anchors from becoming the
    makespan bottleneck.  Without ``service`` every customer is treated as a
    truck node; that mode is used only to normalize raw route strings.
    """
    order = [int(x) for x in route_string if int(x) > 0]
    n = len(order)
    k = problem.num_vehicles if k is None else int(k)
    if k <= 1:
        return [order]
    if n < k:
        raise ValueError('number of customers must be >= number of vehicles')

    d = problem.veh_dist_matrix
    # Precompute reference truck-route cost for every contiguous segment.
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        last = 0
        path = 0.0
        has_truck = False
        for j in range(i, n):
            c = order[j]
            is_truck = service is None or int(service[c]) == 0
            if is_truck:
                path += float(d[last, c])
                last = c
                has_truck = True
            cost[i][j] = path + (float(d[last, 0]) if has_truck else 0.0)

    if n < 85:
        inf = float('inf')
        dp = [[inf] * (n + 1) for _ in range(k + 1)]
        prev = [[-1] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0.0
        for v in range(1, k + 1):
            for j in range(v, n - (k - v) + 1):
                best = inf
                best_i = -1
                for i in range(v - 1, j):
                    old = dp[v - 1][i]
                    if old == inf:
                        continue
                    val = max(old, cost[i][j - 1])
                    if val < best - EPS:
                        best = val
                        best_i = i
                dp[v][j] = best
                prev[v][j] = best_i

    else:
        inf = float('inf')
        # C-level contiguous candidate comparisons; O(k*n^2) decisions retained.
        # Near ties fall back to the original sequential EPS rule.
        cost_array = np.asarray(cost, dtype=np.float64)
        dp = np.full((k + 1, n + 1), inf, dtype=np.float64)
        prev = [[-1] * (n + 1) for _ in range(k + 1)]
        dp[0, 0] = 0.0
        for v in range(1, k + 1):
            for j in range(v, n - (k - v) + 1):
                scores = np.maximum(dp[v - 1, v - 1:j], cost_array[v - 1:j, j - 1])
                m = float(np.min(scores))
                if not math.isfinite(m):
                    continue
                idx = int(np.argmin(scores))
                # Multiple virtually identical costs require the original strict
                # EPS acceptance order (argmin alone could choose differently).
                if int(np.count_nonzero(scores <= m + EPS)) > 1:
                    best = inf
                    best_i = -1
                    for i in range(v - 1, j):
                        old = float(dp[v - 1, i])
                        if old == inf:
                            continue
                        val = max(old, cost[i][j - 1])
                        if val < best - EPS:
                            best = val
                            best_i = i
                    dp[v, j] = best
                    prev[v][j] = best_i
                else:
                    dp[v, j] = m
                    prev[v][j] = v - 1 + idx

    cuts = [n]
    j = n
    for v in range(k, 0, -1):
        i = prev[v][j]
        if i < 0:
            return [list(map(int, x)) for x in np.array_split(order, k)]
        cuts.append(i)
        j = i
    cuts.reverse()
    return [order[cuts[v]:cuts[v + 1]] for v in range(k)]


def _invalidate(c: Chromosome) -> None:
    c.fitness = math.inf
    c.solution = None


# ---------------------------------------------------------------------------
# Decoder
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


def _truck_detour_saving(problem: ProblemData, prev: int, c: int, nxt: int) -> float:
    d = problem.veh_dist_matrix
    saved_m = float(d[prev, c] + d[c, nxt] - d[prev, nxt])
    return saved_m / problem.vehicle_speed


def _least_attractive_for_drone(
    problem: ProblemData, deliveries: Sequence[int], launch: int, recovery: int
) -> int:
    """Customer to return to truck first during feasibility repair.

    The criterion is deterministic and local: low truck detour saving first,
    then high demand.  It is only a decoder feasibility repair, not a search
    neighborhood.
    """
    if len(deliveries) == 1:
        return int(deliveries[0])
    return min(
        deliveries,
        key=lambda c: (
            _truck_detour_saving(problem, launch, int(c), recovery),
            -problem.customer_demands[int(c)],
            int(c),
        ),
    )


def _pack_drone_customers(problem: ProblemData, customers: Sequence[int]) -> List[List[int]]:
    """Next-fit packing in chromosome order, producing full-ish <=3 kg sorties.

    The order of customer *sets* follows the chromosome.  The exact visit order
    inside each sortie is still optimized by ``energy_optimal_sortie``.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    load = 0.0
    for c0 in customers:
        c = int(c0)
        q = float(problem.customer_demands[c])
        if q > problem.drone_payload_capacity + EPS:
            # Caller will demote this impossible drone customer to the truck.
            return [[c]]
        if current and load + q > problem.drone_payload_capacity + EPS:
            groups.append(current)
            current = []
            load = 0.0
        current.append(c)
        load += q
    if current:
        groups.append(current)
    return groups


def _truck_prefix_times(problem: ProblemData, route: Sequence[int]) -> List[float]:
    prefix = [0.0] * len(route)
    for i in range(1, len(route)):
        prefix[i] = prefix[i - 1] + float(problem.veh_dist_matrix[route[i - 1], route[i]]) / problem.vehicle_speed
    return prefix


def _select_windows_for_drone(
    problem: ProblemData,
    route: Sequence[int],
    delivery_groups: Sequence[Sequence[int]],
) -> Optional[List[List[object]]]:
    """Choose launch/recovery windows for one physical drone efficiently.

    Every chronologically valid pair is first scored with the chromosome's own
    delivery order (linear-time flight evaluation).  Only the best few pairs are
    re-evaluated with ``energy_optimal_sortie``.  This keeps the paper-style
    decoder broad without running an exact ordering DP for every truck-stop pair.
    """
    if not delivery_groups:
        return []
    if len(route) < 2 or len(delivery_groups) > len(route) - 1:
        return None

    prefix = _truck_prefix_times(problem, route)
    chosen: List[List[object]] = []
    previous_recovery = 0
    used_energy = 0.0
    total_groups = len(delivery_groups)
    TOP_K = 4

    for g_idx, deliveries0 in enumerate(delivery_groups):
        deliveries = [int(c) for c in deliveries0]
        if not deliveries:
            continue
        fixed_order_flight = prepare_fixed_order_flight(problem, deliveries)
        payload = sum(float(problem.customer_demands[c]) for c in deliveries)
        if payload > problem.drone_payload_capacity + EPS:
            return None

        remaining = total_groups - g_idx - 1
        latest_recovery = (len(route) - 1) - remaining
        approx_candidates = []
        for li in range(previous_recovery, len(route) - 1):
            if li >= latest_recovery:
                break
            for ri in range(li + 1, latest_recovery + 1):
                launch, recovery = int(route[li]), int(route[ri])
                # Cheap exact physics for the current chromosome order; no
                # combinatorial delivery-order DP in this filtering pass.
                flight_t, flight_e = fixed_order_flight(launch, recovery)
                truck_segment_t = float(prefix[ri] - prefix[li])
                wait_t = max(0.0, flight_t - truck_segment_t)
                hover_t = 0.0 if recovery == 0 else max(0.0, truck_segment_t - flight_t)
                hover_e = float(hover_energy_wh(hover_t, 0.0))
                total_e = flight_e + hover_e
                span = ri - li
                approx_candidates.append(((wait_t, total_e, flight_e, -span, li, ri), li, ri))

        if not approx_candidates:
            return None
        approx_candidates.sort(key=lambda x: x[0])

        # Usually the proxy winner is also the exact winner.  Add the widest
        # feasible-looking window (often depot->depot) as a dedicated parallelism
        # candidate, but keep exact DP calls to a small constant per sortie.
        selected = list(approx_candidates[:TOP_K])
        widest = min(approx_candidates, key=lambda x: (-(x[2] - x[1]), x[0]))
        if all((x[1], x[2]) != (widest[1], widest[2]) for x in selected):
            selected.append(widest)

        best = None
        for _proxy_key, li, ri in selected:
            launch, recovery = int(route[li]), int(route[ri])
            flight = energy_optimal_sortie(
                launch, recovery, deliveries,
                problem.dro_dist_matrix, problem.customer_demands,
            )
            flight_t = float(flight['flight_time_s'])
            flight_e = float(flight['flight_energy_wh'])
            truck_segment_t = float(prefix[ri] - prefix[li])
            wait_t = max(0.0, flight_t - truck_segment_t)
            hover_t = 0.0 if recovery == 0 else max(0.0, truck_segment_t - flight_t)
            hover_e = float(hover_energy_wh(hover_t, 0.0))
            total_e = flight_e + hover_e
            if used_energy + total_e > problem.drone_battery_capacity_wh + EPS:
                continue
            span = ri - li
            key = (wait_t, total_e, flight_e, -span, li, ri)
            if best is None or key < best[0]:
                best = (key, li, ri, flight, total_e)

        if best is None:
            return None
        _key, li, ri, flight, total_e = best
        chosen.append([
            None,
            int(route[li]),
            [int(x) for x in flight['order']],
            int(route[ri]),
        ])
        previous_recovery = int(ri)
        used_energy += float(total_e)
    return chosen


def _make_route_and_sorties(
    problem: ProblemData,
    truck_idx: int,
    master: Sequence[int],
    service: List[int],
    drone_gene: Sequence[int],
) -> Tuple[List[int], List[List[object]], bool]:
    """V3 type-aware decoder with independent launch/recovery window selection.

    Unlike V2, drone sorties are not forced to live between two consecutive
    truck-service anchors in ``master``.  This allows a sortie such as depot ->
    several drone customers -> depot while the truck simultaneously traverses
    its entire route, which is essential for matching the solution space used by
    the ALNS evaluator/model.
    """
    D = problem.num_drones_per_vehicle
    changed = False
    if D <= 0:
        for c in master:
            service[int(c)] = 0
        return [0] + [int(c) for c in master] + [0], [], False

    # Heavy customers can never be drone served.
    for c0 in master:
        c = int(c0)
        if service[c] == 1 and problem.customer_demands[c] > problem.drone_payload_capacity + EPS:
            service[c] = 0
            changed = True

    # Retry after local phenotype repairs.  A repair only demotes one offending
    # drone customer to the truck, thereby adding a synchronization anchor and
    # preserving all other genes.
    for _ in range(max(3, len(master) + 3)):
        truck_customers = [int(c) for c in master if int(service[int(c)]) == 0]
        route = [0] + truck_customers + [0]

        by_drone: List[List[int]] = [[] for _ in range(D)]
        for c0 in master:
            c = int(c0)
            if int(service[c]) == 1:
                by_drone[int(drone_gene[c]) % D].append(c)

        sorties: List[List[object]] = []
        failure_group: Optional[List[int]] = None
        for di, seq in enumerate(by_drone):
            if not seq:
                continue
            packed = _pack_drone_customers(problem, seq)
            # A returned singleton can also mean an individually impossible heavy
            # customer, handled below by demotion.
            if any(sum(problem.customer_demands[c] for c in g) > problem.drone_payload_capacity + EPS for g in packed):
                failure_group = max(packed, key=lambda g: sum(problem.customer_demands[c] for c in g))
                break
            windows = _select_windows_for_drone(problem, route, packed)
            if windows is None:
                # Prefer repairing the last/largest packed sortie; it is usually
                # the one that causes battery or temporal-window pressure.
                failure_group = max(packed, key=lambda g: (sum(problem.customer_demands[c] for c in g), len(g)))
                break
            label = f"{truck_idx + 1}-D{di + 1}"
            for s in windows:
                s[0] = label
                sorties.append(s)

        if failure_group is not None and failure_group:
            # Return the locally least useful drone customer to the truck.  With
            # window decoding there is no fixed launch/recovery yet, so use its
            # nearest current truck anchors in master order as the repair context.
            c = max(
                failure_group,
                key=lambda x: (problem.customer_demands[int(x)], int(x)),
            )
            service[int(c)] = 0
            changed = True
            continue

        # Sort by launch/recovery position and label.  Same-drone windows were
        # already constructed in chronological order.
        pos = {int(node): idx for idx, node in enumerate(route) if int(node) != 0}
        def skey(s):
            launch, recovery = int(s[1]), int(s[3])
            li = 0 if launch == 0 else pos[launch]
            ri = len(route) - 1 if recovery == 0 else pos[recovery]
            return (li, ri, str(s[0]))
        sorties.sort(key=skey)
        return route, sorties, changed

    # Guaranteed all-truck fallback.
    for c in master:
        service[int(c)] = 0
    return [0] + [int(c) for c in master] + [0], [], True

def _build_route_solution_from_eval(
    problem: ProblemData,
    truck_idx: int,
    master: Sequence[int],
    route: Sequence[int],
    evaluation: SolutionEvaluation,
) -> RouteSolution:
    out: List[Sortie] = []
    for rep in evaluation.sorties:
        try:
            di = max(0, int(str(rep.drone_label).rsplit("D", 1)[1]) - 1)
        except Exception:
            di = 0
        out.append(Sortie(
            truck=truck_idx,
            drone=di,
            drone_label=str(rep.drone_label),
            launch=int(rep.launch),
            deliveries=[int(c) for c in rep.order],
            recovery=int(rep.rendezvous),
            launch_index=int(rep.launch_idx),
            recovery_index=int(rep.rendezvous_idx),
            flight_time_s=float(rep.flight_time_s),
            flight_energy_wh=float(rep.flight_energy_wh),
            hover_time_s=float(rep.hover_time_s),
            hover_energy_wh=float(rep.hover_energy_wh),
        ))
    energy = [0.0] * problem.num_drones_per_vehicle
    for (_v, label), state in evaluation.drones.items():
        try:
            di = max(0, int(str(label).rsplit("D", 1)[1]) - 1)
        except Exception:
            di = 0
        if 0 <= di < len(energy):
            energy[di] = float(state.energy_wh)
    return RouteSolution(
        truck=truck_idx,
        master_sequence=[int(x) for x in master],
        route=[int(x) for x in route],
        sorties=out,
        completion_time_s=float(evaluation.vehicle_completion_times[0]),
        drone_energy_wh=energy,
    )


def _decode_one_vehicle(
    problem: ProblemData,
    truck_idx: int,
    master: Sequence[int],
    service_gene: Sequence[int],
    drone_gene: Sequence[int],
) -> RouteSolution:
    local_service = list(service_gene)
    local = _OneVehicleView(problem)
    max_repairs = max(2, len(master) + 2)

    for _ in range(max_repairs):
        route, sorties, _ = _make_route_and_sorties(
            problem, truck_idx, master, local_service, drone_gene
        )
        ev = evaluate_solution([route], [sorties], local, check_coverage=False)
        if ev.feasible:
            return _build_route_solution_from_eval(problem, truck_idx, master, route, ev)

        # The only normal failure left here is cumulative battery after hover.
        # Demote one customer from the approximate heaviest physical drone.
        if sorties:
            energy_by_label: Dict[str, float] = {}
            details = []
            for s in sorties:
                label, launch, deliveries, recovery = s
                f = energy_optimal_sortie(
                    int(launch), int(recovery), deliveries,
                    problem.dro_dist_matrix, problem.customer_demands,
                )
                e = float(f["flight_energy_wh"])
                energy_by_label[str(label)] = energy_by_label.get(str(label), 0.0) + e
                details.append((str(label), int(launch), list(deliveries), int(recovery), e))
            worst_label = max(energy_by_label, key=energy_by_label.get)
            choices = [x for x in details if x[0] == worst_label]
            if choices:
                _, launch, deliveries, recovery, _e = max(choices, key=lambda x: x[4])
                c = _least_attractive_for_drone(problem, deliveries, launch, recovery)
                local_service[c] = 0
                continue
        break

    # Guaranteed fallback: all assigned customers are truck-served.
    route = [0] + [int(c) for c in master] + [0]
    ev = evaluate_solution([route], [[]], local, check_coverage=False)
    if not ev.feasible:
        raise RuntimeError("all-truck fallback infeasible: " + ev.reason)
    return _build_route_solution_from_eval(problem, truck_idx, master, route, ev)


def decode(problem: ProblemData, chrom: Chromosome) -> DecodedSolution:
    groups = split_customer_order_minmax(problem, chrom.route_string, problem.num_vehicles)
    routes = [
        _decode_one_vehicle(problem, v, groups[v], chrom.service, chrom.drone)
        for v in range(problem.num_vehicles)
    ]
    vehicle_routes = [r.route for r in routes]
    drone_routes = [
        [[s.drone_label, s.launch, list(s.deliveries), s.recovery] for s in r.sorties]
        for r in routes
    ]
    exact = evaluate_solution(vehicle_routes, drone_routes, problem, check_coverage=True)

    if exact.feasible:
        # Replace per-route reports with the exact global evaluator reports.  This
        # guarantees stored hover, SOC and completion times equal fitness data.
        by_v: Dict[int, List[object]] = {}
        for rep in exact.sorties:
            by_v.setdefault(int(rep.vehicle_idx), []).append(rep)
        for v, r in enumerate(routes):
            converted = []
            for rep in by_v.get(v, []):
                try:
                    di = max(0, int(str(rep.drone_label).rsplit("D", 1)[1]) - 1)
                except Exception:
                    di = 0
                converted.append(Sortie(
                    truck=v, drone=di, drone_label=str(rep.drone_label),
                    launch=int(rep.launch), deliveries=[int(c) for c in rep.order],
                    recovery=int(rep.rendezvous), launch_index=int(rep.launch_idx),
                    recovery_index=int(rep.rendezvous_idx),
                    flight_time_s=float(rep.flight_time_s),
                    flight_energy_wh=float(rep.flight_energy_wh),
                    hover_time_s=float(rep.hover_time_s),
                    hover_energy_wh=float(rep.hover_energy_wh),
                ))
            r.sorties = converted
            r.completion_time_s = float(exact.vehicle_completion_times[v])
            r.drone_energy_wh = [0.0] * problem.num_drones_per_vehicle
            for (vv, label), state in exact.drones.items():
                if int(vv) != v:
                    continue
                try:
                    di = max(0, int(str(label).rsplit("D", 1)[1]) - 1)
                except Exception:
                    di = 0
                if 0 <= di < len(r.drone_energy_wh):
                    r.drone_energy_wh[di] = float(state.energy_wh)

    return DecodedSolution(
        routes=routes,
        feasible=bool(exact.feasible),
        reason=str(exact.reason),
        makespan_s=float(exact.objective),
        exact_evaluation=exact,
    )


def validate_solution(problem: ProblemData, solution: DecodedSolution) -> Tuple[bool, str]:
    ev = evaluate_solution(solution.vehicle_routes, solution.drone_routes, problem, True)
    if not ev.feasible:
        return False, ev.reason
    if abs(float(ev.makespan) - float(solution.makespan_s)) > 1e-8:
        return False, "stored makespan differs from exact evaluator"
    return True, "ok"


# ---------------------------------------------------------------------------
# Seed construction (independent from ALNS)
# ---------------------------------------------------------------------------
def _route_distance(problem: ProblemData, route: Sequence[int]) -> float:
    return sum(float(problem.veh_dist_matrix[route[i], route[i + 1]]) for i in range(len(route) - 1))


def _nearest_neighbor(problem: ProblemData, nodes: Sequence[int], rng: Optional[random.Random] = None, top_k: int = 1) -> List[int]:
    remaining = set(int(x) for x in nodes)
    route = [0]
    cur = 0
    while remaining:
        ranked = sorted(remaining, key=lambda c: (problem.veh_dist_matrix[cur, c], c))
        k = min(max(1, int(top_k)), len(ranked))
        nxt = ranked[0] if rng is None or k == 1 else rng.choice(ranked[:k])
        route.append(int(nxt)); remaining.remove(nxt); cur = int(nxt)
    route.append(0)
    return route


def _two_opt_seed(problem: ProblemData, route: List[int]) -> List[int]:
    """Deterministic 2-opt used only when constructing the initial population."""
    r = list(route)
    if len(r) <= 4:
        return r
    improved = True
    while improved:
        improved = False
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                a, b, c, d = r[i - 1], r[i], r[j], r[j + 1]
                delta = (
                    problem.veh_dist_matrix[a, c] + problem.veh_dist_matrix[b, d]
                    - problem.veh_dist_matrix[a, b] - problem.veh_dist_matrix[c, d]
                )
                if delta < -EPS:
                    r[i:j + 1] = reversed(r[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return r


def _balanced_farthest_partition(problem: ProblemData, rng: random.Random) -> List[List[int]]:
    """Independent spatial seed: farthest-point centers + balanced nearest assignment.

    This is not the project's sweep/KMeans initializer.  It is used only to seed
    a subset of the GA population; assignment remains fully evolvable afterward.
    """
    n, k = problem.num_customers, problem.num_vehicles
    coords = np.asarray(problem.customer_coords, dtype=float)
    depot = np.asarray(problem.depot_coords, dtype=float)
    if k == 1:
        return [list(range(1, n + 1))]

    depot_dist = np.sqrt(np.square(coords - depot).sum(axis=1))
    ranked_first = np.argsort(-depot_dist)[:min(6, n)].tolist()
    first = int(rng.choice(ranked_first))
    centers = [first]
    pair = np.sqrt(np.square(coords[:, None, :] - coords[None, :, :]).sum(axis=2))
    while len(centers) < k:
        min_to_center = np.min(pair[:, centers], axis=1)
        for c in centers:
            min_to_center[c] = -1.0
        ranked = np.argsort(-min_to_center)[:min(6, n - len(centers))].tolist()
        centers.append(int(rng.choice(ranked)))

    # Hard count caps keep a makespan-oriented seed from collapsing into one
    # large route and several tiny routes.
    base, extra = divmod(n, k)
    caps = [base + (1 if v < extra else 0) for v in range(k)]
    groups = [[] for _ in range(k)]

    # Assign customers with the strongest center preference first.
    records = []
    for i in range(n):
        ds = sorted((float(pair[i, c]), v) for v, c in enumerate(centers))
        margin = ds[1][0] - ds[0][0] if len(ds) > 1 else 0.0
        records.append((-margin, i, ds))
    records.sort()
    for _neg_margin, i, ds in records:
        placed = False
        for _dist, v in ds:
            if len(groups[v]) < caps[v]:
                groups[v].append(i + 1)
                placed = True
                break
        if not placed:
            v = min(range(k), key=lambda z: len(groups[z]))
            groups[v].append(i + 1)
    return groups


def _global_nn_split(problem: ProblemData, rng: random.Random, top_k: int = 1) -> List[List[int]]:
    """Build one giant nearest-neighbor tour, then split it into balanced routes."""
    giant = _nearest_neighbor(problem, range(1, problem.num_customers + 1), rng, top_k)[1:-1]
    return [list(map(int, x)) for x in np.array_split(giant, problem.num_vehicles)]

def _join_routes(routes: Sequence[Sequence[int]]) -> List[int]:
    out: List[int] = []
    for v, route in enumerate(routes):
        out.extend(int(c) for c in route)
        if v < len(routes) - 1:
            out.append(-(v + 1))
    return out


def _seed_service_genes(problem: ProblemData, routes: Sequence[Sequence[int]], rng: random.Random, strength: float) -> Tuple[List[int], List[int]]:
    n = problem.num_customers
    service = [0] * (n + 1)
    drone = [0] * (n + 1)
    D = max(1, problem.num_drones_per_vehicle)
    if problem.num_drones_per_vehicle <= 0 or strength <= 0:
        return service, drone

    candidates = []
    for vr in routes:
        closed = [0] + list(vr) + [0]
        for i in range(1, len(closed) - 1):
            prev, c, nxt = closed[i - 1], closed[i], closed[i + 1]
            if problem.customer_demands[c] > problem.drone_payload_capacity + EPS:
                continue
            truck_old = (problem.veh_dist_matrix[prev, c] + problem.veh_dist_matrix[c, nxt]) / problem.vehicle_speed
            truck_direct = problem.veh_dist_matrix[prev, nxt] / problem.vehicle_speed
            drone_t = (problem.dro_dist_matrix[prev, c] + problem.dro_dist_matrix[c, nxt]) / problem.drone_speed
            saving = float(truck_old - max(truck_direct, drone_t))
            if saving > 0:
                candidates.append((saving, c))
    candidates.sort(reverse=True)
    limit = max(1, int(round(strength * n)))
    chosen = set()
    for _saving, c in candidates:
        if len(chosen) >= limit:
            break
        # Mild stochasticity creates distinct but sensible seed individuals.
        if rng.random() <= 0.85:
            service[c] = 1
            drone[c] = len(chosen) % D
            chosen.add(c)
    # Initialize unspecified drone labels too, so future service flips are ready.
    for c in range(1, n + 1):
        if c not in chosen:
            drone[c] = rng.randrange(D)
    return service, drone


# ---------------------------------------------------------------------------
# Classic GA
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    population_size: int = 120
    generations: int = 500
    elite_size: int = 6
    tournament_size: int = 4
    crossover_rate: float = 0.92
    route_exchange_rate: float = 0.40
    route_mutation_rate: float = 0.62
    service_flip_rate: float = 0.018
    service_block_mutation_rate: float = 0.18
    drone_mutation_rate: float = 0.030
    bottleneck_transfer_rate: float = 0.24
    bottleneck_offload_rate: float = 0.38
    random_immigrant_rate: float = 0.03
    heuristic_seed_fraction: float = 0.72
    stagnation_window: int = 18
    seed: int = 13
    # Wall-clock budget in seconds for the whole ``solve()`` call (population
    # construction + all generations).  0.0 keeps the old behaviour: the run
    # stops on ``generations`` alone.  Added so the GA can be compared with
    # ALNS/VND/VNS under an identical time budget.
    time_limit_seconds: float = 0.0


class ClassicGA:
    def __init__(self, problem: ProblemData, config: Optional[GAConfig] = None):
        self.problem = problem
        self.cfg = config or GAConfig()
        if self.cfg.population_size < 6:
            raise ValueError("population_size must be >= 6")
        if not (1 <= self.cfg.elite_size < self.cfg.population_size):
            raise ValueError("invalid elite_size")
        if self.cfg.generations <= 0 and self.cfg.time_limit_seconds <= 0:
            raise ValueError("generations<=0 requires a positive time_limit_seconds")
        if problem.num_vehicles > problem.num_customers:
            raise ValueError("num_vehicles cannot exceed num_customers")
        self.rng = random.Random(self.cfg.seed)
        self.history: List[Dict[str, float]] = []
        # Optional convergence hook; None preserves the original behaviour.
        self.history_callback = None
        # Counts exact decode/evaluator calls (cache hits are not re-evaluated).
        self.evaluation_count = 0
        self.eval_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], Tuple[float, DecodedSolution]] = {}

    @property
    def token_set(self):
        return set(range(1, self.problem.num_customers + 1)) | set(separators(self.problem.num_vehicles))

    def _repair_route_string(self, seq: Sequence[int]) -> List[int]:
        """Restore a valid permutation and avoid empty truck groups."""
        expected = self.token_set
        seen = set()
        out = []
        for x0 in seq:
            x = int(x0)
            if x in expected and x not in seen:
                out.append(x); seen.add(x)
        missing = list(expected - seen)
        self.rng.shuffle(missing)
        out.extend(missing)

        # Move separators away from ends/adjacency so every truck owns >=1
        # customer.  This repair is deterministic enough not to destroy OX.
        k = self.problem.num_vehicles
        if k <= 1:
            return [x for x in out if x > 0]
        # Extract customers and current separator order; rebuild only if empty.
        customers = [x for x in out if x > 0]
        groups = split_customer_order_minmax(self.problem, customers, k)
        return _join_routes(groups)

    def _random_route_string(self) -> List[int]:
        customers = list(range(1, self.problem.num_customers + 1))
        self.rng.shuffle(customers)
        # Random non-empty split.
        cuts = sorted(self.rng.sample(range(1, len(customers)), self.problem.num_vehicles - 1)) if self.problem.num_vehicles > 1 else []
        groups = []
        pts = [0] + cuts + [len(customers)]
        for i in range(self.problem.num_vehicles):
            groups.append(customers[pts[i]:pts[i + 1]])
        return self._repair_route_string(_join_routes(groups))

    def _heuristic_route_candidates(self) -> List[List[List[int]]]:
        """Create independent, generic routing seeds (no ALNS/sweep initializer)."""
        scored = []
        trials = max(16, 8 * self.problem.num_vehicles)
        local_rng = random.Random(self.cfg.seed + 7919)
        for t in range(trials):
            if t % 2 == 0:
                clusters = _balanced_farthest_partition(self.problem, local_rng)
            else:
                clusters = _global_nn_split(
                    self.problem, local_rng, top_k=1 + (t % 3)
                )
            routes = []
            for cl in clusters:
                # Randomized NN only for seed diversity; deterministic 2-opt then
                # removes obvious crossings before GA starts.
                rr = _nearest_neighbor(
                    self.problem, cl, local_rng, top_k=1 + (t % 3)
                )
                rr = _two_opt_seed(self.problem, rr)
                routes.append(rr[1:-1])
            makespan = max(
                _route_distance(self.problem, [0] + r + [0]) / self.problem.vehicle_speed
                for r in routes
            )
            scored.append((makespan, routes))
        scored.sort(key=lambda x: x[0])
        unique, keys = [], set()
        for _m, routes in scored:
            key = tuple(tuple(r) for r in routes)
            if key not in keys:
                keys.add(key); unique.append(routes)
        return unique

    def initial_population(self) -> List[Chromosome]:
        n = self.problem.num_customers
        heuristic_routes = self._heuristic_route_candidates()
        pop: List[Chromosome] = []

        # 1) Strong all-truck baseline.  This guarantees the GA starts no worse
        # than a sensible routing construction.
        best_routes = heuristic_routes[0]
        service = [0] * (n + 1)
        drone = [0] * (n + 1)
        pop.append(Chromosome(self._repair_route_string(_join_routes(best_routes)), service, drone))

        target_heur = max(1, int(round(self.cfg.heuristic_seed_fraction * self.cfg.population_size)))
        idx = 0
        while len(pop) < min(target_heur, self.cfg.population_size):
            base_routes = [list(r) for r in heuristic_routes[idx % len(heuristic_routes)]]
            idx += 1
            # Randomized route perturbation for diversity, while staying local.
            for r in base_routes:
                if len(r) >= 4 and self.rng.random() < 0.7:
                    i, j = sorted(self.rng.sample(range(len(r)), 2))
                    if self.rng.random() < 0.5:
                        r[i], r[j] = r[j], r[i]
                    else:
                        r[i:j + 1] = reversed(r[i:j + 1])
            strength = self.rng.uniform(0.18, 0.62)
            s, d = _seed_service_genes(self.problem, base_routes, self.rng, strength)
            pop.append(Chromosome(self._repair_route_string(_join_routes(base_routes)), s, d))

        # 2) Random immigrants preserve global diversity.
        while len(pop) < self.cfg.population_size:
            s = [0] * (n + 1)
            d = [0] * (n + 1)
            for c in range(1, n + 1):
                s[c] = int(self.rng.random() < self.rng.uniform(0.15, 0.60))
                d[c] = self.rng.randrange(max(1, self.problem.num_drones_per_vehicle))
            pop.append(Chromosome(self._random_route_string(), s, d))
        return pop

    def _emit_history(self, event_type: str, generation: int) -> None:
        callback = getattr(self, "history_callback", None)
        if callback is None:
            return
        value = float(self._last_emitted_best)
        if not math.isfinite(value):
            return
        callback({
            "best_makespan_s": value,
            "iteration_or_generation": int(generation),
            "evaluations": int(self.evaluation_count),
            "event_type": str(event_type),
        })

    def evaluate(self, c: Chromosome) -> float:
        if c.solution is None:
            sig = self._signature(c)
            cached = self.eval_cache.get(sig)
            if cached is not None:
                c.fitness, c.solution = cached
                return c.fitness
            self.evaluation_count += 1
            c.solution = decode(self.problem, c)
            c.fitness = c.solution.makespan_s if c.solution.feasible else math.inf
            if len(self.eval_cache) > 60000:
                self.eval_cache.clear()
            self.eval_cache[sig] = (c.fitness, c.solution)
        return c.fitness

    def _tournament(self, pop: Sequence[Chromosome]) -> Chromosome:
        pool = self.rng.sample(list(pop), k=min(self.cfg.tournament_size, len(pop)))
        return min(pool, key=self.evaluate)

    def _ox_with_origin(
        self, a: Sequence[int], b: Sequence[int]
    ) -> Tuple[List[int], Dict[int, int]]:
        """Order crossover plus provenance for type-aware gene inheritance.

        ``origin[c]`` is 0 when customer ``c`` came from parent ``a``'s copied
        segment and 1 when it came from parent ``b``'s fill order.  This lets the
        service/drone type travel with the route building block instead of being
        independently randomized after crossover.
        """
        n = len(a)
        if n < 2:
            return list(a), {int(x): 0 for x in a if int(x) > 0}
        i, j = sorted(self.rng.sample(range(n), 2))
        j += 1
        child: List[Optional[int]] = [None] * n
        child[i:j] = [int(x) for x in a[i:j]]
        used = set(int(x) for x in a[i:j])
        origin: Dict[int, int] = {
            int(x): 0 for x in a[i:j] if int(x) > 0
        }
        fill = [
            int(b[(j + z) % n]) for z in range(n)
            if int(b[(j + z) % n]) not in used
        ]
        at = j % n
        for x in fill:
            while child[at] is not None:
                at = (at + 1) % n
            child[at] = x
            if x > 0:
                origin[x] = 1
            at = (at + 1) % n
        return [int(x) for x in child], origin

    def _route_estimated_time(self, group: Sequence[int], service: Sequence[int]) -> float:
        route = [0] + [int(c) for c in group if int(service[int(c)]) == 0] + [0]
        return _route_distance(self.problem, route) / self.problem.vehicle_speed

    def _route_exchange_child(
        self, primary: Chromosome, secondary: Chromosome
    ) -> Chromosome:
        """Preserve one complete truck segment, then complete from the other parent.

        This follows the route/segment-preservation idea used in multi-salesman
        and multi-truck evolutionary routing: useful route building blocks are
        inherited intact instead of being cut by OX separators.
        """
        k = self.problem.num_vehicles
        ga = split_customer_order_minmax(self.problem, primary.route_string, k)
        gb = split_customer_order_minmax(self.problem, secondary.route_string, k)
        fixed = self.rng.randrange(k)
        child_groups: List[List[int]] = [[] for _ in range(k)]
        child_groups[fixed] = list(ga[fixed])
        used = set(child_groups[fixed])

        # Preserve all surviving secondary route subsequences.
        for v in range(k):
            if v == fixed:
                continue
            child_groups[v] = [int(c) for c in gb[v] if int(c) not in used]
            used.update(child_groups[v])

        # Customers displaced from secondary's route at ``fixed`` are inserted
        # into the currently light routes by minimum truck insertion cost.
        leftovers = [c for c in range(1, self.problem.num_customers + 1) if c not in used]
        for c in leftovers:
            candidates = [v for v in range(k) if v != fixed] or [fixed]
            best = None
            for v in candidates:
                g = child_groups[v]
                if not g:
                    key = (0.0, 0, v)
                    pos = 0
                else:
                    best_delta = math.inf
                    pos = 0
                    closed = [0] + g + [0]
                    for p in range(len(g) + 1):
                        u, w = closed[p], closed[p + 1]
                        delta = (
                            float(self.problem.veh_dist_matrix[u, c])
                            + float(self.problem.veh_dist_matrix[c, w])
                            - float(self.problem.veh_dist_matrix[u, w])
                        )
                        if delta < best_delta:
                            best_delta = delta
                            pos = p
                    key = (
                        self._route_estimated_time(g, secondary.service)
                        + best_delta / self.problem.vehicle_speed,
                        len(g), v,
                    )
                if best is None or key < best[0]:
                    best = (key, v, pos)
            _, v, pos = best
            child_groups[v].insert(pos, int(c))
            used.add(int(c))

        # Avoid empty routes without destroying the preserved route unless needed.
        empties = [v for v, g in enumerate(child_groups) if not g]
        for v in empties:
            donors = sorted(
                [u for u, g in enumerate(child_groups) if len(g) > 1],
                key=lambda u: len(child_groups[u]), reverse=True,
            )
            if not donors:
                break
            donor = donors[0]
            moved = child_groups[donor].pop()
            child_groups[v].append(moved)

        n = self.problem.num_customers
        service = [0] * (n + 1)
        drone = [0] * (n + 1)
        fixed_set = set(child_groups[fixed])
        for c in range(1, n + 1):
            src = primary if c in fixed_set else secondary
            service[c] = int(src.service[c])
            drone[c] = int(src.drone[c])
        return Chromosome(self._repair_route_string(_join_routes(child_groups)), service, drone)

    def crossover(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() >= self.cfg.crossover_rate:
            c1, c2 = a.clone(), b.clone()
            _invalidate(c1); _invalidate(c2)
            return c1, c2

        if self.rng.random() < self.cfg.route_exchange_rate:
            return self._route_exchange_child(a, b), self._route_exchange_child(b, a)

        r1, origin1 = self._ox_with_origin(a.route_string, b.route_string)
        r2, origin2 = self._ox_with_origin(b.route_string, a.route_string)
        r1 = self._repair_route_string(r1)
        r2 = self._repair_route_string(r2)
        n = self.problem.num_customers
        s1, s2 = [0] * (n + 1), [0] * (n + 1)
        d1, d2 = [0] * (n + 1), [0] * (n + 1)
        for c in range(1, n + 1):
            src1 = a if origin1.get(c, 1) == 0 else b
            src2 = b if origin2.get(c, 1) == 0 else a
            s1[c], d1[c] = int(src1.service[c]), int(src1.drone[c])
            s2[c], d2[c] = int(src2.service[c]), int(src2.drone[c])
        return Chromosome(r1, s1, d1), Chromosome(r2, s2, d2)

    def _customer_positions(self, seq: Sequence[int]) -> List[int]:
        return [i for i, x in enumerate(seq) if int(x) > 0]

    def _bottleneck_transfer(self, c: Chromosome) -> bool:
        """One makespan-aware cross-truck relocation mutation.

        Only one customer is moved.  Candidate customers are sampled, so this is
        a mutation operator rather than an iterative local search.
        """
        k = self.problem.num_vehicles
        if k <= 1:
            return False
        groups = split_customer_order_minmax(self.problem, c.route_string, k)
        times = [self._route_estimated_time(g, c.service) for g in groups]
        src = max(range(k), key=lambda v: times[v])
        targets = sorted(range(k), key=lambda v: times[v])
        targets = [v for v in targets if v != src]
        if not targets or len(groups[src]) <= 1:
            return False

        truck_candidates = [x for x in groups[src] if c.service[int(x)] == 0]
        pool = truck_candidates or list(groups[src])
        sample = self.rng.sample(pool, k=min(len(pool), 7))
        best = None
        for customer in sample:
            src_new = [x for x in groups[src] if x != customer]
            t_src = self._route_estimated_time(src_new, c.service)
            for dst in targets[:min(2, len(targets))]:
                dst_base = list(groups[dst])
                for pos in range(len(dst_base) + 1):
                    dst_new = dst_base[:pos] + [customer] + dst_base[pos:]
                    t_dst = self._route_estimated_time(dst_new, c.service)
                    new_bound = max(
                        [t_src if v == src else t_dst if v == dst else times[v] for v in range(k)]
                    )
                    key = (new_bound, t_src + t_dst, dst, pos)
                    if best is None or key < best[0]:
                        best = (key, customer, dst, pos)
        if best is None:
            return False
        key, customer, dst, pos = best
        # Accept guided moves that improve the crude truck-only max estimate;
        # occasionally accept a neutral/worse one to retain exploration.
        if key[0] > max(times) + EPS and self.rng.random() > 0.12:
            return False
        groups[src].remove(customer)
        groups[dst].insert(pos, customer)
        c.route_string = self._repair_route_string(_join_routes(groups))
        return True

    def _bottleneck_offload(self, c: Chromosome) -> bool:
        """Flip one high-value truck customer on the current bottleneck route to drone.

        This is a problem-tailored mutation, not a local-search loop.  It mirrors
        the GA literature's use of domain-specific mutations while keeping the
        exact evaluator as the sole fitness definition.
        """
        if self.problem.num_drones_per_vehicle <= 0:
            return False
        groups = split_customer_order_minmax(
            self.problem, c.route_string, self.problem.num_vehicles
        )
        times = [self._route_estimated_time(g, c.service) for g in groups]
        src = max(range(len(groups)), key=lambda v: times[v])
        g = groups[src]
        truck_nodes = [int(x) for x in g if int(c.service[int(x)]) == 0]
        eligible = [
            x for x in truck_nodes
            if self.problem.customer_demands[x] <= self.problem.drone_payload_capacity + EPS
        ]
        if not eligible:
            return False
        route = [0] + truck_nodes + [0]
        pos = {node: i for i, node in enumerate(route) if node != 0}
        scored = []
        for customer in eligible:
            i = pos[customer]
            prev, nxt = route[i - 1], route[i + 1]
            saving = _truck_detour_saving(self.problem, prev, customer, nxt)
            # Favor customers whose drone round trip is relatively cheap too.
            drone_direct = (
                float(self.problem.dro_dist_matrix[prev, customer])
                + float(self.problem.dro_dist_matrix[customer, nxt])
            ) / self.problem.drone_speed
            score = saving - 0.25 * drone_direct
            scored.append((score, saving, customer))
        scored.sort(reverse=True)
        top = scored[:min(5, len(scored))]
        if not top or top[0][1] <= EPS:
            return False
        # Mostly exploit the best offload, sometimes take another top candidate.
        chosen = top[0] if self.rng.random() < 0.72 else self.rng.choice(top)
        customer = int(chosen[2])
        c.service[customer] = 1
        D = self.problem.num_drones_per_vehicle
        loads = [0.0] * D
        for x in g:
            x = int(x)
            if x != customer and int(c.service[x]) == 1:
                loads[int(c.drone[x]) % D] += float(self.problem.customer_demands[x])
        c.drone[customer] = min(range(D), key=lambda z: loads[z])
        return True

    def _service_block_mutation(self, c: Chromosome, intensity: float) -> None:
        if self.problem.num_drones_per_vehicle <= 0:
            return
        groups = split_customer_order_minmax(self.problem, c.route_string, self.problem.num_vehicles)
        nonempty = [g for g in groups if g]
        if not nonempty:
            return
        g = self.rng.choice(nonempty)
        max_len = min(len(g), max(2, int(round(3 * intensity))))
        length = self.rng.randint(1, max_len)
        start = self.rng.randrange(0, len(g) - length + 1)
        block = g[start:start + length]
        make_drone = self.rng.random() < 0.74
        if not make_drone:
            for customer in block:
                c.service[int(customer)] = 0
            return

        D = self.problem.num_drones_per_vehicle
        loads = [0.0] * D
        # Put nearby chromosome customers onto the same physical drone until its
        # sortie payload would overflow, then use the lightest drone.  The decoder
        # may later split again if cumulative battery requires it.
        for customer in block:
            customer = int(customer)
            q = float(self.problem.customer_demands[customer])
            if q > self.problem.drone_payload_capacity + EPS:
                c.service[customer] = 0
                continue
            choices = [d for d in range(D) if loads[d] + q <= self.problem.drone_payload_capacity + EPS]
            if not choices:
                loads = [0.0] * D
                choices = list(range(D))
            d = min(choices, key=lambda z: loads[z])
            c.service[customer] = 1
            c.drone[customer] = d
            loads[d] += q

    def mutate(self, c: Chromosome, intensity: float = 1.0) -> None:
        intensity = max(0.55, min(2.6, float(intensity)))

        if self.rng.random() < min(0.98, self.cfg.bottleneck_transfer_rate * intensity):
            self._bottleneck_transfer(c)
        if self.rng.random() < min(0.98, self.cfg.bottleneck_offload_rate * intensity):
            self._bottleneck_offload(c)

        if self.rng.random() < min(0.98, self.cfg.route_mutation_rate * intensity):
            move = self.rng.choice(("relocate", "swap", "invert", "relocate"))
            pos = self._customer_positions(c.route_string)
            if len(pos) >= 2:
                if move == "relocate":
                    i = self.rng.choice(pos)
                    token = c.route_string.pop(i)
                    j = self.rng.randrange(len(c.route_string) + 1)
                    c.route_string.insert(j, token)
                elif move == "swap":
                    i, j = self.rng.sample(pos, 2)
                    c.route_string[i], c.route_string[j] = c.route_string[j], c.route_string[i]
                else:
                    i, j = sorted(self.rng.sample(pos, 2))
                    inds = [z for z in range(i, j + 1) if c.route_string[z] > 0]
                    vals = [c.route_string[z] for z in inds][::-1]
                    for z, val in zip(inds, vals):
                        c.route_string[z] = val
            c.route_string = self._repair_route_string(c.route_string)

        if self.rng.random() < min(0.85, self.cfg.service_block_mutation_rate * intensity):
            self._service_block_mutation(c, intensity)

        n = self.problem.num_customers
        flip_rate = min(0.15, self.cfg.service_flip_rate * intensity)
        drone_rate = min(0.20, self.cfg.drone_mutation_rate * intensity)
        for customer in range(1, n + 1):
            if self.rng.random() < flip_rate:
                # Heavy customers remain truck-only.
                if (c.service[customer] == 1
                        or self.problem.customer_demands[customer] <= self.problem.drone_payload_capacity + EPS):
                    c.service[customer] = 1 - int(c.service[customer])
            if self.problem.num_drones_per_vehicle > 1 and self.rng.random() < drone_rate:
                old = int(c.drone[customer]) % self.problem.num_drones_per_vehicle
                choices = [x for x in range(self.problem.num_drones_per_vehicle) if x != old]
                c.drone[customer] = self.rng.choice(choices)
        _invalidate(c)

    def _random_immigrant(self) -> Chromosome:
        n = self.problem.num_customers
        s = [0] * (n + 1)
        d = [0] * (n + 1)
        drone_prob = self.rng.uniform(0.22, 0.64)
        for customer in range(1, n + 1):
            eligible = self.problem.customer_demands[customer] <= self.problem.drone_payload_capacity + EPS
            s[customer] = int(eligible and self.rng.random() < drone_prob)
            d[customer] = self.rng.randrange(max(1, self.problem.num_drones_per_vehicle))
        return Chromosome(self._random_route_string(), s, d)

    def _signature(self, c: Chromosome) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return (
            tuple(int(x) for x in c.route_string),
            tuple(int(x) for x in c.service[1:]),
            tuple(int(x) for x in c.drone[1:]),
        )

    def _deadline_reached(self, started: float) -> bool:
        """True once the ``time_limit_seconds`` budget is spent (0.0 = no limit)."""
        return (
            self.cfg.time_limit_seconds > 0
            and time.perf_counter() - started >= self.cfg.time_limit_seconds
        )

    def solve(self, verbose: bool = True) -> Chromosome:
        # Runtime covers the complete GA: population construction + initial
        # evaluation + all evolutionary generations.
        started = time.perf_counter()
        self.time_limit_hit = False
        self.evaluation_count = 0
        pop = self.initial_population()

        # Initial-population evaluation is by far the most expensive step on the
        # largest instances: 120 individuals x 0.19 s = 23 s on M-n200, before a
        # single generation runs.  Without a check here a 10 s budget returned
        # after 26 s, which is not a time-budget-comparable run at all.  Stop on
        # the deadline and keep the prefix that was actually evaluated; always
        # evaluate at least one individual so there is a solution to return.
        evaluated = []
        for c in pop:
            if evaluated and self._deadline_reached(started):
                self.time_limit_hit = True
                break
            self.evaluate(c)
            evaluated.append(c)
        pop = evaluated
        if self.time_limit_hit and verbose:
            print(
                f"GA time limit {self.cfg.time_limit_seconds:.1f}s reached while "
                f"evaluating the initial population "
                f"({len(pop)}/{self.cfg.population_size} individuals)"
            )
        best = min(pop, key=lambda x: x.fitness).clone()
        self.initial_best = float(best.fitness)
        self._last_emitted_best = float(best.fitness)
        self._emit_history("initial", 0)
        stagnation = 0

        generation_source = (
            itertools.count() if self.cfg.generations <= 0
            else range(self.cfg.generations + 1)
        )
        for gen in generation_source:
            # Wall-clock budget, checked once per generation.  The population
            # that was just evaluated is complete, so the incumbent stays valid.
            if self._deadline_reached(started):
                self.time_limit_hit = True
                if verbose:
                    print(
                        f"GA time limit {self.cfg.time_limit_seconds:.1f}s reached "
                        f"at gen={gen}"
                    )
                break
            pop.sort(key=lambda x: x.fitness)
            if pop[0].fitness < best.fitness - 1e-12:
                best = pop[0].clone()
                self._last_emitted_best = float(best.fitness)
                self._emit_history("improvement", gen)
                stagnation = 0
            elif gen > 0:
                stagnation += 1

            finite = [x.fitness for x in pop if math.isfinite(x.fitness)]
            unique_count = len({self._signature(x) for x in pop})
            rec = {
                "generation": float(gen),
                "best": float(pop[0].fitness),
                "mean": float(statistics.fmean(finite)) if finite else math.inf,
                "global_best": float(best.fitness),
                "unique": float(unique_count),
                "stagnation": float(stagnation),
            }
            self.history.append(rec)
            print_every = max(1, self.cfg.generations // 20) if self.cfg.generations > 0 else 10
            if verbose and (gen == 0 or (self.cfg.generations > 0 and gen == self.cfg.generations) or gen % print_every == 0):
                print(
                    f"gen={gen:4d} best={rec['best']:10.3f}s "
                    f"mean={rec['mean']:10.3f}s global={rec['global_best']:10.3f}s "
                    f"unique={unique_count:3d} stagn={stagnation:3d}"
                )
            if self.cfg.generations > 0 and gen >= self.cfg.generations:
                break

            # Ha et al.-style diversity reaction in a deliberately simple form:
            # stagnation raises mutation intensity and immigrant share, while
            # elite solutions remain untouched.
            window = max(1, int(self.cfg.stagnation_window))
            mutation_intensity = 1.0 + min(1.5, stagnation / window)
            immigrant_rate = self.cfg.random_immigrant_rate
            if stagnation >= window:
                immigrant_rate += min(0.18, 0.02 * (stagnation // window))
            if unique_count < int(0.55 * self.cfg.population_size):
                immigrant_rate = max(immigrant_rate, 0.10)

            generation_finite_mean = statistics.fmean(finite) if finite else pop[0].fitness
            next_pop = [pop[i].clone() for i in range(self.cfg.elite_size)]
            seen = {self._signature(x) for x in next_pop}
            immigrant_count = int(round(immigrant_rate * self.cfg.population_size))
            target_children = max(self.cfg.elite_size, self.cfg.population_size - immigrant_count)

            attempts = 0
            max_attempts = 8 * self.cfg.population_size
            while len(next_pop) < target_children and attempts < max_attempts:
                # Fine-grained budget checkpoint: one generation on the largest
                # instance (M-n200) costs ~16 s, so checking only between
                # generations turned a 10 s budget into 26 s.  Checking once per
                # offspring pair bounds the overshoot by one crossover+evaluate.
                if self._deadline_reached(started):
                    self.time_limit_hit = True
                    break
                attempts += 1
                pa, pb = self._tournament(pop), self._tournament(pop)
                ca, cb = self.crossover(pa, pb)

                # Poorer parents receive stronger mutation; elites/near-elites
                # receive gentler mutation (adaptive mutation principle).
                best_fit = pop[0].fitness
                finite_mean = generation_finite_mean
                def child_intensity(parent_a, parent_b):
                    pf = 0.5 * (parent_a.fitness + parent_b.fitness)
                    if not math.isfinite(pf) or not math.isfinite(finite_mean) or finite_mean <= best_fit + EPS:
                        return mutation_intensity
                    quality = max(0.0, min(1.0, (pf - best_fit) / (finite_mean - best_fit + EPS)))
                    return mutation_intensity * (0.70 + 0.90 * quality)

                inten = child_intensity(pa, pb)
                self.mutate(ca, inten)
                self.mutate(cb, inten)
                self.evaluate(ca)
                self.evaluate(cb)
                for child in (ca, cb):
                    sig = self._signature(child)
                    if sig in seen:
                        # One extra mutation attempt before rejecting a duplicate.
                        self.mutate(child, min(2.6, inten * 1.25))
                        self.evaluate(child)
                        sig = self._signature(child)
                    if sig not in seen or len(next_pop) + 2 >= target_children:
                        next_pop.append(child)
                        seen.add(sig)
                    if len(next_pop) >= target_children:
                        break

            while len(next_pop) < self.cfg.population_size:
                if self._deadline_reached(started):
                    self.time_limit_hit = True
                    break
                x = self._random_immigrant()
                self.evaluate(x)
                next_pop.append(x)

            if self.time_limit_hit:
                # The generation was cut short.  Do not top it up with random
                # immigrants -- that would burn time and dilute the pool for no
                # gain.  The partially built pool already holds the elites plus
                # every child that was evaluated, so fold the best of them into
                # the incumbent and stop here.
                for child in next_pop:
                    if child.fitness < best.fitness - 1e-12:
                        best = child.clone()
                        self._last_emitted_best = float(best.fitness)
                        self._emit_history("improvement", gen)
                if verbose:
                    print(
                        f"GA time limit {self.cfg.time_limit_seconds:.1f}s reached "
                        f"during gen={gen} ({len(next_pop)}/{self.cfg.population_size} "
                        f"individuals built); stopping"
                    )
                break

            pop = next_pop[:self.cfg.population_size]

        self.runtime_seconds = time.perf_counter() - started
        if verbose:
            print(f"GA runtime: {self.runtime_seconds:.3f}s")
        best.solution = None
        best.fitness = math.inf
        self.evaluate(best)
        return best


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def solution_to_dict(problem: ProblemData, best: Chromosome, history=None, runtime_seconds=None) -> Dict[str, object]:
    sol = best.solution or decode(problem, best)
    ok, reason = validate_solution(problem, sol)
    ev = sol.exact_evaluation or evaluate_solution(sol.vehicle_routes, sol.drone_routes, problem, True)
    out: Dict[str, object] = {
        "algorithm": "literature_informed_ga_v3",
        "feasible": bool(ok),
        "validation": reason,
        "makespan_s": float(sol.makespan_s),
        "vehicle_completion_times_s": [float(x) for x in ev.vehicle_completion_times],
        "battery_capacity_wh": float(problem.drone_battery_capacity_wh),
        "drone_payload_capacity_kg": float(problem.drone_payload_capacity),
        "truck_speed_m_s": float(problem.vehicle_speed),
        "drone_speed_m_s": float(problem.drone_speed),
        "drone_max_distance_constraint": False,
        "total_flight_energy_wh": float(ev.total_flight_energy_wh),
        "total_hover_energy_wh": float(ev.total_hover_energy_wh),
        "vehicle_routes": sol.vehicle_routes,
        "drone_routes": sol.drone_routes,
        "route_string": list(best.route_string),
        "trucks": [],
    }
    if runtime_seconds is not None:
        out["runtime_seconds"] = float(runtime_seconds)
    for r in sol.routes:
        item = {
            "truck": r.truck + 1,
            "master_sequence": list(r.master_sequence),
            "route": list(r.route),
            "completion_time_s": float(r.completion_time_s),
            "drone_energy_wh": [float(x) for x in r.drone_energy_wh],
            "sorties": [],
        }
        for s in r.sorties:
            item["sorties"].append({
                "drone": s.drone + 1,
                "drone_label": s.drone_label,
                "launch": s.launch,
                "deliveries": list(s.deliveries),
                "recovery": s.recovery,
                "launch_index": s.launch_index,
                "recovery_index": s.recovery_index,
                "payload_kg": float(sum(problem.customer_demands[c] for c in s.deliveries)),
                "flight_time_s": float(s.flight_time_s),
                "flight_energy_wh": float(s.flight_energy_wh),
                "hover_time_s": float(s.hover_time_s),
                "hover_energy_wh": float(s.hover_energy_wh),
                "total_energy_wh": float(s.total_energy_wh),
            })
        out["trucks"].append(item)
    if history is not None:
        out["history"] = list(history)
    return out


def save_result(path: str, problem: ProblemData, best: Chromosome, history=None, runtime_seconds=None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution_to_dict(problem, best, history, runtime_seconds), f, ensure_ascii=False, indent=2)


def _alns_style_energy_lines(problem: ProblemData, sol: DecodedSolution) -> List[str]:
    """Build the same human-readable energy/SOC section used by ALNS."""
    ev = sol.exact_evaluation or evaluate_solution(
        sol.vehicle_routes, sol.drone_routes, problem, check_coverage=True
    )
    lines = ["", "===== 无人机能耗与电池 (Wh, 单位: m/s/kg/Wh) ====="]
    if not ev.feasible:
        lines.append(f"solution_infeasible={ev.reason}")
        return lines

    lines.append(f"battery_capacity_wh={problem.drone_battery_capacity_wh:.1f}")
    lines.append(f"payload_capacity_kg={problem.drone_payload_capacity:.2f}")
    lines.append(f"total_flight_energy_wh={ev.total_flight_energy_wh:.4f}")
    lines.append(f"total_hover_energy_wh={ev.total_hover_energy_wh:.4f}")
    for report in ev.sorties:
        lines.append(
            "sortie vehicle={} drone={} launch={} rendezvous={} order={} "
            "start_payload_kg={:.4f} flight_time_s={:.2f} "
            "flight_energy_wh={:.4f} hover_time_s={:.2f} hover_energy_wh={:.4f} "
            "total_energy_wh={:.4f}".format(
                report.vehicle_idx + 1, report.drone_label, report.launch,
                report.rendezvous, report.order, report.start_payload_kg,
                report.flight_time_s, report.flight_energy_wh,
                report.hover_time_s, report.hover_energy_wh,
                report.total_energy_wh,
            )
        )
    for (vehicle_idx, label), state in sorted(ev.drones.items()):
        lines.append(
            "drone vehicle={} label={} sorties={} used_wh={:.4f} "
            "remaining_energy_wh={:.4f}".format(
                vehicle_idx + 1, label, len(state.sortie_keys),
                state.energy_wh, state.remaining_energy_wh,
            )
        )
    lines.append(f"makespan_s={ev.makespan:.4f}")
    return lines


def format_solution_details_alns_style(
    problem: ProblemData, best: Chromosome, include_header: bool = False
) -> str:
    """Format a GA solution exactly like ``model.print_solution_details``.

    Truck routes are printed as node IDs. For each drone sortie, launch/recovery
    are converted to positions in that truck route while delivery customers stay
    as original customer IDs, matching the ALNS result-file convention.
    """
    sol = best.solution or decode(problem, best)
    ev = sol.exact_evaluation or evaluate_solution(
        sol.vehicle_routes, sol.drone_routes, problem, check_coverage=True
    )
    lines: List[str] = []
    if include_header:
        lines.append("===== 配送方案详情 =====")

    lines.append("truck_route=")
    cleaned_routes: List[List[int]] = []
    for route in sol.vehicle_routes:
        cleaned: List[int] = []
        for node in route:
            node = int(node)
            if not cleaned or node != cleaned[-1]:
                cleaned.append(node)
        cleaned_routes.append(cleaned)
    lines.append("[" + ",\n".join(str(route) for route in cleaned_routes) + "]")

    # Match ALNS: one global bucket per physical drone label encountered.
    drone_task_map: Dict[str, List[List[int]]] = {}
    for vehicle_idx, sorties in enumerate(sol.drone_routes):
        route = cleaned_routes[vehicle_idx]
        first_occurrence: Dict[int, int] = {}
        last_occurrence: Dict[int, int] = {}
        for idx, node in enumerate(route):
            if node not in first_occurrence:
                first_occurrence[node] = idx
            last_occurrence[node] = idx

        for sortie in sorties:
            drone_label, launch, deliveries, rendezvous = sortie
            launch = int(launch)
            rendezvous = int(rendezvous)
            launch_idx = first_occurrence.get(launch, launch)
            rendezvous_idx = last_occurrence.get(rendezvous, rendezvous)
            path = [int(launch_idx)] + [int(c) for c in deliveries] + [int(rendezvous_idx)]
            drone_task_map.setdefault(str(drone_label), []).append(path)

    for idx, (_label, paths) in enumerate(drone_task_map.items(), start=1):
        lines.append(f"drone_routes{idx}=")
        lines.append(str(paths))

    lines.append(f"\n总时间成本: {ev.objective:.2f}")
    lines.extend(_alns_style_energy_lines(problem, sol))
    lines.append(f"\n总时长: {ev.objective:.2f} " )
    return "\n".join(lines)


def save_solution_details_alns_style(
    path: str, problem: ProblemData, best: Chromosome
) -> None:
    text = format_solution_details_alns_style(problem, best, include_header=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def print_solution(problem: ProblemData, best: Chromosome) -> None:
    sol = best.solution or decode(problem, best)
    ok, reason = validate_solution(problem, sol)
    ev = sol.exact_evaluation
    print("\n===== LITERATURE-INFORMED GA V3 BEST SOLUTION =====")
    print(f"feasible={ok} ({reason})")
    print(f"makespan={sol.makespan_s:.6f} s")
    if ev:
        print(f"flight_energy={ev.total_flight_energy_wh:.6f} Wh")
        print(f"hover_energy={ev.total_hover_energy_wh:.6f} Wh")
    for r in sol.routes:
        print(f"Truck {r.truck + 1}: completion={r.completion_time_s:.3f}s, truck_customers={len(r.route)-2}, sorties={len(r.sorties)}")
        print("  route=", r.route)
        for s in r.sorties:
            q = sum(problem.customer_demands[c] for c in s.deliveries)
            print(f"  {s.drone_label}: {s.launch}->{s.deliveries}->{s.recovery}, payload={q:.3f}kg, energy={s.total_energy_wh:.3f}Wh")
