"""统一的无人机物理与能耗模型 (single source of truth for drone physics).

单位约定 (units used everywhere in this project)
-----------------------------------------------
    距离 distance : m
    时间 time     : s
    速度 speed    : m/s
    质量 mass     : kg
    功率 power    : W
    能量 energy   : Wh

All coordinates in the benchmark JSON are already physicalised to metres by
``physicalize_datasets.py`` (original coordinate * 100); nothing in this module
(or any runtime module) may scale coordinates or demands again.

Power model (Dorling-type induced-power / momentum-theory hover power)
---------------------------------------------------------------------
    P(q) = (m_frame + m_battery + q) ** 1.5
           * sqrt( g**3 / (2 * rho * A_rotor * n_rotor) )

with ``q`` the *actual* payload carried on the current flight segment (kg).
The result is in W.

There is deliberately **no maximum flight distance** here: whether a sortie is
feasible is decided by payload + real flight energy + hover energy + the
remaining battery of the physical drone that flies it.
"""

import math

# --------------------------------------------------------------------------
# Fixed problem constants (paper configuration, do not tune at runtime)
# --------------------------------------------------------------------------
TRUCK_SPEED_KMH = 30.0
DRONE_SPEED_KMH = 60.0

TRUCK_SPEED_M_S = TRUCK_SPEED_KMH / 3.6          # 8.333333333 m/s
DRONE_SPEED_M_S = DRONE_SPEED_KMH / 3.6          # 16.666666667 m/s

DRONE_PAYLOAD_CAPACITY_KG = 3.0                  # exactly 3.0 kg is feasible
DRONE_BATTERY_CAPACITY_WH = 200.0                # per physical drone, no recharge

DRONE_FRAME_MASS_KG = 3.0
DRONE_BATTERY_MASS_KG = 1.5

GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
ROTOR_DISC_AREA_M2 = 0.126
ROTOR_COUNT = 4

#: ``sqrt(g**3 / (2 * rho * A * n))`` -- the payload-independent part of P(q).
POWER_COEFFICIENT = math.sqrt(
    GRAVITY_M_S2 ** 3
    / (2.0 * AIR_DENSITY_KG_M3 * ROTOR_DISC_AREA_M2 * ROTOR_COUNT)
)

#: Mass that always has to be lifted, payload excluded.
DRONE_EMPTY_MASS_KG = DRONE_FRAME_MASS_KG + DRONE_BATTERY_MASS_KG

#: Wh per (kg**1.5 * m): lets us turn a distance into energy with one multiply.
_ENERGY_PER_POW_M = 1.0 / (DRONE_SPEED_M_S * 3600.0) * POWER_COEFFICIENT

EPS = 1e-9

#: Above this many deliveries per sortie the exact energy DP is replaced by a
#: cached greedy + 2-opt heuristic (19! orderings would be intractable).
MAX_EXACT_DP_DELIVERIES = 12


# --------------------------------------------------------------------------
# Power / energy primitives
# --------------------------------------------------------------------------
def drone_power_w(payload_kg):
    """Dorling-type electrical power (W) required to carry ``payload_kg``."""
    payload = float(payload_kg)
    if payload < 0.0:
        payload = 0.0
    return (DRONE_EMPTY_MASS_KG + payload) ** 1.5 * POWER_COEFFICIENT


def hover_power_w():
    """Power while hovering with all deliveries dropped (payload = 0 kg)."""
    return drone_power_w(0.0)


def flight_time_s(distance_m, speed_m_s=DRONE_SPEED_M_S):
    """``time = distance / speed`` -- no other conversion anywhere."""
    return float(distance_m) / float(speed_m_s)


def segment_energy_wh(distance_m, payload_kg, speed_m_s=DRONE_SPEED_M_S):
    """Energy of one flight segment flown with ``payload_kg`` on board."""
    time_s = flight_time_s(distance_m, speed_m_s)
    power_w = drone_power_w(payload_kg)
    return {
        'distance_m': float(distance_m),
        'payload_kg': float(payload_kg),
        'power_w': power_w,
        'time_s': time_s,
        'energy_wh': power_w * time_s / 3600.0,
    }


def hover_energy_wh(hover_time_s, payload_kg=0.0):
    """Energy burned while hovering for ``hover_time_s`` seconds."""
    if hover_time_s <= 0.0:
        return 0.0
    return drone_power_w(payload_kg) * float(hover_time_s) / 3600.0


def energy_per_meter_wh(payload_kg, speed_m_s=DRONE_SPEED_M_S):
    """Wh burned per metre of flight at a given payload.

    Used to express an energy delta on the same scale as a distance delta when
    an operator has to compare a truck insertion with a drone insertion.  The
    transformation is a pure unit bridge: ordering by
    ``delta_energy / energy_per_meter`` is *identical* to ordering by
    ``delta_energy``.
    """
    return drone_power_w(payload_kg) / (float(speed_m_s) * 3600.0)


# --------------------------------------------------------------------------
# Sortie flight evaluation (payload decreases segment by segment)
# --------------------------------------------------------------------------
def evaluate_sortie_flight(launch, deliveries, rendezvous, dist_matrix, demands,
                           order=None, speed_m_s=DRONE_SPEED_M_S):
    """Full flight breakdown of one sortie: ``launch -> C1 -> ... -> Cn -> recovery``.

    Parameters
    ----------
    launch, rendezvous : int
        Node indices of the launch and recovery points (depot node 0 allowed).
    deliveries : sequence[int]
        Customers served by this sortie (a set; the visiting order comes from
        ``order`` when given).
    dist_matrix : 2-D indexable
        Drone (Euclidean) distance matrix in metres.
    demands : indexable
        ``demands[c]`` = demand of customer ``c`` in kg.
    order : sequence[int], optional
        Explicit visiting order.  When omitted, ``deliveries`` is visited in the
        order given.

    Returns
    -------
    dict with ``start_payload_kg``, ``segments`` (per-leg payload / power /
    time / energy), ``flight_time_s`` and ``flight_energy_wh``.

    Notes
    -----
    The payload on a leg is the *remaining* load:
    ``launch -> C1`` carries everything, each customer's demand is subtracted
    only after it has been served, and the leg back to the recovery point
    always flies with ``payload_kg = 0.0``.
    """
    launch = int(launch)
    rendezvous = int(rendezvous)
    visit_order = [int(c) for c in (deliveries if order is None else order)]

    start_payload_kg = 0.0
    for c in visit_order:
        start_payload_kg += float(demands[c])

    segments = []
    payload_kg = start_payload_kg
    prev = launch
    sequence = visit_order + [rendezvous]
    last_index = len(sequence) - 1
    for idx, node in enumerate(sequence):
        dist = float(dist_matrix[prev, node])
        segment = segment_energy_wh(dist, payload_kg, speed_m_s)
        segment['from'] = prev
        segment['to'] = int(node)
        segments.append(segment)
        if idx < last_index:
            # The demand of the customer just served is dropped only now.  The
            # subtraction is snapped to zero so that the last delivery leaves
            # exactly ``0.0`` kg on board (no 1e-16 residue).
            payload_kg -= float(demands[node])
            if payload_kg < EPS:
                payload_kg = 0.0
        prev = node

    flight_time = sum(s['time_s'] for s in segments)
    flight_energy = sum(s['energy_wh'] for s in segments)
    return {
        'launch': launch,
        'rendezvous': rendezvous,
        'order': list(visit_order),
        'start_payload_kg': start_payload_kg,
        'segments': segments,
        'flight_time_s': flight_time,
        'flight_energy_wh': flight_energy,
    }


def sortie_energy_wh(launch, deliveries, rendezvous, dist_matrix, demands,
                     order=None, speed_m_s=DRONE_SPEED_M_S):
    """Flight-only energy (Wh) of one sortie in the given visiting order."""
    return evaluate_sortie_flight(
        launch, deliveries, rendezvous, dist_matrix, demands,
        order=order, speed_m_s=speed_m_s,
    )['flight_energy_wh']


def sortie_payload_kg(deliveries, demands):
    """Total payload that has to be lifted at launch (kg)."""
    return sum(float(demands[c]) for c in deliveries)


def payload_within_capacity(deliveries, demands, capacity_kg=DRONE_PAYLOAD_CAPACITY_KG):
    """``sum(demands) <= capacity`` -- exactly ``capacity`` is feasible."""
    return sortie_payload_kg(deliveries, demands) <= capacity_kg + EPS


# --------------------------------------------------------------------------
# Energy-optimal delivery order (payload-aware DP)
# --------------------------------------------------------------------------
#: Cache of optimal orders.  Keyed by the *identity* of the distance matrix and
#: demand vector; both objects are kept alive inside the value so that a
#: recycled ``id()`` can never produce a stale hit.
_ORDER_CACHE = {}
_ORDER_CACHE_MAX = 200000


def clear_order_cache():
    """Drop every cached delivery order (used by tests)."""
    _ORDER_CACHE.clear()


def _greedy_energy_order(launch, rendezvous, deliveries, dist_matrix, demands):
    """Fallback heuristic for very large sorties: greedy, then 2-opt."""
    remaining = list(deliveries)
    total_payload = sum(float(demands[c]) for c in remaining)
    order = []
    prev = launch
    payload = total_payload
    while remaining:
        best = None
        best_cost = None
        for c in remaining:
            cost = drone_power_w(payload) * float(dist_matrix[prev, c])
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = c
        order.append(best)
        remaining.remove(best)
        payload = max(0.0, payload - float(demands[best]))
        prev = best

    def _energy(seq):
        return sortie_energy_wh(launch, seq, rendezvous, dist_matrix, demands)

    improved = True
    best_energy = _energy(order)
    while improved and len(order) >= 2:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + list(reversed(order[i:j + 1])) + order[j + 1:]
                candidate_energy = _energy(candidate)
                if candidate_energy < best_energy - 1e-12:
                    order, best_energy = candidate, candidate_energy
                    improved = True
        # 2-opt restarts after every accepted move; the loop terminates because
        # each accepted move strictly decreases the energy.
    return order


def _exact_energy_order(launch, rendezvous, deliveries, dist_matrix, demands):
    """Exact DP over (delivered set, last customer) minimising flight energy.

    The remaining payload on a leg only depends on *which* customers have
    already been served (not on the path), so ``(mask, last)`` is a sufficient
    state and the DP is exact.
    """
    n = len(deliveries)
    size = 1 << n
    inf = float('inf')

    # Demand already delivered for each mask.
    mask_demand = [0.0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        mask_demand[mask] = mask_demand[mask ^ lsb] + float(demands[deliveries[idx]])

    total_payload = mask_demand[size - 1]

    # Wh per metre for each possible payload state, precomputed once.
    leg_factor = [0.0] * size
    for mask in range(size):
        payload = total_payload - mask_demand[mask]
        if payload < 0.0:
            payload = 0.0
        leg_factor[mask] = (DRONE_EMPTY_MASS_KG + payload) ** 1.5 * _ENERGY_PER_POW_M

    dp = [[inf] * n for _ in range(size)]
    prev = [[-1] * n for _ in range(size)]

    for u in range(n):
        dp[1 << u][u] = leg_factor[0] * float(dist_matrix[launch, deliveries[u]])

    for mask in range(1, size):
        row = dp[mask]
        factor = leg_factor[mask]
        for u in range(n):
            current = row[u]
            if current == inf or not (mask & (1 << u)):
                continue
            u_node = deliveries[u]
            remaining = ((1 << n) - 1) ^ mask
            v = 0
            while remaining:
                if remaining & 1:
                    new_mask = mask | (1 << v)
                    new_energy = current + factor * float(dist_matrix[u_node, deliveries[v]])
                    if new_energy < dp[new_mask][v]:
                        dp[new_mask][v] = new_energy
                        prev[new_mask][v] = u
                remaining >>= 1
                v += 1

    full = size - 1
    best_last = 0
    best_energy = inf
    # The recovery leg always flies with an empty payload, so it is weighted by
    # the payload-free factor -- exactly like ``evaluate_sortie_flight`` does.
    recovery_factor = leg_factor[full]
    for u in range(n):
        candidate = (
            dp[full][u]
            + recovery_factor * float(dist_matrix[deliveries[u], rendezvous])
        )
        if candidate < best_energy:
            best_energy = candidate
            best_last = u

    order_local = []
    mask = full
    u = best_last
    while u != -1:
        order_local.append(u)
        pu = prev[mask][u]
        mask &= ~(1 << u)
        u = pu
    return [deliveries[i] for i in reversed(order_local)]


def energy_optimal_sortie(launch, rendezvous, deliveries, dist_matrix, demands):
    """Energy-optimal visiting order for a fixed (launch, customer set, recovery).

    Returns ``dict`` with ``order`` (list of customers), ``flight_energy_wh``
    (Wh), ``start_payload_kg`` and the full per-segment breakdown.

    Optimising the order matters because the order determines the payload of
    every leg, and the payload determines the power: two orders over the same
    customer set have the same distance but generally *different* energy.
    """
    launch = int(launch)
    rendezvous = int(rendezvous)
    customers = sorted(int(c) for c in deliveries)
    if not customers:
        return evaluate_sortie_flight(launch, [], rendezvous, dist_matrix, demands)

    key = (id(dist_matrix), id(demands), launch, rendezvous, tuple(customers))
    cached = _ORDER_CACHE.get(key)
    if cached is not None:
        order = list(cached[0])
    else:
        if len(customers) <= MAX_EXACT_DP_DELIVERIES:
            order = _exact_energy_order(launch, rendezvous, customers, dist_matrix, demands)
        else:
            order = _greedy_energy_order(launch, rendezvous, customers, dist_matrix, demands)
        if len(_ORDER_CACHE) >= _ORDER_CACHE_MAX:
            _ORDER_CACHE.clear()
        # Keep references to both keyed objects so their ids stay unique.
        _ORDER_CACHE[key] = (tuple(order), dist_matrix, demands)

    return evaluate_sortie_flight(
        launch, customers, rendezvous, dist_matrix, demands, order=order
    )


def best_sortie_energy_wh(launch, rendezvous, deliveries, dist_matrix, demands):
    """Flight energy (Wh) of a sortie flown in its energy-optimal order."""
    return energy_optimal_sortie(
        launch, rendezvous, deliveries, dist_matrix, demands
    )['flight_energy_wh']


def dp_energy_fixed_lr(l_idx, r_idx, d_indices, dro_dist_matrix, demands):
    """Energy-aware counterpart of :func:`utils.dp_tsp_fixed_lr`.

    Same call convention -- fixed launch ``l_idx``, fixed recovery ``r_idx``, a
    set of drone customers ``d_indices`` -- but the order returned minimises
    ``flight_energy_wh`` instead of total flight distance.

    Returns ``(launch, ordered_deliveries, rendezvous)``.
    """
    l_idx, r_idx = int(l_idx), int(r_idx)
    deliveries = [int(x) for x in d_indices]
    if not deliveries:
        return l_idx, [], r_idx
    order = energy_optimal_sortie(
        l_idx, r_idx, deliveries, dro_dist_matrix, demands
    )['order']
    return l_idx, list(order), r_idx
