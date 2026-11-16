"""Unified truck-drone solution evaluator.

Everything that needs to know "what does this solution cost, and is it
physically feasible?" goes through :func:`evaluate_solution`.  No other module
re-implements the drone energy formula, the hover logic, or the battery/SOC
accounting.

Handled here
------------
* truck-drone synchronisation (truck waits for the drone at the recovery node);
* **hover** energy: a drone that reaches a non-depot recovery node before the
  truck hovers there until the truck arrives, burning
  ``power(0 kg) * hover_time``; at the final depot (node 0) it lands
  immediately and does not hover while waiting for the truck;
* per-sortie payload, per-segment payload / power / time / energy;
* cumulative battery consumption of every **physical** drone
  (``remaining_energy_wh -= sortie_energy_wh`` and ``remaining >= 0``);
* structural validity: route shape, customer coverage, duplicate deliveries,
  launch-before-recovery order, per-drone temporal ordering, payload capacity.

Removed on purpose
------------------
There is no ``distance <= drone_max_distance`` test anywhere: range is no
longer a constraint.  A sortie is limited only by payload, real flight energy,
hover energy and the remaining battery.
"""

from dataclasses import dataclass, field

from drone_physics import (
    DRONE_BATTERY_CAPACITY_WH,
    DRONE_PAYLOAD_CAPACITY_KG,
    evaluate_sortie_flight,
    hover_energy_wh,
    hover_power_w,
)

INF = float('inf')
EPS = 1e-9


@dataclass
class SortieReport:
    """Everything that happened to one sortie."""

    vehicle_idx: int
    sortie_idx: int
    drone_label: str
    launch: int
    rendezvous: int
    order: list
    launch_idx: int
    rendezvous_idx: int
    start_payload_kg: float
    payload_kg: float
    segments: list = field(default_factory=list)
    flight_time_s: float = 0.0
    flight_energy_wh: float = 0.0
    truck_arrival_s: float = 0.0
    drone_arrival_s: float = 0.0
    hover_time_s: float = 0.0
    hover_energy_wh: float = 0.0

    @property
    def total_energy_wh(self):
        return self.flight_energy_wh + self.hover_energy_wh


@dataclass
class DroneReport:
    """Cumulative battery state of one physical drone."""

    vehicle_idx: int
    drone_label: str
    sortie_keys: list = field(default_factory=list)
    energy_wh: float = 0.0
    remaining_energy_wh: float = DRONE_BATTERY_CAPACITY_WH

    @property
    def feasible(self):
        return self.remaining_energy_wh >= -EPS


@dataclass
class SolutionEvaluation:
    """Result of :func:`evaluate_solution`."""

    feasible: bool
    reason: str = ''
    makespan: float = INF
    vehicle_completion_times: list = field(default_factory=list)
    sorties: list = field(default_factory=list)
    drones: dict = field(default_factory=dict)
    total_flight_energy_wh: float = 0.0
    total_hover_energy_wh: float = 0.0

    @property
    def total_energy_wh(self):
        return self.total_flight_energy_wh + self.total_hover_energy_wh

    @property
    def objective(self):
        return self.makespan if self.feasible else INF

    def drone_state(self, vehicle_idx, drone_label):
        return self.drones.get((int(vehicle_idx), str(drone_label)))


def _infeasible(reason, **kwargs):
    return SolutionEvaluation(feasible=False, reason=reason, makespan=INF, **kwargs)


def _dist_matrix(model, name):
    matrix = getattr(model, name, None)
    if matrix is None:
        raise AttributeError("model is missing %s" % name)
    return matrix


def evaluate_solution(vehicle_routes, drone_routes, model, check_coverage=True):
    """Evaluate a complete truck-drone solution.

    Parameters
    ----------
    vehicle_routes : list[list[int]]
        One closed truck route per vehicle, starting and ending at depot 0.
    drone_routes : list[list[list]]
        ``drone_routes[v][s] = [drone_label, launch, [deliveries in visiting
        order], rendezvous]``.
    model : object
        Object exposing ``num_vehicles``, ``num_customers``,
        ``vehicle_speed``, ``drone_speed``, ``veh_dist_matrix``,
        ``dro_dist_matrix``, ``customer_demands``, ``drone_payload_capacity``
        and (optionally) ``drone_battery_capacity_wh``.
    check_coverage : bool
        When True (default) every customer must be served exactly once.

    Returns
    -------
    SolutionEvaluation
        ``feasible`` is False and ``makespan`` is ``inf`` for any candidate that
        violates a physical or structural constraint, so an infeasible
        candidate can never be mistaken for a good one.
    """
    if vehicle_routes is None or drone_routes is None:
        return _infeasible('missing routes')

    num_vehicles = int(getattr(model, 'num_vehicles', len(vehicle_routes)))
    if len(vehicle_routes) != num_vehicles or len(drone_routes) != num_vehicles:
        return _infeasible('vehicle/sortie group count mismatch')

    try:
        vehicle_speed = float(model.vehicle_speed)
        drone_speed = float(model.drone_speed)
    except (AttributeError, TypeError, ValueError):
        return _infeasible('missing speeds')
    if vehicle_speed <= 0.0 or drone_speed <= 0.0:
        return _infeasible('non-positive speed')

    try:
        veh_dist = _dist_matrix(model, 'veh_dist_matrix')
        dro_dist = _dist_matrix(model, 'dro_dist_matrix')
        demands = model.customer_demands
    except AttributeError as exc:
        return _infeasible(str(exc))

    payload_capacity = float(
        getattr(model, 'drone_payload_capacity', DRONE_PAYLOAD_CAPACITY_KG)
    )
    battery_capacity = float(
        getattr(model, 'drone_battery_capacity_wh', DRONE_BATTERY_CAPACITY_WH)
    )

    n_customers = int(getattr(model, 'num_customers', 0) or 0)
    n_nodes = n_customers + 1
    served = [0] * (n_nodes + 1)

    sortie_reports = []
    drone_states = {}
    vehicle_completion_times = [0.0] * num_vehicles
    total_flight_energy = 0.0
    total_hover_energy = 0.0

    for vehicle_idx, route_raw in enumerate(vehicle_routes):
        try:
            route = [int(x) for x in route_raw]
        except (TypeError, ValueError):
            return _infeasible('non-integer truck node')
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return _infeasible('truck route must start and end at the depot')
        if any(node < 0 or node >= n_nodes for node in route):
            return _infeasible('truck node out of range')
        if any(node == 0 for node in route[1:-1]):
            return _infeasible('intermediate depot visit is not allowed')

        # ---- truck coverage, node positions, base arrival times ------------
        pos = {}
        base_arrival = [0.0] * len(route)
        truck_time = 0.0
        prev = route[0]
        for idx in range(1, len(route)):
            node = route[idx]
            truck_time += float(veh_dist[prev, node]) / vehicle_speed
            base_arrival[idx] = truck_time
            if idx < len(route) - 1:
                if served[node]:
                    return _infeasible('customer %d served twice' % node)
                served[node] += 1
                pos[node] = idx
            prev = node

        sorties = drone_routes[vehicle_idx]
        if sorties is None:
            return _infeasible('missing sortie list')
        if not sorties:
            vehicle_completion_times[vehicle_idx] = truck_time
            continue

        # ---- collect and validate every sortie of this vehicle ------------
        rendez_map = {}
        last_rendez_per_drone = {}
        for sortie_idx, sortie in enumerate(sorties):
            if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                return _infeasible('malformed sortie %r' % (sortie,))
            label, launch_raw, deliveries_raw, rendez_raw = sortie
            try:
                launch = int(launch_raw)
                rendezvous = int(rendez_raw)
                order = [int(c) for c in deliveries_raw]
            except (TypeError, ValueError):
                return _infeasible('non-integer sortie field')
            if not order:
                return _infeasible('empty sortie')
            if sorted(order) != sorted(set(order)):
                return _infeasible('duplicate customer inside a sortie')
            for customer in order:
                if customer <= 0 or customer > n_customers:
                    return _infeasible('drone customer out of range')
                if served[customer]:
                    return _infeasible('customer %d served twice' % customer)
                served[customer] += 1

            launch_idx = 0 if launch == 0 else pos.get(launch)
            rendez_idx = len(route) - 1 if rendezvous == 0 else pos.get(rendezvous)
            if launch_idx is None or rendez_idx is None or launch_idx >= rendez_idx:
                return _infeasible('launch/recovery order violated')

            # The same physical drone may only fly again after it has recovered.
            drone_label = str(label)
            previous_rendez = last_rendez_per_drone.get(drone_label)
            if previous_rendez is not None and launch_idx < previous_rendez:
                return _infeasible('drone %s re-launched before recovery' % drone_label)
            last_rendez_per_drone[drone_label] = rendez_idx

            # ---- payload + real flight energy of this sortie --------------
            start_payload = sum(float(demands[c]) for c in order)
            if start_payload > payload_capacity + EPS:
                return _infeasible(
                    'payload %.6f kg exceeds capacity %.6f kg'
                    % (start_payload, payload_capacity)
                )
            flight = evaluate_sortie_flight(
                launch, order, rendezvous, dro_dist, demands,
                order=order, speed_m_s=drone_speed,
            )

            report = SortieReport(
                vehicle_idx=vehicle_idx,
                sortie_idx=sortie_idx,
                drone_label=str(label),
                launch=launch,
                rendezvous=rendezvous,
                order=order,
                launch_idx=launch_idx,
                rendezvous_idx=rendez_idx,
                start_payload_kg=start_payload,
                payload_kg=start_payload,
                segments=flight['segments'],
                flight_time_s=flight['flight_time_s'],
                flight_energy_wh=flight['flight_energy_wh'],
            )
            sortie_reports.append(report)
            rendez_map.setdefault(rendez_idx, []).append(report)

        # ---- truck-drone synchronisation + hover --------------------------
        effective_arrival = [0.0] * len(route)
        accumulated_wait = 0.0
        for idx, base_time in enumerate(base_arrival):
            truck_arrival = base_time + accumulated_wait
            effective_arrival[idx] = truck_arrival
            tasks = rendez_map.get(idx)
            if not tasks:
                continue

            latest_drone = truck_arrival
            for report in tasks:
                drone_arrival = (
                    effective_arrival[report.launch_idx] + report.flight_time_s
                )
                report.truck_arrival_s = truck_arrival
                report.drone_arrival_s = drone_arrival
                # The load has been fully delivered on arrival, so any hover
                # happens with payload 0 kg.  The final depot is special: once
                # the drone reaches node 0 it can land immediately and therefore
                # does not hover merely because the truck has not returned yet.
                # The team completion time is still synchronized below via
                # ``latest_drone``: if the drone is late, the truck waits for it.
                if report.rendezvous == 0:
                    report.hover_time_s = 0.0
                    report.hover_energy_wh = 0.0
                else:
                    report.hover_time_s = max(0.0, truck_arrival - drone_arrival)
                    report.hover_energy_wh = hover_energy_wh(report.hover_time_s, 0.0)
                if drone_arrival > latest_drone:
                    latest_drone = drone_arrival

            if latest_drone > truck_arrival:
                # Drone is late: the truck waits for it.
                accumulated_wait += latest_drone - truck_arrival
                effective_arrival[idx] = latest_drone

        vehicle_completion_times[vehicle_idx] = truck_time + accumulated_wait

    if check_coverage:
        for customer in range(1, n_customers + 1):
            if served[customer] != 1:
                return _infeasible('customer %d served %d times' % (customer, served[customer]))

    # ---- per physical drone battery / SOC accounting -----------------------
    # A physical drone is identified by (vehicle, label).  Its sorties are
    # charged in chronological order and the consumption accumulates; nothing
    # recharges, and a sortie may not be flown with an empty battery.
    for report in sortie_reports:
        key = (report.vehicle_idx, report.drone_label)
        state = drone_states.get(key)
        if state is None:
            state = DroneReport(
                vehicle_idx=report.vehicle_idx,
                drone_label=report.drone_label,
                remaining_energy_wh=battery_capacity,
            )
            drone_states[key] = state
        state.sortie_keys.append((report.vehicle_idx, report.sortie_idx))
        state.energy_wh += report.total_energy_wh
        state.remaining_energy_wh = battery_capacity - state.energy_wh
        total_flight_energy += report.flight_energy_wh
        total_hover_energy += report.hover_energy_wh

    for state in drone_states.values():
        if not state.feasible:
            return _infeasible(
                'drone %d-%s battery exhausted (%.3f Wh > %.3f Wh)'
                % (state.vehicle_idx + 1, state.drone_label,
                   state.energy_wh, battery_capacity)
            )

    makespan = max(vehicle_completion_times) if vehicle_completion_times else 0.0
    return SolutionEvaluation(
        feasible=True,
        reason='ok',
        makespan=makespan,
        vehicle_completion_times=vehicle_completion_times,
        sorties=sortie_reports,
        drones=drone_states,
        total_flight_energy_wh=total_flight_energy,
        total_hover_energy_wh=total_hover_energy,
    )


def solution_makespan(vehicle_routes, drone_routes, model, check_coverage=True):
    """``inf`` when infeasible, otherwise the truck-drone makespan in seconds."""
    return evaluate_solution(
        vehicle_routes, drone_routes, model, check_coverage=check_coverage
    ).objective


def drone_energy_used_wh(drone_routes, model, vehicle_idx, drone_label):
    """Energy already spent by one physical drone before any new sortie."""
    total = 0.0
    for group_idx, sorties in enumerate(drone_routes):
        for sortie in sorties or []:
            if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                continue
            if group_idx != int(vehicle_idx) or str(sortie[0]) != str(drone_label):
                continue
            try:
                launch = int(sortie[1])
                rendezvous = int(sortie[3])
                order = [int(c) for c in sortie[2]]
            except (TypeError, ValueError):
                continue
            if not order:
                continue
            total += evaluate_sortie_flight(
                launch, order, rendezvous, model.dro_dist_matrix,
                model.customer_demands, order=order, speed_m_s=model.drone_speed,
            )['flight_energy_wh']
    return total


def sortie_hover_energy_wh(hover_time_s):
    """Hover energy for a given hovering duration (payload already delivered)."""
    return hover_energy_wh(hover_time_s, 0.0)


def hover_power():
    """Power (W) drawn while hovering with an empty payload."""
    return hover_power_w()
