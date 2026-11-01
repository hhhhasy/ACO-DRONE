"""Exact fixed-order sortie metrics for repeated drone launch/recovery windows.

No pruning, no approximate physics: the same per-leg operations and Python sum
order as drone_physics.evaluate_sortie_flight, without constructing dictionaries
for each candidate. Interior legs are invariant for a fixed delivery group.
"""
from drone_physics import drone_power_w


def prepare_fixed_order_flight(problem, deliveries):
    seq = [int(c) for c in deliveries]
    if not seq:
        raise ValueError("A sortie must have at least one delivery")
    demands = problem.customer_demands
    dm = problem.dro_dist_matrix
    speed = float(problem.drone_speed)
    payload = sum(float(demands[c]) for c in seq)
    first_power = drone_power_w(payload)
    last_power = drone_power_w(0.0)
    inner_times = []
    inner_energies = []
    remaining = payload
    prev = seq[0]
    for node in seq[1:]:
        remaining -= float(demands[prev])
        if remaining < 1e-9:  # matches drone_physics.EPS
            remaining = 0.0
        t = float(dm[prev, node]) / speed
        inner_times.append(t)
        inner_energies.append(drone_power_w(remaining) * t / 3600.0)
        prev = node

    first_node, last_node = seq[0], seq[-1]
    # Exactly the sequence and summation order in evaluate_sortie_flight.
    def flight(launch, recovery):
        t_first = float(dm[int(launch), first_node]) / speed
        t_last = float(dm[last_node, int(recovery)]) / speed
        e_first = first_power * t_first / 3600.0
        e_last = last_power * t_last / 3600.0
        time_s = sum((t_first, *inner_times, t_last))
        energy_wh = sum((e_first, *inner_energies, e_last))
        return time_s, energy_wh
    return flight
