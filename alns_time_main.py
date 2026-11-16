"""Run ALNS from ACO's exact pre-search solution under a total time budget.

Place beside model.py, aco_solver.py and their existing dependencies.
Example: python alns_time_main.py data.json --time-limit 10 --seed 13
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import time

os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
from aco_solver import ACOConfig, AntColonySolver, load_problem
from model import vrp_with_drones


def build_aco_initial(data_path, seed):
    problem = load_problem(data_path)
    # Construct only: never run ACO search. The active aco_solver.py controls
    # the candidate fraction and exact feasibility repair.
    initializer = AntColonySolver(problem, ACOConfig(seed=seed))
    initial = initializer.global_best.clone()
    model = vrp_with_drones(
        num_customers=problem.num_customers,
        num_vehicles=problem.num_vehicles,
        num_drones_per_vehicle=problem.num_drones_per_vehicle,
        vehicle_speed=problem.vehicle_speed,
        drone_speed=problem.drone_speed,
        drone_payload_capacity=problem.drone_payload_capacity,
        drone_battery_capacity_wh=problem.drone_battery_capacity_wh,
        depot_coords=problem.depot_coords,
        customer_coords=problem.customer_coords,
        customer_demands=problem.customer_demands[1:], seed=seed,
    )
    routes, drones = copy.deepcopy((initial.vehicle_routes, initial.drone_routes))
    ev = model.evaluate_solution(routes, drones)
    if not ev.feasible or not math.isclose(ev.objective, initial.objective, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError('ACO initial solution failed ALNS validation: ' + ev.reason)
    return model, routes, drones, float(ev.objective)


def main():
    parser = argparse.ArgumentParser(description='ALNS with ACO initialization; total wall-clock budget')
    parser.add_argument('data')
    parser.add_argument('--time-limit', type=float, required=True)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--output-dir', default='time_budget_results_aco_initial')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be finite and positive')

    started = time.perf_counter()
    model, routes, drones, initial_value = build_aco_initial(args.data, args.seed)
    # Preserve the exact supplied solution for auditing, before any search.
    initial_payload = {'vehicle_routes': copy.deepcopy(routes),
                       'drone_routes': copy.deepcopy(drones), 'makespan_s': initial_value}
    signature = hashlib.sha256(json.dumps(
        [routes, drones], separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')).hexdigest()
    init_elapsed = time.perf_counter() - started
    remaining = args.time_limit - init_elapsed
    if remaining > 0:
        best_routes, best_drones, best_value, info = model.local_search(
            routes, drones, iterations=0, time_limit_seconds=remaining,
            reset_rng=True, return_info=True, verbose=not args.quiet,
        )
    else:
        best_routes, best_drones, best_value = routes, drones, initial_value
        info = {'search_runtime_seconds': 0.0, 'iterations_used': 0,
                'evaluations': 0, 'tabu_tenure': model.tabu_tenure}
    actual_runtime = time.perf_counter() - started
    ev = model.evaluate_solution(best_routes, best_drones)
    if not ev.feasible or ev.objective > initial_value + 1e-7:
        raise RuntimeError('ALNS returned infeasible or worse-than-initial result: ' + ev.reason)
    if not math.isclose(ev.objective, best_value, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError('ALNS returned inconsistent objective')

    base = os.path.splitext(os.path.basename(args.data))[0]
    tag = ('%g' % args.time_limit).replace('.', 'p')
    outdir = os.path.join(args.output_dir, 'ALNS', tag + 's', 'seed%d' % args.seed)
    os.makedirs(outdir, exist_ok=True)
    model.print_solution_details(best_routes, best_drones,
        file_path=os.path.join(outdir, 'solution_%s.txt' % base),
        total_time=ev.objective, verbose=False)
    initial_path = os.path.join(outdir, 'initial_%s.json' % base)
    initial_payload['signature_sha256'] = signature
    with open(initial_path, 'w', encoding='utf-8') as f:
        json.dump(initial_payload, f, ensure_ascii=False, indent=2)
    payload = dict(info)
    payload.update(algorithm='ALNS', dataset=base, seed=args.seed,
        time_budget_seconds=args.time_limit,
        actual_algorithm_runtime_seconds=actual_runtime,
        initialization_runtime_seconds=init_elapsed,
        initialization_method='aco_pre_search_global_best',
        initial_signature_sha256=signature,
        initialization_exhausted_budget=remaining <= 0,
        initial_makespan_s=initial_value, best_makespan_s=float(ev.objective),
        feasible=True, vehicle_completion_times=ev.vehicle_completion_times)
    path = os.path.join(outdir, 'summary_%s.json' % base)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('ALNS(ACO init) budget=%.3fs actual=%.3fs initial=%.4f best=%.4f' %
          (args.time_limit, actual_runtime, initial_value, ev.objective))
    print(path)


if __name__ == '__main__':
    main()
