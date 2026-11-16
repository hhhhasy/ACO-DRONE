from __future__ import annotations
import argparse, json, os, time
import copy, hashlib, math
from types import SimpleNamespace
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from aco_solver import ACOConfig, AntColonySolver, load_problem
from model import vrp_with_drones
from vnd_solver import UserNeighborhoodVND


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


def _tag(x: float) -> str:
    return ("%g" % x).replace('.', 'p')


def main():
    p = argparse.ArgumentParser(description="VND equal-wall-clock runner")
    p.add_argument("data")
    p.add_argument("--time-limit", type=float, required=True, help="TOTAL algorithm budget in seconds")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--trials-per-neighborhood", type=int, default=20)
    p.add_argument("--initial-strategy", choices=["aco", "farthest", "nearest"], default="aco",
                   help="Compatibility option: all values now use ACO initialization")
    p.add_argument("--clustering", choices=["sweep", "kmeans"], default="sweep",
                   help="Accepted for compatibility; unused with ACO initialization")
    p.add_argument("--output-dir", default="time_budget_results")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")

    algorithm_start = time.perf_counter()
    model, initial_routes, initial_drones, initial_time = build_aco_initial(args.data, args.seed)
    initial_payload = {
        "vehicle_routes": copy.deepcopy(initial_routes),
        "drone_routes": copy.deepcopy(initial_drones),
        "makespan_s": initial_time,
    }
    signature = hashlib.sha256(json.dumps(
        [initial_routes, initial_drones], separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')).hexdigest()
    initial_payload["signature_sha256"] = signature
    init_elapsed = time.perf_counter() - algorithm_start
    remaining = args.time_limit - init_elapsed
    if remaining > 0:
        solver = UserNeighborhoodVND(
            model=model, trials_per_neighborhood=args.trials_per_neighborhood,
            max_evaluations=0, time_limit_seconds=remaining, seed=args.seed,
            verbose=not args.quiet, repeat_until_budget=True)
        result = solver.run(initial_routes, initial_drones)
    else:
        result = SimpleNamespace(
            vehicle_routes=initial_routes, drone_routes=initial_drones,
            best_time=initial_time, search_runtime_seconds=0.0, evaluations=0,
            improvements=0, neighborhood_passes=0, terminated_by_time=True,
            accepted_improvements=0, vns_iterations=0, terminated_by_time_limit=True,
        )
    actual_runtime = time.perf_counter() - algorithm_start
    ev = model.evaluate_solution(result.vehicle_routes, result.drone_routes)
    if not ev.feasible or ev.objective > initial_time + 1e-7:
        raise RuntimeError("VND returned infeasible solution: " + ev.reason)

    if not math.isclose(ev.objective, result.best_time, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError("Search returned inconsistent objective")

    base = os.path.splitext(os.path.basename(args.data))[0]
    outdir = os.path.join(args.output_dir, "VND", _tag(args.time_limit) + "s", "seed%d" % args.seed)
    os.makedirs(outdir, exist_ok=True)
    sol_path = os.path.join(outdir, "solution_%s.txt" % base)
    json_path = os.path.join(outdir, "summary_%s.json" % base)
    model.print_solution_details(result.vehicle_routes, result.drone_routes,
                                 file_path=sol_path, total_time=result.best_time, verbose=False)
    with open(os.path.join(outdir, "initial_%s.json" % base), "w", encoding="utf-8") as f:
        json.dump(initial_payload, f, ensure_ascii=False, indent=2)
    payload = {
        "initialization_method": "aco_pre_search_global_best",
        "initial_signature_sha256": signature,
        "initialization_exhausted_budget": remaining <= 0,
        "algorithm":"VND", "dataset":base, "seed":args.seed,
        "time_budget_seconds":float(args.time_limit),
        "actual_algorithm_runtime_seconds":float(actual_runtime),
        "initialization_runtime_seconds":float(init_elapsed),
        "search_runtime_seconds":float(result.search_runtime_seconds),
        "trials_per_neighborhood":int(args.trials_per_neighborhood),
        "initial_makespan_s":float(initial_time), "best_makespan_s":float(result.best_time),
        "evaluations":int(result.evaluations), "improvements":int(result.improvements),
        "neighborhood_passes":int(result.neighborhood_passes), "feasible":True,
        "terminated_by_time_limit":bool(result.terminated_by_time),
    }
    with open(json_path,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    print("VND budget=%.3fs actual=%.3fs initial=%.4f best=%.4f eval=%d" %
          (args.time_limit, actual_runtime, initial_time, result.best_time, result.evaluations))
    print(json_path)

if __name__ == "__main__": main()
