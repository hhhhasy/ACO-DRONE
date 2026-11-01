from __future__ import annotations
import argparse, json, os, time
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from load_data import build_model_kwargs, load_data_from_json
from model import vrp_with_drones


def _tag(x: float) -> str:
    return ("%g" % x).replace('.', 'p')


def main():
    p = argparse.ArgumentParser(description="ALNS equal-wall-clock runner")
    p.add_argument("data")
    p.add_argument("--time-limit", type=float, required=True, help="TOTAL algorithm budget in seconds")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--initial-strategy", choices=["farthest", "nearest"], default="farthest")
    p.add_argument("--clustering", choices=["sweep", "kmeans"], default="kmeans")
    p.add_argument("--output-dir", default="time_budget_results")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")

    algorithm_start = time.perf_counter()
    data = load_data_from_json(args.data)
    if data is None:
        raise SystemExit("failed to load dataset: %s" % args.data)
    model = vrp_with_drones(**build_model_kwargs(data), seed=args.seed)

    initial_routes, initial_drones, initial_time = model.generate_initial_solution(
        initial_strategy=args.initial_strategy,
        clustering_method=args.clustering,
        verbose=False,
    )
    init_elapsed = time.perf_counter() - algorithm_start
    remaining = max(0.0, args.time_limit - init_elapsed)

    if remaining > 1e-9:
        best_routes, best_drones, best_time, info = model.local_search(
            initial_routes, initial_drones,
            iterations=0,  # 0 = no iteration cap in the patched model
            time_limit_seconds=remaining,
            verbose=not args.quiet,
            return_info=True,
        )
    else:
        best_routes, best_drones, best_time = initial_routes, initial_drones, float(initial_time)
        info = {"search_runtime_seconds": 0.0, "iterations_used": 0, "best_iteration": 0,
                "time_to_best_seconds": 0.0, "time_to_target_seconds": None}

    actual_runtime = time.perf_counter() - algorithm_start
    ev = model.evaluate_solution(best_routes, best_drones)
    if not ev.feasible:
        raise RuntimeError("ALNS returned infeasible solution: " + ev.reason)

    base = os.path.splitext(os.path.basename(args.data))[0]
    outdir = os.path.join(args.output_dir, "ALNS", _tag(args.time_limit) + "s", "seed%d" % args.seed)
    os.makedirs(outdir, exist_ok=True)
    sol_path = os.path.join(outdir, "solution_%s.txt" % base)
    json_path = os.path.join(outdir, "summary_%s.json" % base)
    model.print_solution_details(best_routes, best_drones, file_path=sol_path,
                                 total_time=best_time, verbose=False)
    payload = {
        "algorithm": "ALNS", "dataset": base, "seed": args.seed,
        "time_budget_seconds": float(args.time_limit),
        "actual_algorithm_runtime_seconds": float(actual_runtime),
        "initialization_runtime_seconds": float(init_elapsed),
        "search_runtime_seconds": float(info["search_runtime_seconds"]),
        "iterations_used": int(info["iterations_used"]),
        "initial_makespan_s": float(initial_time),
        "best_makespan_s": float(best_time),
        "feasible": True,
        "vehicle_completion_times": [float(x) for x in ev.vehicle_completion_times],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("ALNS budget=%.3fs actual=%.3fs initial=%.4f best=%.4f iterations=%d" %
          (args.time_limit, actual_runtime, initial_time, best_time, info["iterations_used"]))
    print(json_path)

if __name__ == "__main__":
    main()
