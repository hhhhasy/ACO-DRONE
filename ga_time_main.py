from __future__ import annotations
import argparse, json, os, time
from ga_solver import ClassicGA, GAConfig, load_problem, print_solution, save_solution_details_alns_style


def _tag(x: float) -> str:
    return ("%g" % x).replace('.', 'p')


def main():
    p = argparse.ArgumentParser(description="GA equal-wall-clock runner")
    p.add_argument("data")
    p.add_argument("--time-limit", type=float, required=True, help="TOTAL algorithm budget in seconds")
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--tournament", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output-dir", default="time_budget_results_fast")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")

    algorithm_start = time.perf_counter()
    problem = load_problem(args.data)
    cfg = GAConfig(population_size=args.population, generations=0,
                   elite_size=args.elite, tournament_size=args.tournament,
                   seed=args.seed, time_limit_seconds=args.time_limit)
    ga = ClassicGA(problem, cfg)
    cfg.time_limit_seconds = max(1e-9, args.time_limit - (time.perf_counter() - algorithm_start))
    best = ga.solve(verbose=not args.quiet)
    actual_runtime = time.perf_counter() - algorithm_start
    sol = best.solution
    if sol is None:
        ga.evaluate(best); sol = best.solution

    base = os.path.splitext(os.path.basename(args.data))[0]
    outdir = os.path.join(args.output_dir, "GA", _tag(args.time_limit) + "s", "seed%d" % args.seed)
    os.makedirs(outdir, exist_ok=True)
    sol_path = os.path.join(outdir, "solution_%s.txt" % base)
    json_path = os.path.join(outdir, "summary_%s.json" % base)
    save_solution_details_alns_style(sol_path, problem, best)
    payload = {
        "algorithm": "GA", "dataset": base, "seed": args.seed,
        "time_budget_seconds": float(args.time_limit),
        "actual_algorithm_runtime_seconds": float(actual_runtime),
        "solver_runtime_seconds": float(ga.runtime_seconds),
        "population_size": int(args.population),
        "generations_limit": 0,
        "generations_completed": int(ga.history[-1]["generation"]) if ga.history else 0,
        "best_makespan_s": float(best.fitness),
        "feasible": bool(sol and sol.exact_evaluation and sol.exact_evaluation.feasible),
        "history_records": len(ga.history),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("GA budget=%.3fs actual=%.3fs population=%d best=%.4f generations=%d" %
          (args.time_limit, actual_runtime, args.population, best.fitness, payload["generations_completed"]))
    if not args.quiet:
        print_solution(problem, best)
    print(json_path)

if __name__ == "__main__":
    main()
