from __future__ import annotations
import argparse, json, os, time
from aco_solver import ACOConfig, AntColonySolver, format_solution_details, load_problem, save_solution_txt


def _tag(x: float) -> str:
    return ("%g" % x).replace('.', 'p')


def main():
    p = argparse.ArgumentParser(description="ACO equal-wall-clock runner")
    p.add_argument("data")
    p.add_argument("--time-limit", type=float, required=True, help="TOTAL algorithm budget in seconds")
    p.add_argument("--ants", type=int, default=30)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--q0", type=float, default=0.85)
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--xi", type=float, default=0.10)
    p.add_argument("--sector-bias", type=float, default=2.0)
    p.add_argument("--candidate-list-size", type=int, default=18)
    p.add_argument("--restart-after", type=int, default=45)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output-dir", default="time_budget_results_fast")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")

    algorithm_start = time.perf_counter()
    problem = load_problem(args.data)
    cfg = ACOConfig(ants=args.ants, iterations=0, alpha=args.alpha, beta=args.beta,
                    q0=args.q0, rho=args.rho, xi=args.xi,
                    sector_bias=args.sector_bias, candidate_list_size=args.candidate_list_size,
                    restart_after=args.restart_after, seed=args.seed,
                    verbose_every=(0 if args.quiet else 10),
                    time_limit_seconds=args.time_limit)
    solver = AntColonySolver(problem, cfg)
    solver.cfg.time_limit_seconds = max(1e-9, args.time_limit - (time.perf_counter() - algorithm_start))
    best, history, solver_runtime = solver.solve()
    actual_runtime = time.perf_counter() - algorithm_start

    base = os.path.splitext(os.path.basename(args.data))[0]
    outdir = os.path.join(args.output_dir, "ACO", _tag(args.time_limit) + "s", "seed%d" % args.seed)
    os.makedirs(outdir, exist_ok=True)
    sol_path = os.path.join(outdir, "solution_%s.txt" % base)
    json_path = os.path.join(outdir, "summary_%s.json" % base)
    save_solution_txt(sol_path, problem, best)
    payload = {
        "algorithm": "ACO", "dataset": base, "seed": args.seed,
        "time_budget_seconds": float(args.time_limit),
        "actual_algorithm_runtime_seconds": float(actual_runtime),
        "solver_runtime_seconds": float(solver_runtime),
        "ants_per_iteration": int(args.ants),
        "iterations_limit": 0,
        "iterations_started": int(getattr(solver, "iterations_used", 0)),
        "completed_history_iterations": len(history),
        "best_makespan_s": float(best.objective),
        "feasible": bool(best.feasible),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("ACO budget=%.3fs actual=%.3fs ants=%d best=%.4f completed_iters=%d" %
          (args.time_limit, actual_runtime, args.ants, best.objective, len(history)))
    if not args.quiet:
        print(format_solution_details(problem, best))
    print(json_path)

if __name__ == "__main__":
    main()
