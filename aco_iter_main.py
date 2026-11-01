"""Iteration-limited runner for the original (unmodified) ACO/ACS solver.

Place next to aco_solver.py, drone_physics.py, evaluator.py,
and fast_window_physics.py. No time limit is imposed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from aco_solver import (
    ACOConfig,
    AntColonySolver,
    format_solution_details,
    load_problem,
    save_solution_txt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Original ACO with fixed iteration count (no time limit)")
    parser.add_argument("data", help="Path to a physicalized JSON dataset")
    parser.add_argument("--iterations", type=int, default=80, help="Number of complete generations")
    parser.add_argument("--ants", type=int, default=50, help="Ants per generation")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--assignment-alpha", type=float, default=1.0)
    parser.add_argument("--service-alpha", type=float, default=1.0)
    parser.add_argument("--service-beta", type=float, default=1.5)
    parser.add_argument("--q0", type=float, default=0.8)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--xi", type=float, default=0.10)
    parser.add_argument("--sector-bias", type=float, default=2.0)
    parser.add_argument("--candidate-list-size", type=int, default=18)
    parser.add_argument("--restart-after", type=int, default=45)
    parser.add_argument("--output-dir", default="iteration_results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be > 0 for iteration-only mode")
    if args.ants <= 0:
        parser.error("--ants must be > 0")

    start = time.perf_counter()
    problem = load_problem(args.data)
    cfg = ACOConfig(
        ants=args.ants,
        iterations=args.iterations,
        time_limit_seconds=0.0,  # IMPORTANT: No wall-clock limit
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        assignment_alpha=args.assignment_alpha,
        service_alpha=args.service_alpha,
        service_beta=args.service_beta,
        q0=args.q0,
        rho=args.rho,
        xi=args.xi,
        sector_bias=args.sector_bias,
        candidate_list_size=args.candidate_list_size,
        restart_after=args.restart_after,
        verbose_every=0 if args.quiet else 10,
    )
    solver = AntColonySolver(problem, cfg)
    best, history, solver_runtime = solver.solve()
    actual_runtime = time.perf_counter() - start

    dataset = os.path.splitext(os.path.basename(args.data))[0]
    outdir = os.path.join(args.output_dir, "ACO", f"{args.iterations}iter", f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    sol_path = os.path.join(outdir, f"solution_{dataset}.txt")
    json_path = os.path.join(outdir, f"summary_{dataset}.json")
    save_solution_txt(sol_path, problem, best)
    summary = {
        "algorithm": "ACO",
        "mode": "fixed_iterations_no_time_limit",
        "dataset": dataset,
        "seed": args.seed,
        "iterations_limit": args.iterations,
        "iterations_started": solver.iterations_used,
        "completed_history_iterations": len(history),
        "ants_per_iteration": args.ants,
        "time_budget_seconds": None,
        "actual_algorithm_runtime_seconds": actual_runtime,
        "solver_runtime_seconds": solver_runtime,
        "best_makespan_s": best.objective,
        "feasible": best.feasible,
        "config": vars(cfg),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"ACO no_time_limit iterations={len(history)}/{args.iterations} "
        f"ants={args.ants} seed={args.seed} "
        f"best={best.objective:.4f}s runtime={actual_runtime:.3f}s"
    )
    if not args.quiet:
        print(format_solution_details(problem, best))
    print("Solution:", sol_path)
    print("Summary:", json_path)


if __name__ == "__main__":
    main()
