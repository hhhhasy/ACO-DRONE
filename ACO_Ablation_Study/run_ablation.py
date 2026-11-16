"""Run ONE ablation job: one dataset, one variant, one seed, one budget.

The wall-clock accounting is identical to the official ``aco_time_main.py``:

1. start the clock before loading the instance;
2. construct the solver with the full requested budget;
3. subtract instance loading + solver initialisation from the budget;
4. call ``solve()``, which checks the deadline before every ant.

Output (never outside the chosen results root)::

    results/<variant>/<budget>s/<dataset>/seed<seed>/
        run.json       complete config, metrics, plan, pheromone state
        solution.txt   human-readable delivery plan
        history.csv    per-iteration convergence record

The script refuses to overwrite a completed run unless ``--overwrite`` is
given explicitly.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import platform
import sys
import time
from typing import Dict, List, Optional

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

# Import the study copy only.  This assertion prevents a stale ``aco_solver``
# from another directory (e.g. the original project root) being used by mistake.
import aco_solver  # noqa: E402
from aco_solver import (  # noqa: E402
    AntColonySolver,
    format_solution_details,
    load_problem,
    solution_to_dict,
)
import ablation_config as ac  # noqa: E402
import run_state  # noqa: E402

_EXPECTED_SOLVER = os.path.join(STUDY_DIR, "aco_solver.py")
if os.path.normcase(os.path.abspath(aco_solver.__file__)) != os.path.normcase(_EXPECTED_SOLVER):
    raise RuntimeError(
        "wrong aco_solver imported: %s (expected %s)"
        % (aco_solver.__file__, _EXPECTED_SOLVER)
    )

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ALREADY_COMPLETE = 3
EXIT_CONFLICT = 4

HISTORY_FIELDS = (
    "iteration",
    "iteration_best",
    "global_best",
    "mean_feasible",
    "feasible_ants",
    "ant_proposals",
    "tabu_hits",
    "tabu_retries",
    "tabu_forced_accepts",
    "exact_decode_calls",
    "evaluation_cache_hits",
)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_history_csv(path: str, history: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run one ACO+Tabu ablation job (dataset x variant x seed x budget). "
            "This script never launches a batch."
        )
    )
    p.add_argument("dataset", help="physicalized benchmark JSON")
    p.add_argument("--variant", required=True, choices=list(ac.VARIANT_KEYS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--time-limit", type=float, required=True,
                   help="total wall-clock budget in seconds (includes initialisation)")
    p.add_argument("--ants", type=int, default=ac.ANTS)
    p.add_argument("--iterations", type=int, default=0,
                   help="0 = stop on the time budget only (official setting)")
    p.add_argument("--results-root", default=os.path.join(STUDY_DIR, "results"),
                   help="root of the results tree")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing COMPLETED run for the same combination")
    return p


def _fail(p: argparse.ArgumentParser, message: str) -> None:
    p.error(message)


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.time_limit <= 0:
        _fail(p, "--time-limit must be > 0")
    if args.ants <= 0:
        _fail(p, "--ants must be > 0")
    if args.iterations < 0:
        _fail(p, "--iterations must be >= 0")
    if not os.path.isfile(args.dataset):
        _fail(p, "dataset does not exist: %s" % args.dataset)
    if not args.dataset.lower().endswith(".json"):
        _fail(p, "dataset must be a .json file: %s" % args.dataset)

    identity = ac.dataset_identity(args.dataset)
    dataset_name = str(identity["dataset"])
    dataset_sha = str(identity["dataset_sha256"])
    run_dir = ac.run_directory(
        args.results_root, args.variant, args.time_limit, dataset_name, args.seed
    )
    run_fingerprint = ac.run_config_fingerprint(
        args.variant, args.seed, args.time_limit,
        ants=args.ants, iterations=args.iterations,
    )
    code_fp = ac.code_fingerprint()

    state = run_state.inspect_run(
        run_dir,
        expected_run_fingerprint=run_fingerprint,
        expected_code_fingerprint=code_fp,
        expected_dataset_sha256=dataset_sha,
    )
    if state["state"] == run_state.COMPLETE and not args.overwrite:
        print(
            "[skip] completed run already exists: %s\n"
            "       (use --overwrite only if you intend to replace this exact run)"
            % run_dir
        )
        return EXIT_ALREADY_COMPLETE
    if state["state"] in (
        run_state.CONFIG_MISMATCH, run_state.CODE_MISMATCH,
        run_state.DATASET_MISMATCH, run_state.INVALID,
    ) and not args.overwrite:
        print("[conflict] %s: %s" % (state["state"], state["detail"]))
        print("           existing directory: %s" % run_dir)
        print("           rerun with --overwrite to replace this exact combination.")
        return EXIT_CONFLICT

    # ------------------------------------------------------------------
    # Exact wall-clock protocol of aco_time_main.py
    # ------------------------------------------------------------------
    algorithm_start = time.perf_counter()
    problem = load_problem(args.dataset)
    cfg = ac.build_config(
        args.variant,
        seed=args.seed,
        time_limit_seconds=args.time_limit,
        ants=args.ants,
        iterations=args.iterations,
        verbose_every=(0 if args.quiet else 10),
    )
    # Snapshot before the runner shortens the remaining budget.
    run_config = dict(vars(cfg))
    solver = AntColonySolver(problem, cfg)
    init_seconds = time.perf_counter() - algorithm_start
    remaining = max(1e-9, args.time_limit - init_seconds)
    solver.cfg.time_limit_seconds = remaining

    started_at = _utc_now()
    best, history, solver_runtime = solver.solve()
    actual_runtime = time.perf_counter() - algorithm_start
    finished_at = _utc_now()
    if not math.isfinite(float(best.objective)):
        print("[error] non-finite makespan; run not published")
        return 1

    # ------------------------------------------------------------------
    # Assemble the complete result record.
    # ------------------------------------------------------------------
    payload = solution_to_dict(
        problem, best, history=history,
        runtime_seconds=solver_runtime, config=cfg,
    )
    payload.update({
        "schema_version": ac.SCHEMA_VERSION,
        "status": ac.STATUS_COMPLETED,
        "algorithm": "ACO+Tabu (ablation study copy)",
        "variant": args.variant,
        "variant_title_zh": ac.get_variant(args.variant).title_zh,
        "variant_description_zh": ac.get_variant(args.variant).description_zh,
        "dataset": dataset_name,
        "dataset_path": identity["dataset_path"],
        "dataset_sha256": dataset_sha,
        "seed": int(args.seed),
        "requested_time_budget_seconds": float(args.time_limit),
        "effective_time_limit_seconds": float(solver.cfg.time_limit_seconds),
        "initialisation_seconds": float(init_seconds),
        "ants_per_iteration": int(args.ants),
        "iterations_limit": int(args.iterations),
        "iterations_started": int(getattr(solver, "iterations_used", 0)),
        "completed_history_iterations": int(len(history)),
        "time_limit_hit": bool(getattr(solver, "time_limit_hit", False)),
        "actual_algorithm_runtime_seconds": float(actual_runtime),
        "solver_runtime_seconds": float(solver_runtime),
        "best_makespan_s": float(best.objective),
        "makespan_s": float(best.objective),
        "feasible": bool(best.feasible),
        "evaluation_reason": str(best.evaluation.reason),
        "objective": "makespan_seconds",
        "run_config_fingerprint": run_fingerprint,
        "variant_config_fingerprint": ac.variant_fingerprint(args.variant),
        "code_fingerprint": code_fp,
        "solver_module_file": os.path.abspath(aco_solver.__file__),
        "pheromone": solver.pheromone_statistics(),
        "service_policy": solver.service_policy_statistics(),
        "tabu": solver.tabu_statistics(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_module_file": getattr(__import__("numpy"), "__file__", ""),
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "command": list(sys.argv if argv is None else [sys.executable, *argv]),
    })
    # ``solution_to_dict`` writes vars(cfg) after the runner shortened the
    # budget; replace it with the pre-run snapshot so the record is canonical.
    payload["config"] = run_config

    # Auxiliary files first; run.json is the atomic commit marker written last,
    # so a crash can never leave a "completed" record with missing plan files.
    _atomic_write_text(
        os.path.join(run_dir, ac.SOLUTION_FILENAME),
        format_solution_details(problem, best),
    )
    write_history_csv(os.path.join(run_dir, ac.HISTORY_FILENAME), history)
    _atomic_write_json(os.path.join(run_dir, ac.RUN_FILENAME), payload)

    print(
        "[done] variant=%s dataset=%s seed=%d budget=%gs "
        "makespan=%.4f feasible=%s runtime=%.2fs iters=%d"
        % (
            args.variant, dataset_name, args.seed, args.time_limit,
            float(best.objective), bool(best.feasible), actual_runtime,
            int(getattr(solver, "iterations_used", 0)),
        )
    )
    print(run_dir)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
