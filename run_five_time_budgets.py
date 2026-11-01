from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import math
import statistics
from collections import defaultdict
from typing import Iterable, List


ALGORITHMS = ("ALNS","VND","VNS")



def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join('"%s"' % x if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def resolve_datasets(
    data: str | None,
    data_dir: str | None,
    pattern: str,
    limit: int,
) -> List[str]:
    """Resolve either one positional JSON file or a batch selected from --data-dir."""
    if data and data_dir:
        raise ValueError("use either positional DATA or --data-dir, not both")
    if not data and not data_dir:
        raise ValueError("provide a JSON DATA file or --data-dir")

    if data:
        path = _normalise_path(data)
        if not os.path.isfile(path):
            raise ValueError("dataset does not exist: %s" % data)
        if not path.lower().endswith(".json"):
            raise ValueError("dataset must be a .json file: %s" % data)
        return [path]

    root = _normalise_path(data_dir or "")
    if not os.path.isdir(root):
        raise ValueError("data directory does not exist: %s" % data_dir)

    # glob() supports patterns such as '*.json', 'A-*.json', 'M-n*.json', and
    # recursive patterns containing '**'.  Files are sorted for reproducibility.
    paths = sorted(
        p for p in glob.glob(os.path.join(root, pattern), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(".json")
    )
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError(
            "no JSON datasets matched --data-dir %s --pattern %s" % (data_dir, pattern)
        )
    return [_normalise_path(p) for p in paths]


def _budget_matches(value, budgets: Iterable[float]) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(x - float(b)) <= 1e-9 for b in budgets)


def write_combined_summary(
    output_dir: str,
    selected_datasets: List[str],
    seeds: List[int],
    budgets: List[float],
) -> str:
    """Write one CSV for exactly the datasets/seeds/budgets in this invocation."""
    selected_bases = {os.path.splitext(os.path.basename(p))[0] for p in selected_datasets}
    seed_set = {int(x) for x in seeds}

    rows = []
    pattern = os.path.join(output_dir, "*", "*s", "seed*", "summary_*.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            if d.get("dataset") not in selected_bases:
                continue
            try:
                row_seed = int(d.get("seed"))
            except (TypeError, ValueError):
                continue
            if row_seed not in seed_set:
                continue
            if not _budget_matches(d.get("time_budget_seconds"), budgets):
                continue
            if str(d.get("algorithm")) not in ALGORITHMS:
                continue

            rows.append({
                "algorithm": d.get("algorithm"),
                "dataset": d.get("dataset"),
                "seed": row_seed,
                "time_budget_seconds": d.get("time_budget_seconds"),
                "actual_algorithm_runtime_seconds": d.get("actual_algorithm_runtime_seconds"),
                "best_makespan_s": d.get("best_makespan_s"),
                "feasible": d.get("feasible"),
                "evaluations": d.get("evaluations", ""),
                "iterations_or_generations": d.get(
                    "iterations_used",
                    d.get(
                        "vns_iterations",
                        d.get("generations_completed", d.get("completed_history_iterations", "")),
                    ),
                ),
                "summary_file": os.path.abspath(path),
            })
        except Exception as exc:
            print("Skipping summary %s: %s" % (path, exc))

    rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            float(r["time_budget_seconds"]),
            int(r["seed"]),
            ALGORITHMS.index(str(r["algorithm"])) if str(r["algorithm"]) in ALGORITHMS else 99,
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "algorithm",
            "dataset",
            "seed",
            "time_budget_seconds",
            "actual_algorithm_runtime_seconds",
            "best_makespan_s",
            "feasible",
            "evaluations",
            "iterations_or_generations",
            "summary_file",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return csv_path



def _as_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_statistics_summary(comparison_csv: str, output_dir: str) -> str:
    """Aggregate repeated seeds by dataset + algorithm + time budget.

    Quality statistics use feasible runs with finite makespan values. Runtime
    statistics use every run with a finite recorded runtime. ``std`` is the
    sample standard deviation (n-1); it is 0 when only one valid run exists.
    """
    groups = defaultdict(list)
    with open(comparison_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            budget = _as_float(row.get("time_budget_seconds"))
            if budget is None:
                continue
            key = (str(row.get("dataset", "")), str(row.get("algorithm", "")), budget)
            groups[key].append(row)

    stats_rows = []
    for (dataset, algorithm, budget), rows in groups.items():
        feasible_rows = [r for r in rows if _as_bool(r.get("feasible"))]
        makespans = [
            x for x in (_as_float(r.get("best_makespan_s")) for r in feasible_rows)
            if x is not None
        ]
        runtimes = [
            x for x in (_as_float(r.get("actual_algorithm_runtime_seconds")) for r in rows)
            if x is not None
        ]
        evaluations = [
            x for x in (_as_float(r.get("evaluations")) for r in rows)
            if x is not None
        ]

        if makespans:
            best = min(makespans)
            mean = statistics.fmean(makespans)
            std = statistics.stdev(makespans) if len(makespans) >= 2 else 0.0
            worst = max(makespans)
        else:
            best = mean = std = worst = ""

        stats_rows.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "time_budget_seconds": budget,
            "runs_total": len(rows),
            "runs_feasible": len(makespans),
            "best_makespan_s": best,
            "mean_makespan_s": mean,
            "std_makespan_s": std,
            "worst_makespan_s": worst,
            "mean_actual_runtime_seconds": statistics.fmean(runtimes) if runtimes else "",
            "min_actual_runtime_seconds": min(runtimes) if runtimes else "",
            "max_actual_runtime_seconds": max(runtimes) if runtimes else "",
            "mean_evaluations": statistics.fmean(evaluations) if evaluations else "",
        })

    stats_rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            float(r["time_budget_seconds"]),
            ALGORITHMS.index(str(r["algorithm"])) if str(r["algorithm"]) in ALGORITHMS else 99,
        )
    )

    stats_path = os.path.join(output_dir, "comparison_statistics.csv")
    with open(stats_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "dataset",
            "algorithm",
            "time_budget_seconds",
            "runs_total",
            "runs_feasible",
            "best_makespan_s",
            "mean_makespan_s",
            "std_makespan_s",
            "worst_makespan_s",
            "mean_actual_runtime_seconds",
            "min_actual_runtime_seconds",
            "max_actual_runtime_seconds",
            "mean_evaluations",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)
    return stats_path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run ALNS/GA/ACO/VND/VNS under equal total wall-clock budgets. "
            "Use either positional DATA for one instance or --data-dir for a batch."
        )
    )
    p.add_argument(
        "data",
        nargs="?",
        default=None,
        help="single physicalized benchmark JSON; omit when using --data-dir",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="directory containing benchmark JSON files for batch execution",
    )
    p.add_argument(
        "--pattern",
        default="*.json",
        help='glob pattern inside --data-dir, e.g. "A-*.json" or "**/*.json"',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="after sorting matched datasets, run at most the first N; 0 = all",
    )
    p.add_argument("--times", type=float, nargs="+", default=[30, 50, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=[13])
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--ants", type=int, default=30)
    p.add_argument("--vnd-trials", type=int, default=20)
    p.add_argument("--shake-trials", type=int, default=5)
    p.add_argument("--local-trials", type=int, default=10)
    p.add_argument("--output-dir", default="compare_results")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    if args.limit < 0:
        p.error("--limit must be >= 0")
    if any(t <= 0 for t in args.times):
        p.error("every --times value must be > 0")
    if args.population < 6:
        p.error("--population must be >= 6")
    if args.ants <= 0:
        p.error("--ants must be > 0")

    try:
        datasets = resolve_datasets(args.data, args.data_dir, args.pattern, args.limit)
    except ValueError as exc:
        p.error(str(exc))

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    print("\n===== Equal-time five-algorithm batch =====")
    print("datasets=%d" % len(datasets))
    for i, path in enumerate(datasets, 1):
        print("  [%d/%d] %s" % (i, len(datasets), path))
    print("times=%s" % args.times)
    print("seeds=%s" % args.seeds)
    print("GA population=%d; ACO ants=%d" % (args.population, args.ants))

    total_jobs = len(datasets) * len(args.seeds) * len(args.times) * len(ALGORITHMS)
    job = 0

    # Dataset-first order keeps all five algorithm comparisons for one instance
    # adjacent in the console and makes interrupted batch runs easier to inspect.
    for data_path in datasets:
        base = os.path.splitext(os.path.basename(data_path))[0]
        for seed in args.seeds:
            for budget in args.times:
                common = [
                    data_path,
                    "--time-limit", str(budget),
                    "--seed", str(seed),
                    "--output-dir", args.output_dir,
                    "--quiet",
                ]
                commands = [
                    ("ALNS", [py, os.path.join(here, "alns_time_main.py"), *common]),
                    # ("GA", [py, os.path.join(here, "ga_time_main.py"), *common,
                    #         "--population", str(args.population)]),
                    # ("ACO", [py, os.path.join(here, "aco_time_main.py"), *common,
                    #          "--ants", str(args.ants)]),
                    ("VND", [py, os.path.join(here, "vnd_time_main.py"), *common,
                            "--trials-per-neighborhood", str(args.vnd_trials)]),
                    ("VNS", [py, os.path.join(here, "vns_time_main.py"), *common,
                            "--shake-trials", str(args.shake_trials),
                            "--local-trials", str(args.local_trials)]),           
                ]
                for alg, cmd in commands:
                    job += 1
                    print(
                        "\n===== JOB %d/%d | %s | %s | seed=%d | budget=%gs ====="
                        % (job, total_jobs, base, alg, seed, budget),
                        flush=True,
                    )
                    run(cmd)

    csv_path = write_combined_summary(args.output_dir, datasets, args.seeds, args.times)
    stats_path = write_statistics_summary(csv_path, args.output_dir)
    print("\nPer-seed summary:", os.path.abspath(csv_path))
    print("Seed statistics:", os.path.abspath(stats_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
