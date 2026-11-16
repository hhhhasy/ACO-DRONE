"""Drone battery/payload sensitivity experiment runner.

SAFETY GATE
-----------
Without ``--execute`` this program only prints the task matrix.  It does not
create a solver and does not run any optimisation.  The formal 750-task matrix
(15 parameter combinations x 5 algorithms x 10 seeds, 50 s each) is intentionally
NOT executed during development.  ACO is pinned to the current fixed100 service
policy (100% strategy B, kappa=1).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
for _path in (str(_RESEARCH_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adapters.algorithms import RunSpec, run_algorithm

DEFAULT_CONFIG = _HERE / "config" / "sensitivity_config.json"
ALGORITHMS = ("ACO", "GA", "ALNS", "PSO", "VND")
MANIFEST_FIELDS = [
    "run_id", "status", "algorithm", "dataset", "seed", "battery_wh",
    "payload_kg", "time_budget_seconds", "actual_runtime_seconds",
    "best_makespan_s", "feasible", "failure_reason", "initial_makespan_s",
    "iterations_or_generations", "evaluations", "drone_served_customers",
    "truck_served_customers", "drone_share_pct", "drone_sorties",
    "total_flight_energy_wh", "total_hover_energy_wh", "total_energy_wh",
    "battery_utilization_pct", "max_drone_battery_utilization_pct",
    "fleet_completion_cv", "max_minus_mean_completion_s", "run_file",
    "error_type", "error", "started_at_utc", "finished_at_utc",
]


def _tag(value: float) -> str:
    return ("%g" % float(value)).replace(".", "p")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_project_path(value: str, base: Path = _HERE) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def parameter_combinations(config: Dict[str, Any]) -> List[Tuple[float, float]]:
    battery_levels = [float(x) for x in config["battery_wh_levels"]]
    payload_levels = [float(x) for x in config["payload_kg_levels"]]
    baseline = config["baseline"]
    b0, p0 = float(baseline["battery_wh"]), float(baseline["payload_kg"])
    combos: List[Tuple[float, float]] = []
    for b in battery_levels:
        combos.append((b, p0))
    for p in payload_levels:
        if (b0, p) not in combos:
            combos.append((b0, p))
    # Stable and unique.
    out = []
    seen = set()
    for combo in combos:
        if combo not in seen:
            seen.add(combo)
            out.append(combo)
    return out


def build_tasks(
    config: Dict[str, Any],
    algorithms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    dataset: Optional[str] = None,
    battery_levels: Optional[Sequence[float]] = None,
    payload_levels: Optional[Sequence[float]] = None,
    output_dir: Optional[str] = None,
    time_budget: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg_algorithms = [str(x).upper() for x in (algorithms or config["algorithms"])]
    for alg in cfg_algorithms:
        if alg not in ALGORITHMS:
            raise ValueError("unknown algorithm %r" % alg)
    seed_values = [int(x) for x in (seeds or config["seeds"])]
    budget = float(time_budget if time_budget is not None else config["time_budget_seconds"])

    if battery_levels is not None or payload_levels is not None:
        base = dict(config)
        base["battery_wh_levels"] = (
            [float(x) for x in battery_levels]
            if battery_levels is not None else config["battery_wh_levels"]
        )
        base["payload_kg_levels"] = (
            [float(x) for x in payload_levels]
            if payload_levels is not None else config["payload_kg_levels"]
        )
        combos = parameter_combinations(base)
    else:
        combos = parameter_combinations(config)

    dataset_path = _resolve_project_path(dataset or config["dataset"])
    dataset_name = Path(str(dataset_path)).stem
    if not dataset_path.is_file():
        raise ValueError("dataset not found: %s" % dataset_path)
    out_dir = _resolve_project_path(output_dir or config.get("output_dir", "results"))
    params_by_alg = config.get("algorithm_parameters", {})

    tasks = []
    for alg in cfg_algorithms:
        for battery, payload in combos:
            for seed in seed_values:
                run_id = "%s__%s__b%s__p%s__s%d" % (
                    alg, dataset_name, _tag(battery), _tag(payload), seed
                )
                tasks.append({
                    "run_id": run_id,
                    "algorithm": alg,
                    "dataset_path": str(dataset_path),
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "time_budget_seconds": budget,
                    "battery_wh": float(battery),
                    "payload_kg": float(payload),
                    "params": dict(params_by_alg.get(alg, {})),
                    "output_dir": str(out_dir),
                })
    resolved = {
        "algorithms": cfg_algorithms,
        "seeds": seed_values,
        "time_budget_seconds": budget,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_name,
        "combinations": combos,
        "output_dir": str(out_dir),
        "expected_tasks": len(cfg_algorithms) * len(combos) * len(seed_values),
    }
    return tasks, resolved


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(task["output_dir"])
    run_path = output_dir / "runs" / (task["run_id"] + ".json")
    spec = RunSpec(
        algorithm=task["algorithm"],
        dataset_path=task["dataset_path"],
        dataset_name=task["dataset_name"],
        seed=int(task["seed"]),
        time_budget_seconds=float(task["time_budget_seconds"]),
        battery_wh=float(task["battery_wh"]),
        payload_kg=float(task["payload_kg"]),
        params=dict(task["params"]),
    )
    try:
        result = run_algorithm(spec)
        result.update({
            "run_id": task["run_id"],
            "status": "ok",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_file": str(run_path.relative_to(output_dir)).replace("\\", "/"),
        })
    except Exception as exc:  # noqa: BLE001
        result = {
            "run_id": task["run_id"],
            "status": "error",
            "algorithm": task["algorithm"],
            "dataset": task["dataset_name"],
            "seed": int(task["seed"]),
            "battery_wh": float(task["battery_wh"]),
            "payload_kg": float(task["payload_kg"]),
            "time_budget_seconds": float(task["time_budget_seconds"]),
            "feasible": False,
            "best_makespan_s": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_file": str(run_path.relative_to(output_dir)).replace("\\", "/"),
        }
    _atomic_json(run_path, result)
    return _manifest_row(result)


def _manifest_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    row = {key: payload.get(key, "") for key in MANIFEST_FIELDS}
    if row.get("best_makespan_s") is None:
        row["best_makespan_s"] = ""
    return row


def scan_runs(output_dir: Path) -> List[Dict[str, Any]]:
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return []
    by_id: Dict[str, Tuple[int, float, Dict[str, Any]]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        run_id = str(payload.get("run_id") or path.stem)
        score = (
            1 if str(payload.get("status")) == "ok" else 0,
            path.stat().st_mtime,
        )
        old = by_id.get(run_id)
        if old is None or score > (old[0], old[1]):
            by_id[run_id] = (score[0], score[1], _manifest_row(payload))
    rows = [x[2] for x in by_id.values()]
    rows.sort(key=lambda r: (
        str(r.get("algorithm", "")), float(r.get("battery_wh") or 0),
        float(r.get("payload_kg") or 0), int(r.get("seed") or 0),
    ))
    return rows


def rebuild_manifest(output_dir: Path) -> Path:
    path = output_dir / "runs.csv"
    _atomic_csv(path, MANIFEST_FIELDS, scan_runs(output_dir))
    return path


def completed_ids(output_dir: Path) -> set:
    return {
        str(row["run_id"]) for row in scan_runs(output_dir)
        if str(row.get("status")) == "ok"
    }


def _preview(tasks, resolved, show=8):
    print("\n===== DRONE SENSITIVITY TASK PREVIEW (no solver is created) =====")
    print("dataset      : %s" % resolved["dataset_name"])
    print("algorithms   : %s" % ", ".join(resolved["algorithms"]))
    print("seeds        : %s" % resolved["seeds"])
    print("budget (s)   : %g" % resolved["time_budget_seconds"])
    print("combinations : %d" % len(resolved["combinations"]))
    print("  " + ", ".join("b%g/p%g" % c for c in resolved["combinations"]))
    print("total tasks  : %d" % len(tasks))
    print("output dir   : %s" % resolved["output_dir"])
    print("\nPer-algorithm counts:")
    for alg in resolved["algorithms"]:
        print("  %-5s %d" % (alg, sum(1 for t in tasks if t["algorithm"] == alg)))
    if show and tasks:
        print("\nTask sample:")
        for t in tasks[:show]:
            print("  " + t["run_id"])
        if len(tasks) > show:
            print("  ... (%d more)" % (len(tasks) - show))
    print("\nNo optimisation was executed.  Add --execute to start formal runs.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drone battery/payload sensitivity runner. Without --execute only previews tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--algorithms", nargs="+", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--battery-levels", nargs="+", type=float, default=None)
    p.add_argument("--payload-levels", nargs="+", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--show-tasks", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit("config not found: %s" % config_path)
    config = load_config(config_path)
    try:
        tasks, resolved = build_tasks(
            config, algorithms=args.algorithms, seeds=args.seeds,
            dataset=args.dataset, battery_levels=args.battery_levels,
            payload_levels=args.payload_levels, output_dir=args.output_dir,
            time_budget=args.time_budget,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    if not args.execute:
        _preview(tasks, resolved, args.show_tasks)
        return 0

    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    prior_rows = scan_runs(output_dir)
    other_datasets = sorted({
        str(row.get("dataset", "")) for row in prior_rows
        if str(row.get("dataset", "")) not in ("", resolved["dataset_name"])
    })
    if other_datasets:
        raise SystemExit(
            "output directory already contains runs for another dataset: %s. "
            "Use a separate --output-dir (the configured M-n151-k12 directory "
            "is results/)." % ", ".join(other_datasets)
        )
    existing = {str(row["run_id"]) for row in prior_rows
                if str(row.get("status")) == "ok"}
    pending = [t for t in tasks if args.overwrite or t["run_id"] not in existing]
    skipped = len(tasks) - len(pending)
    _atomic_json(output_dir / "experiment_used.json", {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "resolved": {k: v for k, v in resolved.items() if k != "output_dir"},
        "argv": list(sys.argv),
    })

    print("\n" + "=" * 72)
    print("EXECUTING DRONE SENSITIVITY EXPERIMENT")
    print("=" * 72)
    print("tasks=%d pending=%d skipped=%d" % (len(tasks), len(pending), skipped))
    print("output=%s" % output_dir)
    sys.stdout.flush()

    started = time.perf_counter()
    errors = 0
    for index, task in enumerate(pending, 1):
        try:
            row = run_one(task)
            with open(output_dir / "runs" / (task["run_id"] + ".json"),
                      "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # defensive: one failed run must not kill the batch
            payload = {
                "run_id": task["run_id"], "status": "error",
                "algorithm": task["algorithm"], "dataset": task["dataset_name"],
                "seed": int(task["seed"]), "battery_wh": float(task["battery_wh"]),
                "payload_kg": float(task["payload_kg"]),
                "time_budget_seconds": float(task["time_budget_seconds"]),
                "error": str(exc), "feasible": False,
            }
            _atomic_json(output_dir / "runs" / (task["run_id"] + ".json"), payload)
            row = _manifest_row(payload)
        if str(row.get("status")) != "ok":
            errors += 1
            if args.progress and not args.quiet:
                print("[%d/%d] ERROR %s: %s" % (
                    index, len(pending), row.get("run_id"), row.get("error")
                ), flush=True)
        elif args.progress and not args.quiet:
            print("[%d/%d] ok %s makespan=%s rt=%.2fs" % (
                index, len(pending), row.get("run_id"),
                row.get("best_makespan_s"), row.get("actual_runtime_seconds") or 0.0,
            ), flush=True)
        with open(output_dir / "runs.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            if (output_dir / "runs.csv").stat().st_size == 0:
                writer.writeheader()
            writer.writerow(_manifest_row(payload))
            f.flush()
            os.fsync(f.fileno())

    manifest = rebuild_manifest(output_dir)
    elapsed = time.perf_counter() - started
    print("\nFinished: %d run(s) in %.1fs; errors=%d" % (len(pending), elapsed, errors))
    print("Manifest: %s" % manifest)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
