"""Five-algorithm convergence experiment runner.

SAFETY GATE
-----------
Without ``--execute`` this program only previews the 350-task matrix.  Formal
time-budget runs (100 s in the current config) require the explicit
``--execute`` flag and are not started by preview mode.
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
from adapters.history import FIELDS as HISTORY_FIELDS, HistoryRecorder

DEFAULT_CONFIG = _HERE / "config" / "convergence_config.json"
ALGORITHMS = ("ACO", "GA", "ALNS", "PSO", "VND")
MANIFEST_FIELDS = [
    "run_id", "status", "algorithm", "dataset", "seed", "time_budget_seconds",
    "actual_runtime_seconds", "best_makespan_s", "feasible", "failure_reason",
    "initial_makespan_s", "iterations_or_generations", "evaluations",
    "battery_wh", "payload_kg", "history_file", "history_events",
    "first_feasible_time_s", "last_improvement_time_s", "run_file",
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


def _dataset_customer_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return len(raw["locations"])


def build_tasks(
    config: Dict[str, Any],
    algorithms: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    time_budget: Optional[float] = None,
    output_dir: Optional[str] = None,
    history_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    algs = [str(x).upper() for x in (algorithms or config["algorithms"])]
    for alg in algs:
        if alg not in ALGORITHMS:
            raise ValueError("unknown algorithm %r" % alg)
    seed_values = [int(x) for x in (seeds or config["seeds"])]
    budget = float(time_budget if time_budget is not None else config["time_budget_seconds"])
    selected_names = {str(x) for x in datasets} if datasets else None

    specs = []
    for entry in config["datasets"]:
        name = str(entry["name"])
        if selected_names is not None and name not in selected_names:
            continue
        path = _resolve_project_path(entry["path"])
        if not path.is_file():
            raise ValueError("required dataset is missing: %s" % path)
        actual = _dataset_customer_count(path)
        expected = int(entry.get("expected_customers", actual))
        if actual != expected:
            raise ValueError(
                "dataset %s has %d customers, config expects %d"
                % (name, actual, expected)
            )
        specs.append({
            "name": name, "path": str(path), "family": entry.get("family", ""),
            "expected_customers": actual,
        })
    if not specs:
        raise ValueError("no datasets selected")
    if selected_names is not None:
        missing = selected_names - {s["name"] for s in specs}
        if missing:
            raise ValueError("unknown dataset names: %s" % ", ".join(sorted(missing)))

    out_dir = _resolve_project_path(output_dir or config.get("output_dir", "results"))
    hist_dir = _resolve_project_path(history_dir or config.get("history_dir", "history"))
    physics = config.get("physics", {})
    battery = float(physics.get("battery_wh", 200.0))
    payload = float(physics.get("payload_kg", 3.0))
    params_by_alg = config.get("algorithm_parameters", {})

    tasks = []
    for spec in specs:
        for alg in algs:
            for seed in seed_values:
                run_id = "%s__%s__s%d__b%s" % (
                    alg, spec["name"], seed, _tag(budget)
                )
                tasks.append({
                    "run_id": run_id,
                    "algorithm": alg,
                    "dataset_path": spec["path"],
                    "dataset_name": spec["name"],
                    "seed": seed,
                    "time_budget_seconds": budget,
                    "battery_wh": battery,
                    "payload_kg": payload,
                    "params": dict(params_by_alg.get(alg, {})),
                    "output_dir": str(out_dir),
                    "history_dir": str(hist_dir),
                })
    resolved = {
        "algorithms": algs,
        "seeds": seed_values,
        "time_budget_seconds": budget,
        "datasets": specs,
        "output_dir": str(out_dir),
        "history_dir": str(hist_dir),
        "battery_wh": battery,
        "payload_kg": payload,
        "expected_tasks": len(specs) * len(algs) * len(seed_values),
    }
    return tasks, resolved


def _manifest_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    row = {key: payload.get(key, "") for key in MANIFEST_FIELDS}
    if row.get("best_makespan_s") is None:
        row["best_makespan_s"] = ""
    return row


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(task["output_dir"])
    history_dir = Path(task["history_dir"])
    run_path = output_dir / "runs" / (task["run_id"] + ".json")
    hist_path = history_dir / (task["run_id"] + ".csv")
    recorder = HistoryRecorder(
        algorithm=task["algorithm"],
        dataset=task["dataset_name"],
        seed=int(task["seed"]),
        time_budget_seconds=float(task["time_budget_seconds"]),
    )
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
        result = run_algorithm(spec, recorder=recorder)
        events = recorder.rows()
        _atomic_csv(hist_path, HISTORY_FIELDS, events)
        result.update({
            "run_id": task["run_id"],
            "status": "ok",
            "history_file": str(hist_path),
            "history_events": len(events),
            "first_feasible_time_s": (
                float(events[0]["elapsed_time_seconds"]) if events else None
            ),
            "last_improvement_time_s": (
                max(
                    (float(e["elapsed_time_seconds"]) for e in events
                     if e["event_type"] == "improvement"),
                    default=None,
                ) if events else None
            ),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_file": str(run_path.relative_to(output_dir)).replace("\\", "/"),
        })
    except Exception as exc:  # noqa: BLE001
        _atomic_csv(hist_path, HISTORY_FIELDS, [])
        result = {
            "run_id": task["run_id"], "status": "error",
            "algorithm": task["algorithm"], "dataset": task["dataset_name"],
            "seed": int(task["seed"]),
            "time_budget_seconds": float(task["time_budget_seconds"]),
            "battery_wh": float(task["battery_wh"]),
            "payload_kg": float(task["payload_kg"]),
            "feasible": False, "best_makespan_s": None,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "history_file": str(hist_path),
            "history_events": 0,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_file": str(run_path.relative_to(output_dir)).replace("\\", "/"),
        }
    _atomic_json(run_path, result)
    return _manifest_row(result)


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
        str(r.get("dataset", "")), str(r.get("algorithm", "")),
        int(r.get("seed") or 0),
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
    print("\n===== CONVERGENCE TASK PREVIEW (no solver is created) =====")
    print("datasets     : %s" % ", ".join(s["name"] for s in resolved["datasets"]))
    print("algorithms   : %s" % ", ".join(resolved["algorithms"]))
    print("seeds        : %s" % resolved["seeds"])
    print("budget (s)   : %g" % resolved["time_budget_seconds"])
    print("physics      : %g Wh, %g kg" % (
        resolved["battery_wh"], resolved["payload_kg"]))
    print("total tasks  : %d" % len(tasks))
    print("results dir  : %s" % resolved["output_dir"])
    print("history dir  : %s" % resolved["history_dir"])
    print("\nPer-dataset counts:")
    for spec in resolved["datasets"]:
        print("  %-12s %d" % (spec["name"], sum(1 for t in tasks if t["dataset_name"] == spec["name"])))
    if show and tasks:
        print("\nTask sample:")
        for t in tasks[:show]:
            print("  " + t["run_id"])
        if len(tasks) > show:
            print("  ... (%d more)" % (len(tasks) - show))
    print("\nNo optimisation was executed.  Add --execute to start formal runs.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Five-algorithm convergence runner. Without --execute only previews tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--algorithms", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--history-dir", default=None)
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
            config, algorithms=args.algorithms, datasets=args.datasets,
            seeds=args.seeds, time_budget=args.time_budget,
            output_dir=args.output_dir, history_dir=args.history_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    if not args.execute:
        _preview(tasks, resolved, args.show_tasks)
        return 0

    output_dir = Path(resolved["output_dir"])
    history_dir = Path(resolved["history_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "experiment_used.json", {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "resolved": {k: v for k, v in resolved.items()
                     if k not in ("output_dir", "history_dir")},
        "argv": list(sys.argv),
    })
    existing = completed_ids(output_dir)
    pending = [t for t in tasks if args.overwrite or t["run_id"] not in existing]
    skipped = len(tasks) - len(pending)

    print("\n" + "=" * 72)
    print("EXECUTING FIVE-ALGORITHM CONVERGENCE EXPERIMENT")
    print("=" * 72)
    print("tasks=%d pending=%d skipped=%d" % (len(tasks), len(pending), skipped))
    print("results=%s" % output_dir)
    print("history=%s" % history_dir)
    sys.stdout.flush()

    started = time.perf_counter()
    errors = 0
    for index, task in enumerate(pending, 1):
        try:
            row = run_one(task)
        except Exception as exc:  # defensive
            errors += 1
            if args.progress and not args.quiet:
                print("[%d/%d] ERROR %s: %s" % (
                    index, len(pending), task["run_id"], exc), flush=True)
            continue
        if str(row.get("status")) != "ok":
            errors += 1
            if args.progress and not args.quiet:
                print("[%d/%d] ERROR %s: %s" % (
                    index, len(pending), row.get("run_id"), row.get("error")
                ), flush=True)
        elif args.progress and not args.quiet:
            print("[%d/%d] ok %s makespan=%s hist=%s rt=%.2fs" % (
                index, len(pending), row.get("run_id"),
                row.get("best_makespan_s"), row.get("history_events"),
                row.get("actual_runtime_seconds") or 0.0,
            ), flush=True)
        with open(output_dir / "runs.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            if (output_dir / "runs.csv").stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    manifest = rebuild_manifest(output_dir)
    elapsed = time.perf_counter() - started
    print("\nFinished: %d run(s) in %.1fs; errors=%d" % (len(pending), elapsed, errors))
    print("Manifest: %s" % manifest)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
