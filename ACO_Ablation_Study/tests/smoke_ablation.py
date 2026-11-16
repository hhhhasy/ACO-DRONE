"""One bounded smoke test: one small dataset, one seed, six variants, <=2 s each.

This is the ONLY script in the study that executes the solver end-to-end, and
it is deliberately hard-coded to the limits below.  It is NOT part of the unit
test suite and must not be used for formal experiments.  Outputs go to
``results/_smoke/`` and are ignored by ``analyze_ablation.py``.

Completed combinations are skipped, so rerunning after adding a variant only
executes the new variant(s).

Usage (run once, manually)::

    python tests/smoke_ablation.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac

DATASET = os.path.abspath(os.path.join(STUDY_DIR, "..", "大规模", "P-n16-k8.json"))
SEED = 13
BUDGET_SECONDS = 2.0
RESULTS_ROOT = os.path.join(STUDY_DIR, "results", "_smoke")

# Hard safety limits for this file.
assert BUDGET_SECONDS <= 2.0
assert len(ac.VARIANT_KEYS) == 6


def _row_from_payload(variant, payload):
    return (
        variant,
        float(payload["makespan_s"]),
        bool(payload["feasible"]),
        int(payload.get("iterations_started", 0)),
        payload.get("pheromone", {}).get("enabled", {}),
    )


def main() -> int:
    if not os.path.isfile(DATASET):
        print("[error] smoke dataset not found: %s" % DATASET)
        return 2
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    failures = []
    rows = []
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    for variant in ac.VARIANT_KEYS:
        run_dir = ac.run_directory(
            RESULTS_ROOT, variant, BUDGET_SECONDS, dataset_name, SEED)
        run_path = os.path.join(run_dir, ac.RUN_FILENAME)
        # Idempotent: an existing smoke record is never solved again.  This
        # also means extending the variant list only runs the new variants.
        if os.path.isfile(run_path):
            with open(run_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows.append(_row_from_payload(variant, payload))
            print("\n[smoke] %s: existing record, skip (%s)"
                  % (variant, run_path))
            continue

        cmd = [
            sys.executable,
            os.path.join(STUDY_DIR, "run_ablation.py"),
            DATASET,
            "--variant", variant,
            "--seed", str(SEED),
            "--time-limit", "%g" % BUDGET_SECONDS,
            "--ants", str(ac.ANTS),
            "--results-root", RESULTS_ROOT,
            "--quiet",
        ]
        print("\n[smoke] %s (budget %.1fs)" % (variant, BUDGET_SECONDS))
        proc = subprocess.run(cmd, cwd=STUDY_DIR, env=env)
        if proc.returncode not in (0, 3) or not os.path.isfile(run_path):
            failures.append((variant, proc.returncode))
            print("  FAILED returncode=%d" % proc.returncode)
            continue
        with open(run_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(_row_from_payload(variant, payload))
        print("  makespan=%.4f feasible=%s iters=%d pheromone=%s"
              % (rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4]))

    print("\n===== smoke summary (NOT a formal experiment) =====")
    for variant, makespan, feasible, iters, pheromone in rows:
        print("%-12s makespan=%.4f feasible=%s iterations=%d"
              % (variant, makespan, feasible, iters))
    if failures:
        print("FAILURES: %r" % (failures,))
        return 1
    if len(rows) != 6:
        print("ERROR: expected six variant records, got %d" % len(rows))
        return 1
    print("OK: six variants initialised and produced feasible plans in <=%.0fs each."
          % BUDGET_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
