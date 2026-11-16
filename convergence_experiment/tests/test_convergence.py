"""Unit, history and short smoke tests for the convergence project.

Run from the project directory:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_RESEARCH_ROOT = _PROJECT.parent
for _p in (str(_RESEARCH_ROOT), str(_PROJECT), str(_PROJECT / "adapters")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.algorithms import RunSpec, run_algorithm
from adapters.history import FIELDS as HISTORY_FIELDS, HistoryRecorder
import analyze_convergence
import run_convergence

DATASET = _RESEARCH_ROOT / "大规模" / "A-n32-k5.json"
HAVE_DATASET = DATASET.is_file()
CONFIG = _PROJECT / "config" / "convergence_config.json"


class TestConfigAndTasks(unittest.TestCase):
    def test_seven_datasets_five_algorithms_350_tasks(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        tasks, resolved = run_convergence.build_tasks(config)
        self.assertEqual(len(config["datasets"]), 7)
        self.assertEqual(len(config["algorithms"]), 5)
        self.assertEqual(len(config["seeds"]), 10)
        self.assertEqual(len(tasks), 350)
        self.assertEqual(resolved["expected_tasks"], 350)
        self.assertEqual(len({t["run_id"] for t in tasks}), 350)
        self.assertEqual(config["time_budget_seconds"], 100.0)
        self.assertEqual(config["physics"]["battery_wh"], 200.0)
        self.assertEqual(config["physics"]["payload_kg"], 3.0)
        aco = config["algorithm_parameters"]["ACO"]
        self.assertEqual(aco["service_policy_mode"], "fixed")
        self.assertEqual(aco["adaptive_service_fraction"], 1.0)
        self.assertEqual(aco["service_proxy_strength"], 1.0)
        aco_tasks = [task for task in tasks if task["algorithm"] == "ACO"]
        self.assertEqual(len(aco_tasks), 70)
        for task in aco_tasks:
            self.assertEqual(task["params"]["service_policy_mode"], "fixed")
            self.assertEqual(task["params"]["adaptive_service_fraction"], 1.0)
            self.assertEqual(task["params"]["service_proxy_strength"], 1.0)
        for t in tasks:
            self.assertEqual(t["battery_wh"], 200.0)
            self.assertEqual(t["payload_kg"], 3.0)

    def test_preview_creates_no_output(self):
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_test_data_", dir=str(_HERE)))
        try:
            out = tmp / "results"
            hist = tmp / "history"
            code = run_convergence.main([
                "--datasets", "P-n40-k5", "--algorithms", "ACO",
                "--seeds", "13", "--output-dir", str(out),
                "--history-dir", str(hist), "--dry-run", "--show-tasks", "1",
            ])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            self.assertFalse(hist.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dataset_customer_counts(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        tasks, resolved = run_convergence.build_tasks(config)
        expected = {d["name"]: d["expected_customers"] for d in config["datasets"]}
        for d in resolved["datasets"]:
            self.assertEqual(d["expected_customers"], expected[d["name"]])


class TestHistoryRecorder(unittest.TestCase):
    def test_monotone_and_first_feasible(self):
        rec = HistoryRecorder("ACO", "A-n32-k5", 13, 60.0)
        rec.initial(100.0, 0, 0)
        rec({"best_makespan_s": 90.0, "iteration_or_generation": 5,
             "evaluations": 10, "event_type": "improvement"})
        rec({"best_makespan_s": 95.0, "iteration_or_generation": 6,
             "evaluations": 11, "event_type": "improvement"})  # ignored
        rec({"best_makespan_s": 90.0, "iteration_or_generation": 7,
             "evaluations": 12, "event_type": "improvement"})  # ignored
        rec.final(90.0, 8, 13)
        events = rec.rows()
        self.assertEqual([e["event_type"] for e in events],
                         ["initial", "improvement", "final"])
        values = [e["best_makespan_s"] for e in events]
        self.assertEqual(values, [100.0, 90.0, 90.0])
        times = [e["elapsed_time_seconds"] for e in events]
        self.assertTrue(all(b >= a for a, b in zip(times, times[1:])))
        self.assertTrue(all(v < 60.0 for v in times))

    def test_infeasible_values_are_not_recorded(self):
        rec = HistoryRecorder("GA", "A-n32-k5", 13, 60.0)
        rec({"best_makespan_s": float("inf"), "iteration_or_generation": 1,
             "evaluations": 1, "event_type": "improvement"})
        rec({"best_makespan_s": None, "iteration_or_generation": 2,
             "evaluations": 2, "event_type": "improvement"})
        self.assertEqual(rec.rows(), [])

    def test_forward_hold_and_missing_prefix(self):
        history = pd.DataFrame([
            {"elapsed_time_seconds": 5.0, "best_makespan_s": 100.0,
             "event_type": "initial"},
            {"elapsed_time_seconds": 10.0, "best_makespan_s": 90.0,
             "event_type": "improvement"},
            {"elapsed_time_seconds": 20.0, "best_makespan_s": 90.0,
             "event_type": "final"},
        ])
        grid = np.arange(0.0, 20.5, 2.5)
        curve = analyze_convergence.align_curve(history, grid)
        self.assertTrue(np.isnan(curve[0]))
        self.assertTrue(np.isnan(curve[1]))
        self.assertEqual(float(curve[2]), 100.0)  # t=5
        self.assertEqual(float(curve[4]), 90.0)   # t=10
        self.assertEqual(float(curve[-1]), 90.0)  # t=20 held

    def test_history_fields_exact(self):
        rec = HistoryRecorder("VND", "P-n40-k5", 21, 60.0)
        rec.initial(500.0, 0, 1)
        row = rec.rows()[0]
        self.assertEqual(set(row.keys()), set(HISTORY_FIELDS))


class TestAnalysisWithSyntheticData(unittest.TestCase):
    def _make_fixture(self, root: Path):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        results = root / "results"
        hist = root / "history"
        (results / "runs").mkdir(parents=True)
        hist.mkdir(parents=True)
        times = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 100.0]
        for di, ds in enumerate(config["datasets"]):
            name = ds["name"]
            base = 1000.0 + 50.0 * di
            for ai, alg in enumerate(config["algorithms"]):
                for si, seed in enumerate([13, 21, 42]):
                    final = base + 10.0 * ai + 2.0 * si
                    rid = "%s__%s__s%d__b100" % (alg, name, seed)
                    hpath = hist / (rid + ".csv")
                    rows = []
                    for j, t in enumerate(times):
                        value = final + (100.0 - 15.0 * j) + 5.0 * (len(times) - j)
                        if j == len(times) - 1:
                            value = final
                        rows.append({
                            "algorithm": alg, "dataset": name, "seed": seed,
                            "time_budget_seconds": 100.0,
                            "elapsed_time_seconds": t,
                            "best_makespan_s": max(final, value),
                            "iteration_or_generation": j * 10,
                            "evaluations": j * 100,
                            "event_type": "initial" if j == 0 else (
                                "final" if j == len(times) - 1 else "improvement"
                            ),
                        })
                    pd.DataFrame(rows, columns=HISTORY_FIELDS).to_csv(
                        hpath, index=False, encoding="utf-8-sig"
                    )
                    payload = {
                        "run_id": rid, "status": "ok", "algorithm": alg,
                        "dataset": name, "seed": seed,
                        "time_budget_seconds": 100.0,
                        "battery_wh": 200.0, "payload_kg": 3.0,
                        "actual_runtime_seconds": 100.1,
                        "best_makespan_s": final, "feasible": True,
                        "history_file": str(hpath),
                        "history_events": len(rows),
                    }
                    (results / "runs" / (rid + ".json")).write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

    def test_full_analysis_and_plots(self):
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_test_data_", dir=str(_HERE)))
        try:
            self._make_fixture(tmp)
            runs, curves, final_summary, targets, missing = analyze_convergence.run_analysis(
                tmp / "results", tmp / "history", tmp / "analysis", tmp / "figures",
                CONFIG, make_plots=True, dpi=50, quiet=True,
            )
            self.assertEqual(len(runs), 105)
            self.assertFalse(curves.empty)
            self.assertEqual(len(final_summary), 35)  # 7 datasets x 5 algorithms
            self.assertTrue((tmp / "analysis" / "convergence_final_summary.csv").is_file())
            self.assertTrue((tmp / "analysis" / "convergence_report.md").is_file())
            self.assertTrue((tmp / "analysis" / "convergence_tables.tex").is_file())
            # 7 per-dataset + fig06 + fig07 + fig08 + fig09 = 11 figures.
            self.assertGreaterEqual(len(list((tmp / "figures").glob("fig*.png"))), 9)
            self.assertEqual(len(list((tmp / "figures").glob("fig*.png"))),
                             len(list((tmp / "figures").glob("fig*.pdf"))))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_results_report_no_data(self):
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_test_data_", dir=str(_HERE)))
        try:
            (tmp / "results").mkdir()
            (tmp / "history").mkdir()
            runs, curves, final_summary, targets, missing = analyze_convergence.run_analysis(
                tmp / "results", tmp / "history", tmp / "analysis", tmp / "figures",
                CONFIG, make_plots=True, dpi=50, quiet=True,
            )
            self.assertTrue(runs.empty)
            report = (tmp / "analysis" / "convergence_report.md").read_text(encoding="utf-8")
            self.assertIn("尚无正式结果", report)
            self.assertEqual(list((tmp / "figures").glob("fig*.png")), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_DATASET, "A-n32-k5.json not found")
class TestShortSmoke(unittest.TestCase):
    def test_all_five_adapters_one_second_with_history(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        params = config["algorithm_parameters"]
        for alg in config["algorithms"]:
            with self.subTest(algorithm=alg):
                rec = HistoryRecorder(alg, "A-n32-k5", 13, 1.0)
                spec = RunSpec(
                    alg, str(DATASET), "A-n32-k5", 13, 1.0, 200.0, 3.0,
                    dict(params[alg]),
                )
                result = run_algorithm(spec, recorder=rec)
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["feasible"])
                events = rec.rows()
                self.assertGreater(len(events), 0)
                self.assertEqual(events[0]["event_type"], "initial")
                values = [e["best_makespan_s"] for e in events]
                self.assertTrue(all(b <= a + 1e-9 for a, b in zip(values, values[1:])))
                times = [e["elapsed_time_seconds"] for e in events]
                self.assertTrue(all(b >= a for a, b in zip(times, times[1:])))

    def test_convergence_runner_one_second_and_resume(self):
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_test_data_", dir=str(_HERE)))
        try:
            results = tmp / "results"
            hist = tmp / "history"
            args = [
                "--datasets", "P-n40-k5", "--algorithms", "ACO",
                "--seeds", "13", "--time-budget", "1",
                "--output-dir", str(results), "--history-dir", str(hist),
                "--execute", "--quiet",
            ]
            self.assertEqual(run_convergence.main(args), 0)
            files = list((results / "runs").glob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["feasible"])
            hp = Path(payload["history_file"])
            self.assertTrue(hp.is_file())
            df = pd.read_csv(hp, encoding="utf-8-sig")
            self.assertFalse(df.empty)
            self.assertEqual(set(df.columns), set(HISTORY_FIELDS))
            self.assertGreater(len(list(hist.glob("*.csv"))), 0)
            # Resume must skip the completed run.
            self.assertEqual(run_convergence.main(args), 0)
            self.assertEqual(len(list((results / "runs").glob("*.json"))), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
