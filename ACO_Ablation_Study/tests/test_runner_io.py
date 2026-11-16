"""Resume/conflict logic, dry-run safety, and output-path isolation tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac
import run_state


def write_minimal_run(
    results_root: str,
    variant: str,
    dataset: str,
    seed: int,
    budget: float,
    *,
    code_fingerprint: str = "test-code",
    dataset_sha: str = "test-dataset-sha",
    makespan: float = 123.0,
    feasible: bool = True,
    status: str = "completed",
    config_fingerprint=None,
) -> str:
    run_dir = ac.run_directory(results_root, variant, budget, dataset, seed)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "status": status,
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "requested_time_budget_seconds": float(budget),
        "variant_config_fingerprint": ac.variant_fingerprint(variant),
        "run_config_fingerprint": (
            config_fingerprint
            if config_fingerprint is not None
            else ac.run_config_fingerprint(variant, seed, budget)
        ),
        "code_fingerprint": code_fingerprint,
        "dataset_sha256": dataset_sha,
        "feasible": feasible,
        "makespan_s": makespan,
    }
    with open(os.path.join(run_dir, ac.SOLUTION_FILENAME), "w", encoding="utf-8") as f:
        f.write("synthetic\n")
    with open(os.path.join(run_dir, ac.HISTORY_FILENAME), "w", encoding="utf-8") as f:
        f.write("iteration\n")
    with open(os.path.join(run_dir, ac.RUN_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return run_dir


class TestRunInspection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.run_dir = write_minimal_run(
            self.root, "full", "D1", 13, 50.0,
            code_fingerprint="code-A", dataset_sha="sha-A")

    def tearDown(self):
        self.tmp.cleanup()

    def _inspect(self, **kwargs):
        return run_state.inspect_run(
            self.run_dir,
            expected_run_fingerprint=kwargs.pop(
                "run_fp", ac.run_config_fingerprint("full", 13, 50.0)),
            expected_code_fingerprint=kwargs.pop("code_fp", "code-A"),
            expected_dataset_sha256=kwargs.pop("dataset_sha", "sha-A"),
        )

    def test_complete(self):
        self.assertEqual(self._inspect()["state"], run_state.COMPLETE)

    def test_missing_directory(self):
        state = run_state.inspect_run(os.path.join(self.root, "nope"))
        self.assertEqual(state["state"], run_state.MISSING)

    def test_partial_status(self):
        write_minimal_run(self.root, "full", "D1", 13, 50.0,
                          code_fingerprint="code-A", dataset_sha="sha-A",
                          status="running")
        self.assertEqual(self._inspect()["state"], run_state.PARTIAL)

    def test_config_mismatch(self):
        self.assertEqual(self._inspect(run_fp="other")["state"],
                         run_state.CONFIG_MISMATCH)

    def test_code_mismatch(self):
        self.assertEqual(self._inspect(code_fp="other")["state"],
                         run_state.CODE_MISMATCH)

    def test_dataset_mismatch(self):
        self.assertEqual(self._inspect(dataset_sha="other")["state"],
                         run_state.DATASET_MISMATCH)

    def test_missing_auxiliary_file_is_partial(self):
        os.remove(os.path.join(self.run_dir, ac.SOLUTION_FILENAME))
        self.assertEqual(self._inspect()["state"], run_state.PARTIAL)

    def test_infeasible_is_invalid(self):
        write_minimal_run(self.root, "full", "D1", 13, 50.0,
                          code_fingerprint="code-A", dataset_sha="sha-A",
                          feasible=False)
        self.assertEqual(self._inspect()["state"], run_state.INVALID)

    def test_dataset_hash_changes_with_content(self):
        path = os.path.join(self.root, "data.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"a": 1}')
        first = ac.file_sha256(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
        self.assertNotEqual(first, ac.file_sha256(path))


class TestBatchSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmp.name, "data")
        os.makedirs(self.data_dir)
        # Deliberately invalid JSON: --dry-run must never parse or solve it.
        self.dataset = os.path.join(self.data_dir, "dummy.json")
        with open(self.dataset, "w", encoding="utf-8") as f:
            f.write("this is not valid json")
        self.results = os.path.join(self.tmp.name, "results")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra):
        cmd = [
            sys.executable,
            os.path.join(STUDY_DIR, "run_all_ablations.py"),
            "--data-dir", self.data_dir,
            "--variants", "no_edge", "full",
            "--budgets", "2",
            "--seeds", "13",
            "--results-root", self.results,
            *extra,
        ]
        env = dict(os.environ)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        return subprocess.run(cmd, cwd=STUDY_DIR, capture_output=True,
                              text=True, env=env)

    def test_dry_run_creates_nothing_and_does_not_parse_json(self):
        proc = self._run("--dry-run")
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertIn("NO SOLVE EXECUTED", output)
        self.assertIn("would_launch=2", output)
        self.assertFalse(os.path.exists(self.results))

    def test_missing_confirm_token_refuses_to_start(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("formal batch not started", proc.stderr)
        self.assertFalse(os.path.exists(self.results))

    def test_wrong_confirm_token_refuses_to_start(self):
        proc = self._run("--confirm", "yes")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("wrong --confirm token", proc.stderr)
        self.assertFalse(os.path.exists(self.results))

    def test_check_only_reports_incomplete_without_creating_results(self):
        proc = self._run("--check-only")
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, msg=output)
        self.assertIn("INCOMPLETE", output)
        self.assertFalse(os.path.exists(self.results))

    def test_check_only_accepts_one_complete_audited_run(self):
        sha = ac.file_sha256(self.dataset)
        write_minimal_run(
            self.results, "no_edge", "dummy", 13, 2.0,
            code_fingerprint=ac.code_fingerprint(), dataset_sha=sha)
        write_minimal_run(
            self.results, "full", "dummy", 13, 2.0,
            code_fingerprint=ac.code_fingerprint(), dataset_sha=sha)
        proc = self._run("--check-only")
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertIn("expected grid is complete", output)

    def test_single_runner_skips_a_completed_run(self):
        sha = ac.file_sha256(self.dataset)
        write_minimal_run(
            self.results, "no_edge", "dummy", 13, 2.0,
            code_fingerprint=ac.code_fingerprint(), dataset_sha=sha)
        cmd = [
            sys.executable,
            os.path.join(STUDY_DIR, "run_ablation.py"),
            self.dataset,
            "--variant", "no_edge",
            "--seed", "13",
            "--time-limit", "2",
            "--results-root", self.results,
            "--quiet",
        ]
        env = dict(os.environ)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.run(cmd, cwd=STUDY_DIR, capture_output=True,
                              text=True, env=env)
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 3, msg=output)
        self.assertIn("[skip]", output)


if __name__ == "__main__":
    unittest.main()
