"""Configuration and task-matrix tests for the sensitivity experiment.

No solver is constructed; this only verifies the current algorithm set
(ACO/GA/ALNS/PSO/VND), the ACO pinning, and the 15-combination / 750-task matrix.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_RESEARCH_ROOT = _PROJECT.parent
for _path in (str(_RESEARCH_ROOT), str(_PROJECT), str(_PROJECT / "adapters")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_sensitivity

CONFIG = _PROJECT / "config" / "sensitivity_config.json"
EXPECTED = ["ACO", "GA", "ALNS", "PSO", "VND"]


class TestSensitivityConfig(unittest.TestCase):
    def test_current_algorithms_and_task_matrix(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["algorithms"], EXPECTED)
        self.assertNotIn("VNS", config["algorithms"])
        self.assertNotIn("VNS", config["algorithm_parameters"])
        self.assertIn("PSO", config["algorithm_parameters"])

        pso = config["algorithm_parameters"]["PSO"]
        for key in ("swarm_size", "chi", "phi1", "phi2",
                    "velocity_limit", "seed_fraction",
                    "perturbation_sigma", "cache_size"):
            self.assertIn(key, pso)

        combos = run_sensitivity.parameter_combinations(config)
        self.assertEqual(len(combos), 15)
        self.assertEqual(config["time_budget_seconds"], 50.0)
        self.assertEqual(config["dataset_name"], "M-n151-k12")

        aco = config["algorithm_parameters"]["ACO"]
        self.assertEqual(aco["service_policy_mode"], "fixed")
        self.assertEqual(aco["adaptive_service_fraction"], 1.0)
        self.assertEqual(aco["service_proxy_strength"], 1.0)

        tasks, resolved = run_sensitivity.build_tasks(config)
        self.assertEqual(len(tasks), 750)
        self.assertEqual(resolved["expected_tasks"], 750)
        self.assertEqual(len({task["run_id"] for task in tasks}), 750)
        per_alg = {alg: sum(1 for t in tasks if t["algorithm"] == alg)
                   for alg in EXPECTED}
        self.assertEqual(set(per_alg.values()), {150})


if __name__ == "__main__":
    unittest.main()
