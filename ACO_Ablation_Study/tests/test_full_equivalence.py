"""Prove that ``full`` reproduces the official current ACO solver exactly.

The official ``aco_solver.py`` is loaded from the parent project under an alias
and both solvers are run on the same tiny deterministic instance with the same
seed, ant count and iteration count.  Because the algorithm is pure-Python
deterministic at a fixed seed, the constructive decisions and makespan must be
bit-for-bit identical.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac
from aco_solver import AntColonySolver
from tests.synthetic import make_problem

ORIGINAL_PATH = os.path.abspath(os.path.join(STUDY_DIR, "..", "aco_solver.py"))


def load_original_solver():
    if not os.path.isfile(ORIGINAL_PATH):
        return None
    spec = importlib.util.spec_from_file_location(
        "official_aco_solver_for_equivalence_test", ORIGINAL_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__], so
    # the alias must be registered before execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestFullEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_original_solver()

    def test_original_file_available(self):
        if self.original is None:
            self.skipTest("official aco_solver.py not found next to the study folder")

    def test_config_defaults_identical(self):
        if self.original is None:
            self.skipTest("official aco_solver.py not found")
        official = vars(self.original.ACOConfig(
            ants=ac.ANTS, iterations=0, seed=13,
            time_limit_seconds=0.0, verbose_every=0))
        ours = vars(ac.build_config("full", seed=13, time_limit_seconds=0.0))
        for key, value in official.items():
            self.assertIn(key, ours)
            self.assertEqual(ours[key], value, msg=key)

    def test_same_seed_same_decisions_and_makespan(self):
        if self.original is None:
            self.skipTest("official aco_solver.py not found")
        problem = make_problem(n=7, k=2, d=2, seed=3)
        official_cfg = self.original.ACOConfig(
            ants=3, iterations=3, seed=42, time_limit_seconds=0.0,
            verbose_every=0,
        )
        ours_cfg = ac.build_config(
            "full", seed=42, time_limit_seconds=0.0,
            ants=3, iterations=3, verbose_every=0,
        )
        official_solution, official_history, _ = \
            self.original.AntColonySolver(problem, official_cfg).solve()
        our_solution, our_history, _ = \
            AntColonySolver(problem, ours_cfg).solve()

        self.assertEqual(official_solution.objective, our_solution.objective)
        self.assertEqual(official_solution.code.master_routes,
                         our_solution.code.master_routes)
        self.assertEqual(official_solution.code.service,
                         our_solution.code.service)
        self.assertEqual(official_solution.code.drone,
                         our_solution.code.drone)
        self.assertEqual([r["global_best"] for r in official_history],
                         [r["global_best"] for r in our_history])
        self.assertEqual(len(official_history), len(our_history))

    def test_forced_pheromone_restarts_match(self):
        if self.original is None:
            self.skipTest("official aco_solver.py not found")
        problem = make_problem(n=7, k=2, d=2, seed=5)
        common = dict(ants=3, iterations=8, seed=77, time_limit_seconds=0.0,
                      verbose_every=0, restart_after=2)
        our_cfg = ac.build_config(
            "full", seed=77, time_limit_seconds=0.0,
            ants=3, iterations=8, verbose_every=0)
        our_cfg.restart_after = 2
        official_solution, official_history, _ = \
            self.original.AntColonySolver(
                problem, self.original.ACOConfig(**common)).solve()
        our_solution, our_history, _ = AntColonySolver(
            problem, our_cfg).solve()
        self.assertEqual(official_solution.objective, our_solution.objective)
        self.assertEqual([r["global_best"] for r in official_history],
                         [r["global_best"] for r in our_history])


if __name__ == "__main__":
    unittest.main()
