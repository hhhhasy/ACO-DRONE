"""Pheromone switches, workload-bias ablation and coverage tests."""
from __future__ import annotations

import math
import os
import sys
import unittest

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import numpy as np

import ablation_config as ac
import aco_solver
import drone_physics
import evaluator
from aco_solver import AntColonySolver
from tests.synthetic import collect_served_customers, make_problem

LAYER_ARRAY = {
    "edge": "tau_edge",
    "assign": "tau_assign",
    "service": "tau_service",
    "drone": "tau_drone",
}
VARIANT_TARGET = {
    "no_edge": "edge",
    "no_assign": "assign",
    "no_service": "service",
    "no_drone": "drone",
}


def make_solver(variant: str, seed: int = 7, ants: int = 3,
                iterations: int = 2) -> AntColonySolver:
    cfg = ac.build_config(
        variant, seed=seed, time_limit_seconds=60.0,
        ants=ants, iterations=iterations, verbose_every=0,
    )
    return AntColonySolver(make_problem(), cfg)


class _LoadBomb:
    @property
    def evaluation(self):
        raise RuntimeError("fixed mode must not read dynamic load state")


class TestSharedEvaluator(unittest.TestCase):
    def test_all_variants_use_the_study_evaluator(self):
        self.assertTrue(os.path.samefile(
            evaluator.__file__, os.path.join(STUDY_DIR, "evaluator.py")))
        self.assertTrue(os.path.samefile(
            drone_physics.__file__, os.path.join(STUDY_DIR, "drone_physics.py")))
        self.assertIs(aco_solver.evaluate_solution, evaluator.evaluate_solution)
        self.assertTrue(os.path.samefile(
            aco_solver.__file__, os.path.join(STUDY_DIR, "aco_solver.py")))


class TestSharedAlgorithmAndEvaluator(unittest.TestCase):
    def test_all_six_variants_share_problem_and_evaluator(self):
        problem = make_problem()
        for variant in ac.VARIANT_KEYS:
            cfg = ac.build_config(
                variant, seed=13, time_limit_seconds=60.0,
                ants=3, iterations=2, verbose_every=0)
            solver = AntColonySolver(problem, cfg)
            self.assertIs(solver.p, problem, msg=variant)
            self.assertIs(aco_solver.evaluate_solution,
                          evaluator.evaluate_solution, msg=variant)
            self.assertTrue(solver.cfg.tabu_enabled, msg=variant)
            self.assertEqual(solver.cfg.evaluation_cache_size, 1024,
                             msg=variant)


class TestPheromoneFlags(unittest.TestCase):
    def test_enabled_flags_match_variant_spec(self):
        for variant in ac.VARIANT_KEYS:
            solver = make_solver(variant)
            spec = ac.get_variant(variant)
            self.assertEqual(solver.pheromone_enabled, {
                "edge": spec.pheromone_edge,
                "assign": spec.pheromone_assign,
                "service": spec.pheromone_service,
                "drone": spec.pheromone_drone,
            })
            self.assertEqual(solver.pheromone_statistics()["enabled"],
                             solver.pheromone_enabled)

    def test_disabled_term_is_neutral_one(self):
        for variant in ac.VARIANT_KEYS:
            solver = make_solver(variant)
            solver.tau_edge[1, 2] = 3.7
            solver.tau_assign[0, 1] = 3.7
            solver.tau_service[1, 0] = 3.7
            solver.tau_drone[1, 0] = 3.7
            enabled = solver.pheromone_enabled
            self.assertEqual(solver._tau_edge_term(1, 2),
                             3.7 if enabled["edge"] else 1.0)
            self.assertEqual(solver._tau_assign_term(0, 1),
                             3.7 if enabled["assign"] else 1.0)
            self.assertEqual(solver._tau_service_term(1, 0),
                             3.7 if enabled["service"] else 1.0)
            self.assertEqual(solver._tau_drone_term(1, 0),
                             3.7 if enabled["drone"] else 1.0)


class TestDisabledLayerDoesNotInfluenceConstruction(unittest.TestCase):
    def test_no_edge_ignores_tau_edge(self):
        a, b = make_solver("no_edge"), make_solver("no_edge")
        a.tau_edge.fill(1e6)
        self.assertEqual(a._construct_master_routes(), b._construct_master_routes())

    def test_no_assign_ignores_tau_assign(self):
        a, b = make_solver("no_assign"), make_solver("no_assign")
        a.tau_assign.fill(1e6)
        self.assertEqual(a._construct_master_routes(), b._construct_master_routes())

    def test_no_service_ignores_tau_service(self):
        a, b = make_solver("no_service"), make_solver("no_service")
        routes_a = a._construct_master_routes()
        routes_b = b._construct_master_routes()
        self.assertEqual(routes_a, routes_b)
        a.tau_service.fill(1e6)
        self.assertEqual(a._construct_service_and_drone(routes_a),
                         b._construct_service_and_drone(routes_b))

    def test_no_drone_ignores_tau_drone(self):
        a, b = make_solver("no_drone"), make_solver("no_drone")
        routes_a = a._construct_master_routes()
        routes_b = b._construct_master_routes()
        self.assertEqual(routes_a, routes_b)
        a.tau_drone.fill(1e6)
        self.assertEqual(a._construct_service_and_drone(routes_a),
                         b._construct_service_and_drone(routes_b))


class TestDisabledLayerUpdatesSkipped(unittest.TestCase):
    def test_only_target_layer_untouched_by_all_update_paths(self):
        for variant, target in VARIANT_TARGET.items():
            solver = make_solver(variant)
            for array in LAYER_ARRAY.values():
                getattr(solver, array)[:] = 2.5 * solver.tau0
            before = {
                name: getattr(solver, array).copy()
                for name, array in LAYER_ARRAY.items()
            }

            solver._local_update_edge(0, 1)
            solver._local_update_assignment(0, 1)
            solver._pheromone_restart()
            solver._global_update(solver.global_best)

            after = {
                name: getattr(solver, array)
                for name, array in LAYER_ARRAY.items()
            }
            self.assertTrue(
                np.array_equal(before[target], after[target]),
                msg="%s must leave %s untouched" % (variant, target),
            )
            for name in LAYER_ARRAY:
                if name == target:
                    continue
                self.assertFalse(
                    np.array_equal(before[name], after[name]),
                    msg="%s: enabled layer %s should still be updated"
                        % (variant, name),
                )


class TestWorkloadBiasAblation(unittest.TestCase):
    def test_full_and_no_bias_use_adaptive_strategy_only(self):
        for variant in ("full", "no_workload_bias"):
            solver = make_solver(variant)
            routes = solver._greedy_all_truck_code().master_routes

            def forbidden(*_args, **_kwargs):
                raise AssertionError("%s called the legacy strategy" % variant)

            # The legacy path is the only one that consults _service_heuristics.
            solver._service_heuristics = forbidden
            for _ in range(50):
                solver._construct_service_and_drone(routes)
            stats = solver.service_policy_statistics()
            self.assertEqual(stats["mode"], "fixed", msg=variant)
            self.assertEqual(stats["adaptive_ants"], 50, msg=variant)
            self.assertEqual(stats["legacy_ants"], 0, msg=variant)
            self.assertEqual(stats["probability_mean"], 1.0, msg=variant)
            self.assertEqual(stats["probability_last"], 1.0, msg=variant)

    def test_fixed_probability_does_not_read_load_state(self):
        for variant in ("full", "no_workload_bias"):
            solver = make_solver(variant)
            solver.global_best = _LoadBomb()
            self.assertEqual(solver._service_policy_probability(), 1.0,
                             msg=variant)
            self.assertEqual(solver.policy_workload_last, 0.0)
            self.assertEqual(solver.policy_busy_last, 0.0)
            self.assertEqual(solver.policy_energy_last, 0.0)
            self.assertEqual(solver.policy_truck_factor_last, 0.0)

    def test_no_bias_terms_are_exactly_one(self):
        solver = make_solver("no_workload_bias")
        self.assertEqual(solver.cfg.service_proxy_strength, 0.0)
        for signed in (-2.2, -1.0, 0.0, 1.0, 2.2):
            self.assertEqual(solver._service_bias_terms(signed), (1.0, 1.0))

    def test_full_bias_terms_are_reciprocal(self):
        solver = make_solver("full")
        self.assertEqual(solver.cfg.service_proxy_strength, 1.0)
        for signed in (-2.2, -1.0, 0.0, 1.0, 2.2):
            eta_truck, eta_drone = solver._service_bias_terms(signed)
            self.assertAlmostEqual(eta_truck * eta_drone, 1.0, places=15)
            self.assertAlmostEqual(eta_drone / eta_truck,
                                   math.exp(signed), places=15)
            self.assertAlmostEqual(eta_truck,
                                   math.exp(-0.5 * signed), places=15)

    def test_no_bias_still_uses_tau_service_for_acs_choice(self):
        truck_solver = make_solver("no_workload_bias")
        drone_solver = make_solver("no_workload_bias")
        routes_t = truck_solver._construct_master_routes()
        routes_d = drone_solver._construct_master_routes()
        self.assertEqual(routes_t, routes_d)
        # Force the ACS greedy branch so the test isolates pheromone scores.
        truck_solver.cfg.q0 = 1.0
        drone_solver.cfg.q0 = 1.0

        # With kappa=0 the only service score is tau_service ** alpha * 1.
        # Opposite pheromone priors must therefore produce opposite choices.
        truck_solver.tau_service[:, 0] = 1e12
        truck_solver.tau_service[:, 1] = 1e-12
        drone_solver.tau_service[:, 0] = 1e-12
        drone_solver.tau_service[:, 1] = 1e12
        service_t, _ = truck_solver._construct_service_and_drone(routes_t)
        service_d, _ = drone_solver._construct_service_and_drone(routes_d)

        n = truck_solver.p.num_customers
        self.assertEqual(sum(1 for c in range(1, n + 1) if service_t[c] == 0), n)
        self.assertEqual(sum(1 for c in range(1, n + 1) if service_d[c] == 1), n)
        self.assertNotEqual(service_t, service_d)


class TestCoverageUnderAblation(unittest.TestCase):
    def test_each_variant_serves_every_customer_exactly_once(self):
        for variant in ac.VARIANT_KEYS:
            solver = make_solver(variant, seed=13, ants=2, iterations=1)
            solution = solver.construct_ant()
            self.assertTrue(solution.feasible, msg=variant)
            served = collect_served_customers(solution)
            self.assertEqual(sorted(served), list(range(1, 8)), msg=variant)
            self.assertEqual(len(served), len(set(served)), msg=variant)


if __name__ == "__main__":
    unittest.main()
