"""Variant registry, config isolation and output-path tests."""
from __future__ import annotations

import os
import sys
import unittest

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac


class TestVariantRegistry(unittest.TestCase):
    def test_six_variants_in_order(self):
        self.assertEqual(
            list(ac.VARIANT_KEYS),
            ["full", "no_edge", "no_assign", "no_service", "no_drone",
             "no_workload_bias"],
        )

    def test_reference_is_full(self):
        self.assertEqual(ac.REFERENCE_VARIANT, "full")

    def test_strict_single_factor_isolation(self):
        # Raises AssertionError if a variant differs in more than one factor.
        ac.validate_variant_isolation()

    def test_full_is_current_aco_setting(self):
        cfg = ac.build_config("full", seed=13, time_limit_seconds=50.0)
        self.assertEqual(cfg.service_policy_mode, "fixed")
        self.assertEqual(cfg.adaptive_service_fraction, 1.0)
        self.assertEqual(cfg.service_proxy_strength, 1.0)
        self.assertTrue(cfg.tabu_enabled)
        self.assertEqual(cfg.evaluation_cache_size, 1024)
        for flag in ("pheromone_edge", "pheromone_assign",
                     "pheromone_service", "pheromone_drone"):
            self.assertTrue(getattr(cfg, flag), msg=flag)

    def test_each_no_variant_disables_one_layer(self):
        expected = {
            "no_edge": "pheromone_edge",
            "no_assign": "pheromone_assign",
            "no_service": "pheromone_service",
            "no_drone": "pheromone_drone",
        }
        for variant, flag in expected.items():
            diff = ac.config_diff(variant, ac.REFERENCE_VARIANT)
            self.assertEqual(set(diff), {flag}, msg=variant)
            self.assertEqual(diff[flag], (False, True), msg=variant)

    def test_no_workload_bias_changes_only_kappa(self):
        diff = ac.config_diff("no_workload_bias", ac.REFERENCE_VARIANT)
        self.assertEqual(set(diff), {"service_proxy_strength"})
        self.assertEqual(diff["service_proxy_strength"], (0.0, 1.0))
        cfg = ac.build_config("no_workload_bias", seed=13,
                              time_limit_seconds=50.0)
        self.assertEqual(cfg.service_policy_mode, "fixed")
        self.assertEqual(cfg.adaptive_service_fraction, 1.0)
        self.assertTrue(cfg.pheromone_edge)
        self.assertTrue(cfg.pheromone_assign)
        self.assertTrue(cfg.pheromone_service)
        self.assertTrue(cfg.pheromone_drone)

    def test_all_variants_share_non_ablation_parameters(self):
        allowed = {
            "service_proxy_strength",
            "pheromone_edge",
            "pheromone_assign",
            "pheromone_service",
            "pheromone_drone",
        }
        reference = ac.stable_config_dict(
            ac.build_config(ac.REFERENCE_VARIANT, seed=13,
                            time_limit_seconds=50.0))
        for variant in ac.VARIANT_KEYS:
            cfg = ac.build_config(variant, seed=13, time_limit_seconds=50.0)
            self.assertEqual(cfg.service_policy_mode, "fixed", msg=variant)
            self.assertEqual(cfg.adaptive_service_fraction, 1.0, msg=variant)
            current = ac.stable_config_dict(cfg)
            for key, value in reference.items():
                if key in allowed:
                    continue
                self.assertEqual(current[key], value,
                                 msg="%s changed non-ablation field %s"
                                     % (variant, key))

    def test_fingerprints_are_stable_and_distinct(self):
        for variant in ac.VARIANT_KEYS:
            self.assertEqual(ac.variant_fingerprint(variant),
                             ac.variant_fingerprint(variant))
        fps = {ac.variant_fingerprint(v) for v in ac.VARIANT_KEYS}
        self.assertEqual(len(fps), len(ac.VARIANT_KEYS))

    def test_run_fingerprint_responds_to_seed_and_budget(self):
        base = ac.run_config_fingerprint("full", 13, 50.0)
        self.assertNotEqual(base, ac.run_config_fingerprint("full", 21, 50.0))
        self.assertNotEqual(base, ac.run_config_fingerprint("full", 13, 30.0))


class TestOutputPaths(unittest.TestCase):
    def test_every_combination_has_a_unique_path(self):
        seen = set()
        for variant in ac.VARIANT_KEYS:
            for budget in ac.FORMAL_BUDGETS:
                for seed in ac.FORMAL_SEEDS:
                    path = ac.run_directory(
                        "results_root", variant, budget, "A-n32-k5", seed)
                    self.assertNotIn(path, seen)
                    seen.add(path)
                    self.assertTrue(path.endswith(
                        os.path.join(variant, "%gs" % budget,
                                     "A-n32-k5", "seed%d" % seed)))

    def test_paths_stay_below_results_root(self):
        root = os.path.abspath("results_root")
        path = ac.run_directory(root, "no_workload_bias", 50.0,
                                "P-n16-k8", 2024)
        self.assertEqual(os.path.commonpath([root, path]), root)

    def test_budget_tag(self):
        self.assertEqual(ac.format_budget_tag(50.0), "50s")
        self.assertEqual(ac.format_budget_tag(10), "10s")
        self.assertEqual(ac.format_budget_tag(2.0), "2s")

    def test_formal_grid_size(self):
        self.assertEqual(len(ac.FORMAL_SEEDS), 10)
        self.assertEqual(len(ac.FORMAL_BUDGETS), 4)
        self.assertEqual(len(ac.VARIANT_KEYS), 6)
        self.assertEqual(ac.MAIN_EXPERIMENT_RUNS, 1800)
        self.assertEqual(
            len(ac.FORMAL_SEEDS) * len(ac.VARIANT_KEYS), 60,
            msg="one dataset at the main budget has 60 cells")
        self.assertEqual(
            len(ac.FORMAL_SEEDS) * len(ac.FORMAL_BUDGETS)
            * len(ac.VARIANT_KEYS), 240,
            msg="one dataset across all budgets has 240 cells")


if __name__ == "__main__":
    unittest.main()
