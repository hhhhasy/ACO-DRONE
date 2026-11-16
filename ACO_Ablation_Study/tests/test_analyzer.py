"""Analyzer tests: completeness, paired unit, Holm correction, no imputation."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac
import analyze_ablation as an

BUDGET = 2.0
VARIANTS = ["full", "no_edge", "no_workload_bias"]
DATASETS = ["D1", "D2", "D3", "D4", "D5", "D6"]
SEEDS = [1, 2, 3]


def write_run(results_root, variant, dataset, seed, makespan,
              code_fingerprint="test-code"):
    run_dir = ac.run_directory(results_root, variant, BUDGET, dataset, seed)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "status": ac.STATUS_COMPLETED,
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "requested_time_budget_seconds": BUDGET,
        "variant_config_fingerprint": ac.variant_fingerprint(variant),
        "run_config_fingerprint": ac.run_config_fingerprint(
            variant, seed, BUDGET),
        "code_fingerprint": code_fingerprint,
        "feasible": True,
        "makespan_s": float(makespan),
        "actual_algorithm_runtime_seconds": BUDGET,
        "iterations_started": 1,
    }
    with open(os.path.join(run_dir, ac.RUN_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return run_dir


def _base_makespan(dataset: str) -> float:
    return 1000.0 + 10.0 * DATASETS.index(dataset)


def build_complete(root: str) -> None:
    for dataset in DATASETS:
        ref = _base_makespan(dataset)
        for seed in SEEDS:
            offset = {1: -1.0, 2: 0.0, 3: 1.0}[seed]
            write_run(root, "full", dataset, seed, ref + offset)
            write_run(root, "no_edge", dataset, seed, ref + 10.0 + offset)
            write_run(root, "no_workload_bias", dataset, seed,
                      ref - 5.0 + offset)


class TestHolm(unittest.TestCase):
    def test_step_down_is_monotone_and_capped(self):
        adjusted = an.holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        for value in adjusted.values():
            self.assertLessEqual(value, 1.0)

    def test_empty(self):
        self.assertEqual(an.holm_correction({}), {})


class TestDescriptive(unittest.TestCase):
    def test_sample_std_and_cv(self):
        d = an.descriptive([10.0, 10.0, 10.0])
        self.assertEqual(d["n"], 3)
        self.assertEqual(d["mean"], 10.0)
        self.assertEqual(d["std"], 0.0)
        self.assertEqual(d["best"], 10.0)
        self.assertEqual(d["worst"], 10.0)
        d = an.descriptive([8.0, 10.0, 12.0])
        self.assertAlmostEqual(d["mean"], 10.0)
        self.assertAlmostEqual(d["std"], 2.0)
        self.assertAlmostEqual(d["cv"], 0.2)

    def test_empty_is_none_not_zero(self):
        d = an.descriptive([])
        self.assertEqual(d["n"], 0)
        self.assertIsNone(d["mean"])
        self.assertIsNone(d["std"])


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "results")
        build_complete(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_grid_uses_dataset_means(self):
        analysis = an.build_analysis(
            self.root, BUDGET, VARIANTS, DATASETS, SEEDS, min_pairs=5)
        self.assertEqual(analysis.ready_datasets, DATASETS)
        self.assertTrue(analysis.completeness["full_coverage"])
        ref = next(r for r in analysis.variant_summary
                   if r["variant"] == "full")
        no_edge = next(r for r in analysis.variant_summary
                       if r["variant"] == "no_edge")
        no_bias = next(r for r in analysis.variant_summary
                       if r["variant"] == "no_workload_bias")
        self.assertAlmostEqual(ref["equal_weight_mean_makespan"],
                               sum(_base_makespan(d) for d in DATASETS) / 6.0)
        self.assertAlmostEqual(no_edge["equal_weight_mean_makespan"],
                               ref["equal_weight_mean_makespan"] + 10.0)
        self.assertAlmostEqual(no_bias["equal_weight_mean_makespan"],
                               ref["equal_weight_mean_makespan"] - 5.0)
        self.assertEqual(no_bias["champion_count"], 6)

        stats = analysis.dataset_stats[("full", "D1")]
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["std"], 1.0)
        self.assertAlmostEqual(stats["cv"], 1.0 / 1000.0)

        edge_pair = next(r for r in analysis.paired
                         if r["variant"] == "no_edge")
        self.assertEqual(edge_pair["n_pairs"], 6)
        self.assertEqual(edge_pair["worse_datasets"], 6)
        self.assertAlmostEqual(
            edge_pair["mean_change_pct_vs_reference"],
            sum(100.0 * 10.0 / _base_makespan(d) for d in DATASETS) / 6.0,
            places=8)
        self.assertIsNotNone(edge_pair["wilcoxon_p_raw"])
        self.assertIsNotNone(edge_pair["holm_p"])
        self.assertGreaterEqual(edge_pair["holm_p"], edge_pair["wilcoxon_p_raw"])

        bias_pair = next(r for r in analysis.paired
                         if r["variant"] == "no_workload_bias")
        self.assertEqual(bias_pair["improved_datasets"], 6)
        self.assertAlmostEqual(
            bias_pair["mean_change_pct_vs_reference"],
            sum(-100.0 * 5.0 / _base_makespan(d) for d in DATASETS) / 6.0,
            places=8)

    def test_incomplete_data_is_reported_and_not_tested(self):
        for dataset in ("D3", "D4"):
            shutil.rmtree(ac.run_directory(
                self.root, "no_edge", BUDGET, dataset, SEEDS[0]))
        analysis = an.build_analysis(
            self.root, BUDGET, VARIANTS, DATASETS, SEEDS, min_pairs=5)
        self.assertEqual(len(analysis.ready_datasets), 4)
        self.assertFalse(analysis.completeness["full_coverage"])
        self.assertTrue(any("INCOMPLETE" in w for w in analysis.warnings))
        self.assertTrue(any("NO_PAIRED_TEST" in w for w in analysis.warnings))

        stats = analysis.dataset_stats[("no_edge", "D3")]
        self.assertFalse(stats["complete"])
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["expected_seeds"], 3)
        # Seed 1 was deleted; mean is over seeds 2 and 3 only (offsets 0, +1).
        self.assertAlmostEqual(stats["mean"],
                               (_base_makespan("D3") + 10.0
                                + _base_makespan("D3") + 10.0 + 1.0) / 2.0)

        edge_pair = next(r for r in analysis.paired
                         if r["variant"] == "no_edge")
        self.assertEqual(edge_pair["n_pairs"], 4)
        self.assertIsNone(edge_pair["wilcoxon_p_raw"])
        self.assertIsNone(edge_pair["holm_p"])
        self.assertEqual(edge_pair["test_method"], "not run")

    def test_mixed_code_versions_are_flagged(self):
        write_run(self.root, "no_edge", "D1", SEEDS[0],
                  _base_makespan("D1") + 10.0 - 1.0,
                  code_fingerprint="different-code")
        analysis = an.build_analysis(
            self.root, BUDGET, VARIANTS, DATASETS, SEEDS, min_pairs=5)
        self.assertEqual(len(analysis.code_fingerprints), 2)
        self.assertTrue(any("MIXED_CODE" in w for w in analysis.warnings))

    def test_tables_and_report_are_written(self):
        analysis = an.build_analysis(
            self.root, BUDGET, VARIANTS, DATASETS, SEEDS, min_pairs=5)
        out = os.path.join(self.tmp.name, "analysis")
        paths = an.write_tables(analysis, out)
        report = an.write_report(analysis, out, {}, paths)
        for name in ("dataset_means.csv", "variant_summary.csv",
                     "paired_comparisons.csv", "completeness.csv",
                     "champion_counts.csv"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), msg=name)
        self.assertTrue(os.path.isfile(report))
        with open(report, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("配对单位", text)
        self.assertIn("样本完整", text)
        self.assertIn("no_workload_bias", text)

    def test_charts_render_when_matplotlib_available(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib unavailable")
        analysis = an.build_analysis(
            self.root, BUDGET, VARIANTS, DATASETS, SEEDS, min_pairs=5)
        out = os.path.join(self.tmp.name, "charts")
        charts = an.make_charts(analysis, out)
        self.assertIn("relative_change", charts)
        self.assertIn("champion_counts", charts)
        for path in charts.values():
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 1000)


class TestSixVariantAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "results")
        self.datasets = ["E1", "E2", "E3", "E4", "E5"]
        self.seeds = [1, 2, 3]
        deltas = {
            "full": 0.0,
            "no_edge": 10.0,
            "no_assign": 5.0,
            "no_service": 15.0,
            "no_drone": 2.0,
            "no_workload_bias": 8.0,
        }
        for idx, dataset in enumerate(self.datasets):
            base = 2000.0 + 20.0 * idx
            for seed in self.seeds:
                offset = {1: -1.0, 2: 0.0, 3: 1.0}[seed]
                for variant, delta in deltas.items():
                    write_run(self.root, variant, dataset, seed,
                              base + delta + offset)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_six_variants_and_five_pairwise_tests(self):
        analysis = an.build_analysis(
            self.root, BUDGET, list(ac.VARIANT_KEYS),
            self.datasets, self.seeds, min_pairs=5)
        self.assertEqual(len(analysis.variant_summary), 6)
        self.assertEqual(len(analysis.paired), 5)
        self.assertEqual(len(analysis.ready_datasets), len(self.datasets))
        variants = {row["variant"] for row in analysis.paired}
        self.assertEqual(variants, set(ac.VARIANT_KEYS) - {ac.REFERENCE_VARIANT})
        for row in analysis.paired:
            self.assertEqual(row["n_pairs"], 5)
            self.assertIsNotNone(row["wilcoxon_p_raw"], msg=row["variant"])
            self.assertIsNotNone(row["holm_p"], msg=row["variant"])
        full = next(r for r in analysis.variant_summary
                    if r["variant"] == "full")
        self.assertEqual(full["champion_count"], 5)

    def test_six_variant_charts_render(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib unavailable")
        analysis = an.build_analysis(
            self.root, BUDGET, list(ac.VARIANT_KEYS),
            self.datasets, self.seeds, min_pairs=5)
        out = os.path.join(self.tmp.name, "charts6")
        charts = an.make_charts(analysis, out)
        self.assertIn("relative_change", charts)
        self.assertIn("relative_change_box", charts)
        self.assertIn("relative_heatmap", charts)
        for path in charts.values():
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main()
