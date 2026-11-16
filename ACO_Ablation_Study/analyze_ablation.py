"""Analyse completed ACO+Tabu ablation results.

Statistical unit
----------------
The paired unit is **one dataset**, represented by the mean makespan of its
10 configured seeds.  Individual seeds are NOT treated as independent
datasets.  A dataset enters the paired comparison only when every selected
variant has all expected seeds completed and feasible.

If the result set is incomplete, the analyser:

* reports exactly what is missing;
* computes descriptive statistics only from runs that actually exist;
* refuses to run the paired Wilcoxon test on an incomplete pair set;
* never fills, imputes or invents a value.

Outputs
-------
``dataset_means.csv``, ``variant_summary.csv``, ``paired_comparisons.csv``,
``completeness.csv``, ``analysis_manifest.json``, ``analysis_report.md`` and
(if matplotlib is available) several PNG figures, all below
``--output-dir``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac  # noqa: E402

TIE_TOLERANCE_S = 1e-6
MISSING = "missing"
PARTIAL = "partial"
INFEASIBLE = "infeasible"
INVALID = "invalid"
CONFIG_MISMATCH = "config_mismatch"
COMPLETE = "complete"

# Reference chart colors (validated default palette, light mode).
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_BLUE = "#2a78d6"
COLOR_BLUE_DARK = "#184f95"
COLOR_BLUE_LIGHT = "#9ec5f4"
COLOR_RED = "#e34948"
COLOR_NEUTRAL = "#f0efec"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    variant: str
    dataset: str
    seed: int
    state: str
    makespan: Optional[float] = None
    feasible: Optional[bool] = None
    runtime: Optional[float] = None
    iterations: Optional[int] = None
    code_fingerprint: Optional[str] = None
    run_config_fingerprint: Optional[str] = None
    detail: str = ""
    path: str = ""


def _read_json(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def discover_result_datasets(
    results_root: str,
    budget: float,
    variants: Sequence[str],
) -> List[str]:
    """Union of dataset directory names present for the selected variants."""
    tag = ac.format_budget_tag(budget)
    names = set()
    for variant in variants:
        base = os.path.join(os.path.abspath(results_root), variant, tag)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if os.path.isdir(os.path.join(base, name)):
                names.add(name)
    return sorted(names)


def _cell_from_paths(
    results_root: str,
    variant: str,
    budget: float,
    dataset: str,
    seed: int,
    expected_variant_fp: str,
) -> Cell:
    run_dir = ac.run_directory(results_root, variant, budget, dataset, seed)
    run_path = os.path.join(run_dir, ac.RUN_FILENAME)
    if not os.path.isdir(run_dir):
        return Cell(variant, dataset, seed, MISSING, path=run_dir)
    payload = _read_json(run_path)
    if payload is None:
        return Cell(variant, dataset, seed, PARTIAL,
                    detail="run.json missing or unreadable", path=run_dir)
    if payload.get("status") != ac.STATUS_COMPLETED:
        return Cell(variant, dataset, seed, PARTIAL,
                    detail="status=%r" % payload.get("status"), path=run_dir)
    if (str(payload.get("variant")) != variant
            or str(payload.get("dataset")) != dataset
            or int(payload.get("seed", -1)) != int(seed)):
        return Cell(variant, dataset, seed, INVALID,
                    detail="run identity does not match its directory", path=run_dir)
    recorded_budget = payload.get("requested_time_budget_seconds")
    if recorded_budget is None or abs(float(recorded_budget) - float(budget)) > 1e-9:
        return Cell(variant, dataset, seed, INVALID,
                    detail="recorded budget %r != %r" % (recorded_budget, budget),
                    path=run_dir)
    if payload.get("variant_config_fingerprint") != expected_variant_fp:
        return Cell(variant, dataset, seed, CONFIG_MISMATCH,
                    detail="variant configuration fingerprint differs", path=run_dir)
    if not bool(payload.get("feasible")):
        return Cell(variant, dataset, seed, INFEASIBLE,
                    detail="recorded run infeasible", path=run_dir)
    makespan = payload.get("makespan_s")
    if not isinstance(makespan, (int, float)) or not math.isfinite(float(makespan)):
        return Cell(variant, dataset, seed, INVALID,
                    detail="non-finite makespan", path=run_dir)
    return Cell(
        variant=variant,
        dataset=dataset,
        seed=int(seed),
        state=COMPLETE,
        makespan=float(makespan),
        feasible=bool(payload.get("feasible")),
        runtime=(float(payload["actual_algorithm_runtime_seconds"])
                 if isinstance(payload.get("actual_algorithm_runtime_seconds"),
                               (int, float)) else None),
        iterations=int(payload.get("iterations_started", 0) or 0),
        code_fingerprint=(str(payload.get("code_fingerprint"))
                          if payload.get("code_fingerprint") else None),
        run_config_fingerprint=(str(payload.get("run_config_fingerprint"))
                                if payload.get("run_config_fingerprint") else None),
        detail="ok",
        path=run_dir,
    )


def collect_cells(
    results_root: str,
    budget: float,
    variants: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> List[Cell]:
    cells: List[Cell] = []
    for variant in variants:
        expected_fp = ac.variant_fingerprint(variant)
        for dataset in datasets:
            for seed in seeds:
                cells.append(_cell_from_paths(
                    results_root, variant, budget, dataset, int(seed), expected_fp))
    return cells


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _sample_std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def descriptive(values: Sequence[float]) -> Dict[str, Optional[float]]:
    mean = _mean(values)
    std = _sample_std(values)
    return {
        "n": len(values),
        "best": min(values) if values else None,
        "worst": max(values) if values else None,
        "mean": mean,
        "std": std,
        "cv": (std / mean) if (std is not None and mean not in (None, 0.0)) else None,
    }


def holm_correction(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(((k, float(p)) for k, p in pvalues.items()), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def wilcoxon_two_sided(
    a: Sequence[float],
    b: Sequence[float],
) -> Dict[str, object]:
    """Two-sided paired Wilcoxon signed-rank test (b vs a).

    Returns ``{statistic, p_value, method, note}``.  ``p_value`` is None when
    SciPy is unavailable; no approximation is fabricated in that case.
    """
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    if not diffs:
        return {"statistic": None, "p_value": None, "method": "unavailable",
                "note": "no paired data"}
    if all(abs(d) <= TIE_TOLERANCE_S for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0, "method": "degenerate",
                "note": "all paired differences are zero"}
    try:
        from scipy.stats import wilcoxon  # noqa: WPS433 (optional dependency)
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"statistic": None, "p_value": None, "method": "unavailable",
                "note": "scipy unavailable: %s" % exc}
    try:
        result = wilcoxon(list(b), list(a), zero_method="wilcox",
                          correction=False, alternative="two-sided")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy.stats.wilcoxon",
            "note": "two-sided paired signed-rank; zero_method=wilcox",
        }
    except Exception as exc:  # e.g. all-zero differences in older SciPy
        return {"statistic": None, "p_value": None, "method": "error",
                "note": "Wilcoxon failed: %s" % exc}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class Analysis:
    budget: float
    variants: List[str]
    datasets: List[str]
    seeds: List[int]
    cells: List[Cell]
    dataset_stats: Dict[Tuple[str, str], Dict[str, object]]
    variant_complete_datasets: Dict[str, List[str]]
    ready_datasets: List[str]
    variant_summary: List[Dict[str, object]]
    paired: List[Dict[str, object]]
    completeness: Dict[str, object]
    warnings: List[str] = field(default_factory=list)
    code_fingerprints: Dict[str, int] = field(default_factory=dict)

    @property
    def inference_allowed(self) -> bool:
        return bool(self.ready_datasets) and not any(
            "no paired test" in w for w in self.warnings
        )


def _aggregate_variant(
    variant: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    cells_by_key: Mapping[Tuple[str, str, int], Cell],
) -> Tuple[Dict[Tuple[str, str], Dict[str, object]], List[str]]:
    stats: Dict[Tuple[str, str], Dict[str, object]] = {}
    complete_datasets: List[str] = []
    for dataset in datasets:
        values: List[float] = []
        missing: List[int] = []
        for seed in seeds:
            cell = cells_by_key[(variant, dataset, int(seed))]
            if cell.state == COMPLETE:
                values.append(float(cell.makespan))
            else:
                missing.append(int(seed))
        d = descriptive(values)
        d.update({
            "variant": variant,
            "dataset": dataset,
            "expected_seeds": len(seeds),
            "missing_seeds": missing,
            "complete": len(missing) == 0,
        })
        stats[(variant, dataset)] = d
        if d["complete"]:
            complete_datasets.append(dataset)
    return stats, complete_datasets


def build_analysis(
    results_root: str,
    budget: float,
    variants: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    reference: str = ac.REFERENCE_VARIANT,
    alpha: float = 0.05,
    min_pairs: int = 5,
) -> Analysis:
    cells = collect_cells(results_root, budget, variants, datasets, seeds)
    by_key = {(c.variant, c.dataset, c.seed): c for c in cells}

    dataset_stats: Dict[Tuple[str, str], Dict[str, object]] = {}
    variant_complete: Dict[str, List[str]] = {}
    for variant in variants:
        stats, complete = _aggregate_variant(variant, datasets, seeds, by_key)
        for key, value in stats.items():
            dataset_stats[key] = value
        variant_complete[variant] = complete

    common = set(datasets)
    for variant in variants:
        common &= set(variant_complete[variant])
    ready = sorted(common)

    warnings: List[str] = []
    code_counts: Dict[str, int] = {}
    for cell in cells:
        if cell.state == COMPLETE and cell.code_fingerprint:
            code_counts[cell.code_fingerprint] = code_counts.get(cell.code_fingerprint, 0) + 1
    if len(code_counts) > 1:
        warnings.append(
            "MIXED_CODE: results come from %d different solver source "
            "fingerprints; inference is refused until rerun under one code version."
            % len(code_counts)
        )

    # Completeness summary.
    state_counts: Dict[str, int] = {}
    for cell in cells:
        state_counts[cell.state] = state_counts.get(cell.state, 0) + 1
    per_variant_missing: Dict[str, Dict[str, List[int]]] = {}
    for variant in variants:
        per_variant_missing[variant] = {}
        for dataset in datasets:
            miss = [s for s in seeds
                    if by_key[(variant, dataset, int(s))].state != COMPLETE]
            if miss:
                per_variant_missing[variant][dataset] = miss
    expected_cells = len(variants) * len(datasets) * len(seeds)
    completeness = {
        "expected_cells": int(expected_cells),
        "complete_cells": int(state_counts.get(COMPLETE, 0)),
        "state_counts": state_counts,
        "per_variant_missing": per_variant_missing,
        "ready_datasets": ready,
        "ready_dataset_count": len(ready),
        "expected_dataset_count": len(datasets),
        "full_coverage": len(ready) == len(datasets) and len(datasets) > 0,
    }
    if len(ready) < len(datasets):
        warnings.append(
            "INCOMPLETE: %d of %d datasets are complete for every selected "
            "variant; paired inference uses only the %d complete datasets "
            "(never imputed)." % (len(ready), len(datasets), len(ready))
        )

    # Per-variant equal-weight means over the common ready set.
    variant_rows: List[Dict[str, object]] = []
    for variant in variants:
        means = [float(dataset_stats[(variant, ds)]["mean"]) for ds in ready]
        own_means = [float(dataset_stats[(variant, ds)]["mean"])
                     for ds in variant_complete[variant]]
        cvs = [float(dataset_stats[(variant, ds)]["cv"]) for ds in ready
               if dataset_stats[(variant, ds)]["cv"] is not None]
        champions: Optional[int] = 0 if ready else None
        champion_ties: Optional[int] = 0 if ready else None
        for ds in ready:
            best = min(float(dataset_stats[(v, ds)]["mean"]) for v in variants)
            own = float(dataset_stats[(variant, ds)]["mean"])
            if abs(own - best) <= TIE_TOLERANCE_S:
                champions += 1
                if sum(1 for v in variants
                       if abs(float(dataset_stats[(v, ds)]["mean"]) - best)
                       <= TIE_TOLERANCE_S) > 1:
                    champion_ties += 1
        variant_rows.append({
            "variant": variant,
            "is_reference": variant == reference,
            "n_datasets_ready": len(ready),
            "equal_weight_mean_makespan": _mean(means),
            "mean_std_across_datasets": _sample_std(means),
            "mean_cv_across_datasets": _mean(cvs),
            "champion_count": champions,
            "champion_tie_count": champion_ties,
            "n_datasets_own_complete": len(variant_complete[variant]),
            "own_complete_equal_weight_mean": _mean(own_means),
        })

    ref_row = next((r for r in variant_rows if r["variant"] == reference), None)
    ref_mean = None if ref_row is None else ref_row["equal_weight_mean_makespan"]

    # Paired comparisons: one pair per dataset (mean over seeds).
    paired_rows: List[Dict[str, object]] = []
    pvals: Dict[str, float] = {}
    for variant in variants:
        if variant == reference:
            continue
        pairs: List[Tuple[float, float]] = []
        for ds in ready:
            r = dataset_stats[(reference, ds)]["mean"]
            v = dataset_stats[(variant, ds)]["mean"]
            if r is None or v is None:
                continue
            pairs.append((float(r), float(v)))
        improved = worse = tied = 0
        changes_pct: List[float] = []
        for r, v in pairs:
            delta = v - r
            if delta < -TIE_TOLERANCE_S:
                improved += 1
            elif delta > TIE_TOLERANCE_S:
                worse += 1
            else:
                tied += 1
            if abs(r) > 0.0:
                changes_pct.append(100.0 * delta / r)
        ref_vals = [x for x, _ in pairs]
        var_vals = [y for _, y in pairs]
        mean_ref = _mean(ref_vals)
        mean_var = _mean(var_vals)
        mean_delta = (mean_var - mean_ref) if (mean_ref is not None
                                                and mean_var is not None) else None
        mean_change_pct = (_mean(changes_pct) if changes_pct else None)

        test_note = ""
        if len(pairs) < min_pairs:
            test = {"statistic": None, "p_value": None,
                    "method": "not run",
                    "note": "only %d paired datasets (< %d required)"
                            % (len(pairs), min_pairs)}
            test_note = test["note"]
        else:
            test = wilcoxon_two_sided(ref_vals, var_vals)
            test_note = str(test["note"])
        if test["p_value"] is not None:
            pvals[variant] = float(test["p_value"])

        paired_rows.append({
            "variant": variant,
            "reference": reference,
            "n_pairs": len(pairs),
            "reference_equal_weight_mean": mean_ref,
            "variant_equal_weight_mean": mean_var,
            "mean_delta_makespan_s": mean_delta,
            "mean_change_pct_vs_reference": mean_change_pct,
            "improved_datasets": improved,
            "worse_datasets": worse,
            "tied_datasets": tied,
            "wilcoxon_statistic": test["statistic"],
            "wilcoxon_p_raw": test["p_value"],
            "test_method": test["method"],
            "test_note": test_note,
        })

    holm = holm_correction(pvals) if pvals else {}
    for row in paired_rows:
        variant = str(row["variant"])
        raw = row["wilcoxon_p_raw"]
        row["holm_p"] = holm.get(variant)
        row["significant_after_holm"] = (
            None if row["holm_p"] is None else bool(float(row["holm_p"]) <= alpha)
        )

    if not ready:
        warnings.append(
            "NO_PAIRED_TEST: no dataset has a complete feasible result for "
            "every selected variant; p-values and champion counts are not "
            "reported for the reference comparison."
        )
    elif any(row["n_pairs"] < min_pairs for row in paired_rows):
        warnings.append(
            "NO_PAIRED_TEST for at least one variant: fewer than %d complete "
            "paired datasets." % min_pairs
        )

    return Analysis(
        budget=budget,
        variants=list(variants),
        datasets=list(datasets),
        seeds=[int(s) for s in seeds],
        cells=cells,
        dataset_stats=dataset_stats,
        variant_complete_datasets=variant_complete,
        ready_datasets=ready,
        variant_summary=variant_rows,
        paired=paired_rows,
        completeness=completeness,
        warnings=warnings,
        code_fingerprints=code_counts,
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return ("%.*f" % (digits, value))
    return str(value)


def _write_csv(path: str, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k), 6)
                             if not isinstance(row.get(k), (list, dict))
                             else json.dumps(row.get(k), ensure_ascii=False)
                             for k in fields})


def write_tables(analysis: Analysis, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    dataset_rows = []
    for variant in analysis.variants:
        for dataset in analysis.datasets:
            s = analysis.dataset_stats[(variant, dataset)]
            dataset_rows.append({
                "variant": variant,
                "dataset": dataset,
                "n_seeds": s["n"],
                "expected_seeds": s["expected_seeds"],
                "complete": s["complete"],
                "best_makespan_s": s["best"],
                "worst_makespan_s": s["worst"],
                "mean_makespan_s": s["mean"],
                "std_makespan_s": s["std"],
                "cv": s["cv"],
                "missing_seeds": s["missing_seeds"],
            })
    paths["dataset_means"] = os.path.join(output_dir, "dataset_means.csv")
    _write_csv(paths["dataset_means"], [
        "variant", "dataset", "n_seeds", "expected_seeds", "complete",
        "best_makespan_s", "worst_makespan_s", "mean_makespan_s",
        "std_makespan_s", "cv", "missing_seeds",
    ], dataset_rows)

    paths["variant_summary"] = os.path.join(output_dir, "variant_summary.csv")
    _write_csv(paths["variant_summary"], [
        "variant", "is_reference", "n_datasets_ready",
        "equal_weight_mean_makespan", "mean_std_across_datasets",
        "mean_cv_across_datasets", "champion_count", "champion_tie_count",
        "n_datasets_own_complete", "own_complete_equal_weight_mean",
    ], analysis.variant_summary)

    paths["paired"] = os.path.join(output_dir, "paired_comparisons.csv")
    _write_csv(paths["paired"], [
        "variant", "reference", "n_pairs", "reference_equal_weight_mean",
        "variant_equal_weight_mean", "mean_delta_makespan_s",
        "mean_change_pct_vs_reference", "improved_datasets", "worse_datasets",
        "tied_datasets", "wilcoxon_statistic", "wilcoxon_p_raw", "holm_p",
        "significant_after_holm", "test_method", "test_note",
    ], analysis.paired)

    completeness_rows = []
    for cell in analysis.cells:
        completeness_rows.append({
            "variant": cell.variant,
            "dataset": cell.dataset,
            "seed": cell.seed,
            "state": cell.state,
            "makespan_s": cell.makespan,
            "runtime_seconds": cell.runtime,
            "iterations": cell.iterations,
            "detail": cell.detail,
            "path": cell.path,
        })
    paths["completeness"] = os.path.join(output_dir, "completeness.csv")
    _write_csv(paths["completeness"], [
        "variant", "dataset", "seed", "state", "makespan_s",
        "runtime_seconds", "iterations", "detail", "path",
    ], completeness_rows)

    champion_rows = []
    for variant in analysis.variants:
        row = next(r for r in analysis.variant_summary if r["variant"] == variant)
        champion_rows.append({
            "variant": variant,
            "champion_count": row["champion_count"],
            "champion_tie_count": row["champion_tie_count"],
            "n_datasets_ready": row["n_datasets_ready"],
        })
    paths["champions"] = os.path.join(output_dir, "champion_counts.csv")
    _write_csv(paths["champions"], [
        "variant", "champion_count", "champion_tie_count", "n_datasets_ready",
    ], champion_rows)
    return paths


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(
    analysis: Analysis,
    output_dir: str,
    charts: Mapping[str, str],
    paths: Mapping[str, str],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ref = ac.REFERENCE_VARIANT
    complete = analysis.completeness
    lines: List[str] = []
    lines.append("# ACO+Tabu 消融实验分析报告")
    lines.append("")
    lines.append("- 时间预算: %gs" % analysis.budget)
    lines.append("- 数据集数: %d（期望）; 全部版本完整可配对: %d"
                 % (len(analysis.datasets), len(analysis.ready_datasets)))
    lines.append("- 随机种子: %s" % ", ".join(str(s) for s in analysis.seeds))
    lines.append("- 版本: %s" % ", ".join(analysis.variants))
    lines.append("- 参照版本: `%s`" % ref)
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append(
        "配对单位是**一个数据集在 %d 个种子上的平均 Makespan**，不是单个随机种子。"
        "只有当一个数据集在全部所选版本、全部种子上都完成且可行时，才进入配对检验。"
        "缺失值不会被插补或编造。" % len(analysis.seeds)
    )
    lines.append("")
    lines.append("相对变化 = 100 × (版本均值 − 参照均值) / 参照均值；"
                 "**负值表示 Makespan 更低（更好）**，正值表示退步。")
    lines.append("")
    if analysis.warnings:
        lines.append("## 完整性警告")
        lines.append("")
        for warning in analysis.warnings:
            lines.append("- %s" % warning)
        lines.append("")
    lines.append("## 完整性与样本")
    lines.append("")
    lines.append(_markdown_table(
        ["项目", "数值"],
        [
            ["期望运行单元", complete["expected_cells"]],
            ["完成运行单元", complete["complete_cells"]],
            ["完整可配对数据集", "%d / %d" % (complete["ready_dataset_count"],
                                              complete["expected_dataset_count"])],
            ["运行状态计数", json.dumps(complete["state_counts"], ensure_ascii=False)],
            ["代码指纹种类", len(analysis.code_fingerprints)],
        ],
    ))
    lines.append("")
    if complete["per_variant_missing"]:
        missing_rows = []
        for variant, per_ds in complete["per_variant_missing"].items():
            for dataset, seeds in per_ds.items():
                missing_rows.append([variant, dataset, ",".join(str(s) for s in seeds)])
        lines.append("### 缺失/不完整运行")
        lines.append("")
        lines.append(_markdown_table(["版本", "数据集", "缺失种子"], missing_rows[:200]))
        if len(missing_rows) > 200:
            lines.append("")
            lines.append("（仅显示前 200 行，完整清单见 `completeness.csv`）")
        lines.append("")

    lines.append("## 各版本汇总（等权数据集平均）")
    lines.append("")
    summary_rows = []
    for row in analysis.variant_summary:
        summary_rows.append([
            row["variant"],
            _fmt(row["n_datasets_ready"]),
            _fmt(row["equal_weight_mean_makespan"], 3),
            _fmt(row["mean_cv_across_datasets"], 4),
            _fmt(row["champion_count"]) if row["champion_count"] is not None else "—",
            _fmt(row["champion_tie_count"]) if row["champion_tie_count"] is not None else "—",
        ])
    lines.append(_markdown_table(
        ["版本", "可用数据集", "等权平均 Makespan (s)", "数据集 CV 均值",
         "冠军数", "含并列冠军数"], summary_rows))
    lines.append("")

    lines.append("## 相对 %s 的配对比较" % ref)
    lines.append("")
    if analysis.ready_datasets:
        paired_rows = []
        for row in analysis.paired:
            paired_rows.append([
                row["variant"],
                row["n_pairs"],
                _fmt(row["mean_change_pct_vs_reference"], 3),
                "%d / %d / %d" % (row["improved_datasets"],
                                  row["worse_datasets"], row["tied_datasets"]),
                _fmt(row["wilcoxon_statistic"], 3),
                _fmt(row["wilcoxon_p_raw"], 6),
                _fmt(row["holm_p"], 6),
                ("是" if row["significant_after_holm"] else "否")
                if row["significant_after_holm"] is not None else "不可用",
            ])
        lines.append(_markdown_table(
            ["版本", "配对数据集", "平均变化 %（负=更好）", "改善/退步/持平",
             "Wilcoxon W", "原始 p", "Holm p", "Holm 后显著 (α=0.05)"],
            paired_rows))
        lines.append("")
        lines.append(
            "Wilcoxon 检验为双侧配对符号秩检验，输入是每个数据集 10 个种子的均值。"
            "Holm 校正覆盖本次所有有 p 值的版本比较。"
        )
    else:
        lines.append("**没有完整可配对的数据集，未执行任何配对检验，也不报告显著性结论。**")
    lines.append("")

    lines.append("## 各数据集详细均值")
    lines.append("")
    detail_rows = []
    for dataset in analysis.datasets:
        row = [dataset]
        for variant in analysis.variants:
            s = analysis.dataset_stats[(variant, dataset)]
            if s["mean"] is None:
                row.append("—")
            elif s["complete"]:
                row.append("%.2f ± %.2f (CV %.4f, n=%d)"
                           % (s["mean"], s["std"], s["cv"], s["n"]))
            else:
                row.append("%.2f (不完整, n=%d/%d)"
                           % (s["mean"], s["n"], s["expected_seeds"]))
        detail_rows.append(row)
    lines.append(_markdown_table(["数据集"] + list(analysis.variants), detail_rows))
    lines.append("")

    if charts:
        lines.append("## 图表")
        lines.append("")
        for name, path in charts.items():
            lines.append("- %s: `%s`" % (name, os.path.basename(path)))
        lines.append("")

    lines.append("## 结论边界")
    lines.append("")
    if len(analysis.ready_datasets) == len(analysis.datasets) and analysis.datasets \
            and len(analysis.code_fingerprints) <= 1:
        lines.append(
            "样本完整且代码指纹一致；上表的配对检验可用于版本间比较。"
            "任何结论仍以配对数据集数为准（n=%d）。" % len(analysis.ready_datasets)
        )
    else:
        lines.append(
            "**样本不完整或来源不一致，本报告不给出算法优劣结论。** "
            "描述性均值只基于实际存在的运行；禁止把不完整结果解释为性能差异。"
        )
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    for name, path in paths.items():
        lines.append("- %s: `%s`" % (name, os.path.basename(path)))
    lines.append("")

    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": COLOR_SURFACE,
        "axes.facecolor": COLOR_SURFACE,
        "savefig.facecolor": COLOR_SURFACE,
        "text.color": COLOR_INK,
        "axes.labelcolor": COLOR_SECONDARY,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "axes.edgecolor": COLOR_BASELINE,
        "axes.linewidth": 0.8,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
    })
    return plt


def _relative_matrix(analysis: Analysis) -> Tuple[List[str], List[str], List[List[float]]]:
    variants = [v for v in analysis.variants if v != ac.REFERENCE_VARIANT]
    datasets = list(analysis.ready_datasets)
    matrix = []
    for ds in datasets:
        ref_mean = float(analysis.dataset_stats[(ac.REFERENCE_VARIANT, ds)]["mean"])
        row = []
        for variant in variants:
            value = float(analysis.dataset_stats[(variant, ds)]["mean"])
            row.append(100.0 * (value - ref_mean) / ref_mean if ref_mean else 0.0)
        matrix.append(row)
    return datasets, variants, matrix


def make_charts(analysis: Analysis, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    try:
        plt = _setup_matplotlib()
    except Exception as exc:  # pragma: no cover - environment dependent
        print("[warn] matplotlib unavailable, charts skipped: %s" % exc)
        return {}
    try:
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception as exc:  # pragma: no cover
        print("[warn] chart dependencies unavailable: %s" % exc)
        return {}

    charts: Dict[str, str] = {}
    ready = analysis.ready_datasets
    subtitle = ("paired unit = dataset mean over %d seed%s; n=%d complete datasets"
                % (len(analysis.seeds),
                   "" if len(analysis.seeds) == 1 else "s", len(ready)))

    # 1. Mean relative change vs reference: diverging horizontal bars.
    if analysis.paired:
        rows = list(analysis.paired)
        labels = [str(r["variant"]) for r in rows]
        values = [float(r["mean_change_pct_vs_reference"] or 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
        colors = [COLOR_BLUE if v <= 0 else COLOR_RED for v in values]
        y = range(len(labels))
        ax.barh(list(y), values, color=colors, height=0.52, edgecolor="none")
        ax.axvline(0.0, color=COLOR_BASELINE, linewidth=0.9)
        span = max(0.5, max(abs(x) for x in values))
        for yi, v in zip(y, values):
            offset = 0.035 * span
            ax.text(v + (offset if v >= 0 else -offset), yi,
                    "%+.2f%%" % v, va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=9, color=COLOR_SECONDARY)
        ax.set_xlim(min(0.0, min(values)) - 0.28 * span,
                    max(0.0, max(values)) + 0.28 * span)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean change vs %s (%%) — negative is better"
                      % ac.REFERENCE_VARIANT)
        ax.set_title("Ablation effect on makespan", loc="left", fontsize=12,
                     color=COLOR_INK, pad=22)
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=COLOR_MUTED, va="bottom")
        ax.grid(axis="x", linewidth=0.8, color=COLOR_GRID)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        path = os.path.join(output_dir, "ablation_relative_change.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        charts["relative_change"] = path

    # 2. Distribution of per-dataset change: box plot.
    if ready and analysis.paired:
        datasets, variants, matrix = _relative_matrix(analysis)
        cols = [[matrix[i][j] for i in range(len(datasets))]
                for j in range(len(variants))]
        fig, ax = plt.subplots(figsize=(8.0, 3.9), constrained_layout=True)
        box = ax.boxplot(cols, patch_artist=True, widths=0.42,
                         medianprops={"color": COLOR_INK, "linewidth": 1.2},
                         whiskerprops={"color": COLOR_BASELINE, "linewidth": 0.9},
                         capprops={"color": COLOR_BASELINE, "linewidth": 0.9},
                         flierprops={"marker": "o", "markersize": 3.5,
                                     "markerfacecolor": COLOR_BLUE_LIGHT,
                                     "markeredgecolor": COLOR_SURFACE,
                                     "alpha": 0.9})
        for patch in box["boxes"]:
            patch.set_facecolor(COLOR_BLUE_LIGHT)
            patch.set_edgecolor(COLOR_SURFACE)
            patch.set_linewidth(1.5)
        rng = np.random.default_rng(13)
        for j, col in enumerate(cols, start=1):
            x = j + rng.uniform(-0.11, 0.11, size=len(col))
            ax.scatter(x, col, s=9, color=COLOR_BLUE, alpha=0.55,
                       linewidths=0.0, zorder=3)
        ax.axhline(0.0, color=COLOR_BASELINE, linewidth=0.9)
        ax.set_xticks(range(1, len(variants) + 1))
        ax.set_xticklabels(variants)
        ax.set_ylabel("Change vs %s (%%)" % ac.REFERENCE_VARIANT)
        ax.set_title("Per-dataset ablation effect", loc="left", fontsize=12,
                     pad=22)
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=COLOR_MUTED, va="bottom")
        ax.grid(axis="y", linewidth=0.8, color=COLOR_GRID)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        path = os.path.join(output_dir, "ablation_relative_change_box.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        charts["relative_change_box"] = path

    # 3. Champion counts.
    if ready:
        labels = [str(r["variant"]) for r in analysis.variant_summary]
        counts = [int(r["champion_count"]) for r in analysis.variant_summary]
        fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
        colors = [COLOR_BLUE_DARK if lab == ac.REFERENCE_VARIANT else COLOR_BLUE_LIGHT
                  for lab in labels]
        x = range(len(labels))
        ax.bar(list(x), counts, color=colors, width=0.5, edgecolor="none")
        for xi, c in zip(x, counts):
            ax.text(xi, c + 0.15, str(c), ha="center", va="bottom",
                    fontsize=9, color=COLOR_SECONDARY)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Datasets won (lowest mean makespan)")
        ax.set_title("Instance champions", loc="left", fontsize=12, pad=22)
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=COLOR_MUTED, va="bottom")
        ax.grid(axis="y", linewidth=0.8, color=COLOR_GRID)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(0, max(counts + [1]) * 1.18)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        path = os.path.join(output_dir, "ablation_champion_counts.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        charts["champion_counts"] = path

    # 4. Absolute equal-weight mean makespan: dot plot (not a truncated bar).
    if ready:
        labels = [str(r["variant"]) for r in analysis.variant_summary]
        values = [float(r["equal_weight_mean_makespan"]) for r in analysis.variant_summary]
        fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
        y = list(range(len(labels)))
        for yi, val, lab in zip(y, values, labels):
            color = COLOR_BLUE if lab == ac.REFERENCE_VARIANT else COLOR_MUTED
            ax.plot([min(values), val], [yi, yi], color=COLOR_GRID, linewidth=0.9,
                    zorder=1)
            ax.scatter([val], [yi], s=64, color=color, edgecolor=COLOR_SURFACE,
                       linewidth=1.5, zorder=3)
            ax.text(val, yi - 0.22, "%.1f" % val, ha="center", va="top",
                    fontsize=8.5, color=COLOR_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Equal-weight mean makespan (s) — lower is better")
        ax.set_title("Absolute makespan by variant", loc="left", fontsize=12,
                     pad=22)
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=COLOR_MUTED, va="bottom")
        ax.grid(axis="x", linewidth=0.8, color=COLOR_GRID)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        path = os.path.join(output_dir, "ablation_absolute_mean_dot.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        charts["absolute_mean_dot"] = path

    # 5. Dataset x variant heatmap of relative change (diverging).
    if ready and analysis.paired:
        datasets, variants, matrix = _relative_matrix(analysis)
        data = np.asarray(matrix, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size:
            limit = max(1e-6, float(np.max(np.abs(finite))))
            cmap = LinearSegmentedColormap.from_list(
                "ablation_diverging",
                [COLOR_BLUE_DARK, COLOR_BLUE, COLOR_NEUTRAL, COLOR_RED, "#b02f2f"],
            )
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            fig, ax = plt.subplots(figsize=(7.8, 8.6), constrained_layout=True)
            image = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                              interpolation="nearest")
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(variants, rotation=0)
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels(datasets, fontsize=7.5)
            if data.size <= 400:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        # Text inside a filled cell: white on saturated poles,
                        # ink near the neutral midpoint (WCAG contrast).
                        strong = abs(float(data[i, j])) / limit > 0.55
                        ax.text(j, i, "%.1f" % data[i, j], ha="center",
                                va="center", fontsize=6.5,
                                color="#ffffff" if strong else COLOR_INK)
            ax.set_title("Per-dataset change vs %s (%%)"
                         % ac.REFERENCE_VARIANT, loc="left", fontsize=12)
            cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("negative = lower makespan (better)", fontsize=8)
            cbar.outline.set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            path = os.path.join(output_dir, "ablation_relative_heatmap.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            charts["relative_heatmap"] = path

    return charts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyse completed ACO+Tabu ablation results. Paired unit = one "
            "dataset's mean over seeds. Missing data is reported, never imputed."
        )
    )
    p.add_argument("--results-root", default=os.path.join(STUDY_DIR, "results"))
    p.add_argument("--budget", type=float, default=ac.MAIN_BUDGET)
    p.add_argument("--variants", nargs="+", default=list(ac.VARIANT_KEYS),
                   choices=list(ac.VARIANT_KEYS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(ac.FORMAL_SEEDS))
    p.add_argument("--data-dir", default=None,
                   help="optional benchmark directory defining the expected 30 datasets")
    p.add_argument("--output-dir", default=None,
                   help="default: <results-root>/_analysis/<budget>s")
    p.add_argument("--reference", default=ac.REFERENCE_VARIANT,
                   choices=list(ac.VARIANT_KEYS))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-pairs", type=int, default=5)
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when the expected grid is not fully complete")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.budget <= 0:
        p.error("--budget must be > 0")
    if not (0.0 < args.alpha < 1.0):
        p.error("--alpha must be in (0,1)")
    if args.min_pairs < 1:
        p.error("--min-pairs must be >= 1")
    if args.reference not in args.variants:
        p.error("--reference must be one of --variants")
    for name, values in (("--variants", args.variants), ("--seeds", args.seeds)):
        if len(set(values)) != len(values):
            p.error("%s contains duplicates; each value must appear once" % name)

    if args.data_dir:
        try:
            datasets = [
                os.path.splitext(os.path.basename(x))[0]
                for x in ac.discover_dataset_files(args.data_dir)
            ]
        except ValueError as exc:
            p.error(str(exc))
    else:
        datasets = discover_result_datasets(
            args.results_root, args.budget, args.variants)
        if not datasets:
            print("[error] no result directories found under %s for budget %gs"
                  % (args.results_root, args.budget))
            return 2

    output_dir = args.output_dir or os.path.join(
        os.path.abspath(args.results_root), "_analysis",
        ac.format_budget_tag(args.budget))
    os.makedirs(output_dir, exist_ok=True)

    analysis = build_analysis(
        args.results_root, args.budget, args.variants, datasets, args.seeds,
        reference=args.reference, alpha=args.alpha, min_pairs=args.min_pairs,
    )
    paths = write_tables(analysis, output_dir)
    charts = {} if args.no_charts else make_charts(analysis, output_dir)
    report = write_report(analysis, output_dir, charts, paths)

    manifest = {
        "schema_version": ac.SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "results_root": os.path.abspath(args.results_root),
        "budget": args.budget,
        "variants": analysis.variants,
        "seeds": analysis.seeds,
        "datasets_expected": analysis.datasets,
        "ready_datasets": analysis.ready_datasets,
        "completeness": analysis.completeness,
        "warnings": analysis.warnings,
        "code_fingerprints": analysis.code_fingerprints,
        "alpha": args.alpha,
        "min_pairs": args.min_pairs,
        "charts": {k: os.path.basename(v) for k, v in charts.items()},
        "tables": {k: os.path.basename(v) for k, v in paths.items()},
    }
    manifest_path = os.path.join(output_dir, "analysis_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("===== ablation analysis =====")
    print("datasets expected : %d" % len(analysis.datasets))
    print("complete for all  : %d" % len(analysis.ready_datasets))
    print("cells complete    : %d / %d"
          % (analysis.completeness["complete_cells"],
             analysis.completeness["expected_cells"]))
    for warning in analysis.warnings:
        print("[warning] %s" % warning)
    for row in analysis.variant_summary:
        print("%-12s mean=%s champion=%s"
              % (row["variant"],
                 _fmt(row["equal_weight_mean_makespan"], 3),
                 _fmt(row["champion_count"])))
    print("report  : %s" % report)
    print("manifest: %s" % manifest_path)

    complete = bool(analysis.completeness["full_coverage"]) and \
        len(analysis.code_fingerprints) <= 1
    if args.strict and not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
