"""Paper figures for the five-algorithm convergence experiment."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ALGORITHMS = ("ACO", "GA", "ALNS", "PSO", "VND")
COLORS = {
    "ACO": "#2a78d6",
    "GA": "#eb6834",
    "ALNS": "#1baf7a",
    "PSO": "#eda100",
    "VND": "#e87ba4",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def _save(fig, output_dir: Path, stem: str, dpi: int) -> List[str]:
    paths = []
    for ext in ("png", "pdf"):
        path = output_dir / ("%s.%s" % (stem, ext))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def fig_per_dataset(curves, dataset, output_dir, dpi, config=None):
    panel = curves[curves["dataset"] == dataset]
    if panel.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for alg in ALGORITHMS:
        sub = panel[panel["algorithm"] == alg].sort_values("time_s")
        if sub.empty:
            continue
        x = sub["time_s"].to_numpy(float)
        mean = sub["mean_makespan_s"].to_numpy(float)
        low = sub["ci95_low_makespan_s"].to_numpy(float)
        high = sub["ci95_high_makespan_s"].to_numpy(float)
        ax.plot(x, mean, color=COLORS[alg], label=alg)
        valid = np.isfinite(low) & np.isfinite(high)
        if valid.any():
            ax.fill_between(x, low, high, color=COLORS[alg], alpha=0.10,
                            linewidth=0)
    ax.set_xlabel("Real wall-clock time (s)")
    ax.set_ylabel("Best feasible makespan (s)")
    ax.set_title("Convergence on %s" % dataset)
    ax.legend(ncol=5, loc="upper right")
    return _save(fig, output_dir, "fig%02d_%s_convergence" % (
        _dataset_index(dataset, config), dataset.replace("/", "_")
    ), dpi)


def _dataset_index(dataset: str, config=None) -> int:
    if config:
        names = [str(d["name"]) for d in config.get("datasets", [])]
        if dataset in names:
            return names.index(dataset) + 1
    return 1


def fig_normalized(curves, config, output_dir, dpi):
    if curves is None or curves.empty:
        return None
    datasets = [str(d["name"]) for d in config["datasets"]]
    nrows = max(1, math.ceil(len(datasets) / 3.0))
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 3.0 * nrows + 1.0),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    used = 0
    for ax, dataset in zip(axes, datasets):
        panel = curves[curves["dataset"] == dataset]
        if panel.empty:
            ax.axis("off")
            continue
        used += 1
        for alg in ALGORITHMS:
            sub = panel[panel["algorithm"] == alg].sort_values("time_s")
            if sub.empty:
                continue
            ax.plot(sub["time_s"], sub["mean_gap_pct"], color=COLORS[alg], label=alg)
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.set_title(dataset)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gap to experiment BKS (%)")
    for ax in axes[len(datasets):]:
        ax.axis("off")
    if used:
        axes[0].legend(ncol=5, loc="upper right")
    fig.suptitle("Normalized mean convergence (gap to best known %g s solution)"
                 % float(config.get("time_budget_seconds", 60.0)),
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, output_dir, "fig06_normalized_gap", dpi)


def fig_gap_at_times(curves, config, output_dir, dpi):
    if curves is None or curves.empty:
        return None
    nodes = [float(x) for x in config.get("report_time_nodes", [1, 5, 10, 20, 30, 40, 50, 60])]
    if curves.empty:
        return None
    # Mean over datasets at each (algorithm, node).
    values = {alg: [] for alg in ALGORITHMS}
    errors = {alg: [] for alg in ALGORITHMS}
    for alg in ALGORITHMS:
        for node in nodes:
            per_dataset = []
            for dataset in curves["dataset"].unique():
                sub = curves[(curves["dataset"] == dataset) &
                             (curves["algorithm"] == alg)]
                if sub.empty:
                    continue
                idx = int(np.argmin(np.abs(sub["time_s"].to_numpy(float) - node)))
                gap = sub["mean_gap_pct"].to_numpy(float)[idx]
                if math.isfinite(gap):
                    per_dataset.append(gap)
            values[alg].append(float(np.mean(per_dataset)) if per_dataset else np.nan)
            if len(per_dataset) >= 2:
                s = float(np.std(per_dataset, ddof=1))
                errors[alg].append(1.96 * s / math.sqrt(len(per_dataset)))
            else:
                errors[alg].append(0.0)
    x = np.arange(len(nodes))
    width = 0.8 / len(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    for i, alg in enumerate(ALGORITHMS):
        ax.bar(x + (i - 2) * width, values[alg], width * 0.92,
               yerr=errors[alg], color=COLORS[alg], label=alg,
               error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 2})
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["%gs" % n for n in nodes])
    ax.set_xlabel("Time node")
    ax.set_ylabel("Mean gap to experiment BKS (%)")
    ax.set_title("Gap at specified times (mean over datasets; 95% CI across datasets)")
    ax.legend(ncol=5, loc="upper right")
    return _save(fig, output_dir, "fig07_gap_at_times", dpi)


def fig_final_distribution(run_summary, final_summary, config, output_dir, dpi):
    if run_summary is None or run_summary.empty:
        return None
    budget = float(config.get("time_budget_seconds", 60.0))
    datasets = [str(d["name"]) for d in config["datasets"]]
    nrows = max(1, math.ceil(len(datasets) / 3.0))
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 3.0 * nrows + 1.0),
                             sharey=False)
    axes = np.asarray(axes).ravel()
    used = 0
    for ax, dataset in zip(axes, datasets):
        panel = run_summary[run_summary["dataset"] == dataset] if not run_summary.empty else pd.DataFrame()
        if panel.empty:
            ax.axis("off")
            continue
        used += 1
        data = []
        labels = []
        colors = []
        for alg in ALGORITHMS:
            vals = panel[panel["algorithm"] == alg]["final_makespan_s"].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(alg)
            colors.append(COLORS[alg])
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops={"color": INK, "linewidth": 1.1},
                        whiskerprops={"color": AXIS, "linewidth": 0.8},
                        capprops={"color": AXIS, "linewidth": 0.8},
                        flierprops={"marker": "o", "markersize": 2.5,
                                    "markerfacecolor": MUTED,
                                    "markeredgecolor": MUTED, "alpha": 0.6})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(dataset)
        ax.set_ylabel("Makespan at %g s (s)" % budget)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    if used:
        fig.suptitle("Distribution of %g s final makespan over %d seeds"
                     % (budget, len(config.get("seeds", []))),
                     fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, output_dir, "fig08_final_distribution", dpi)


def fig_target_times(targets, config, output_dir, dpi):
    if targets.empty:
        return None
    datasets = [str(d["name"]) for d in config["datasets"]]
    target_values = [float(x) for x in config.get("target_gap_percent", [1.0, 2.0, 5.0])]
    nrows = max(1, math.ceil(len(datasets) / 3.0))
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 3.0 * nrows + 1.0),
                             sharex=True, sharey=False)
    axes = np.asarray(axes).ravel()
    used = 0
    width = 0.8 / len(ALGORITHMS)
    x = np.arange(len(target_values))
    for ax, dataset in zip(axes, datasets):
        panel = targets[targets["dataset"] == dataset]
        if panel.empty:
            ax.axis("off")
            continue
        used += 1
        for i, alg in enumerate(ALGORITHMS):
            vals = []
            for target in target_values:
                r = panel[(panel["algorithm"] == alg) &
                          np.isclose(panel["target_gap_pct"], target)]
                vals.append(float(r["mean_time_to_target_s"].iloc[0])
                            if not r.empty and pd.notna(r["mean_time_to_target_s"].iloc[0])
                            else np.nan)
            ax.bar(x + (i - 2) * width, vals, width * 0.92,
                   color=COLORS[alg], label=alg)
        ax.set_xticks(x)
        ax.set_xticklabels(["%g%%" % t for t in target_values])
        ax.set_title(dataset)
        ax.set_xlabel("Target gap to experiment BKS")
        ax.set_ylabel("Mean time to reach (s)")
        ax.set_ylim(bottom=0)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    if used:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[a]) for a in ALGORITHMS]
        fig.legend(handles, list(ALGORITHMS), loc="lower center", ncol=5,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Time to reach target solution quality (censored runs excluded)",
                     fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, output_dir, "fig09_target_times", dpi)


def make_all_plots(curves, final_summary, targets, config, output_dir, dpi=300,
                   run_summary=None):
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for dataset in [str(d["name"]) for d in config["datasets"]]:
        result = fig_per_dataset(curves, dataset, output_dir, dpi, config)
        if result:
            generated.extend(result)
    for job in (
        lambda: fig_normalized(curves, config, output_dir, dpi),
        lambda: fig_gap_at_times(curves, config, output_dir, dpi),
        lambda: fig_final_distribution(run_summary if run_summary is not None else pd.DataFrame(),
                                       final_summary, config, output_dir, dpi),
        lambda: fig_target_times(targets, config, output_dir, dpi),
    ):
        result = job()
        if result:
            generated.extend(result)
    (output_dir / "figure_manifest.json").write_text(
        __import__("json").dumps({"generated": generated, "dpi": int(dpi)}, indent=2),
        encoding="utf-8",
    )
    return generated


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate convergence figures from analysis CSVs.")
    p.add_argument("--analysis-dir", default=str(_HERE / "analysis"))
    p.add_argument("--figures-dir", default=str(_HERE / "figures"))
    p.add_argument("--config", default=str(_HERE / "config" / "convergence_config.json"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    curves = _read(Path(args.analysis_dir) / "convergence_curves.csv")
    final_summary = _read(Path(args.analysis_dir) / "convergence_final_summary.csv")
    targets = _read(Path(args.analysis_dir) / "convergence_target_times.csv")
    run_summary = _read(Path(args.analysis_dir) / "convergence_run_summary.csv")
    if curves.empty:
        print("No formal convergence curves found. Run analyze_convergence.py first.",
              file=sys.stderr)
        return 2
    import json
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    generated = make_all_plots(
        curves, final_summary, targets, config, Path(args.figures_dir),
        args.dpi, run_summary=run_summary,
    )
    print("Generated %d figure file(s) in %s" % (len(generated), args.figures_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
