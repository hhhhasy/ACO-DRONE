"""Paper figures for the drone resource sensitivity experiment."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
MARKERS = {"ACO": "o", "GA": "s", "ALNS": "^", "PSO": "D", "VND": "v"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
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


def _panel(summary: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if summary.empty or "dimension" not in summary.columns:
        return pd.DataFrame()
    return summary[summary["dimension"] == dimension].copy()


def _series(panel: pd.DataFrame, alg: str, value: str):
    sub = panel[panel["algorithm"] == alg].sort_values("level")
    return sub["level"].to_numpy(float), sub[value].to_numpy(float), sub["n_quality"].to_numpy(float)


def _mean_plot(summary, config, dimension, value, ylabel, title, stem, output_dir, dpi):
    panel = _panel(summary, dimension)
    if panel.empty or panel[value].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for alg in ALGORITHMS:
        x, y, n = _series(panel, alg, value)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=COLORS[alg], marker=MARKERS[alg], markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=alg)
    ax.set_xlabel("Battery capacity (Wh)" if dimension == "battery"
                  else "Maximum drone payload (kg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=5, loc="best")
    return _save(fig, output_dir, stem, dpi)


def _errorbar_plot(summary, dimension, value, std_col, ylabel, title, stem, output_dir, dpi):
    panel = _panel(summary, dimension)
    if panel.empty or panel[value].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for alg in ALGORITHMS:
        sub = panel[panel["algorithm"] == alg].sort_values("level")
        if sub.empty:
            continue
        x = sub["level"].to_numpy(float)
        y = sub[value].to_numpy(float)
        n = sub["n_quality"].to_numpy(float)
        std = sub[std_col].to_numpy(float)
        sem = np.where(n > 1, 1.96 * std / np.sqrt(np.maximum(n, 1)), 0.0)
        ax.errorbar(x, y, yerr=sem, color=COLORS[alg], marker=MARKERS[alg],
                    markersize=4.5, capsize=2.5, elinewidth=0.9,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=alg)
    ax.set_xlabel("Battery capacity (Wh)" if dimension == "battery"
                  else "Maximum drone payload (kg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=5, loc="best")
    return _save(fig, output_dir, stem, dpi)


def _change_plot(changes, output_dir, dpi):
    if changes.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
    for ax, dim, title in [
        (axes[0], "battery", "Battery capacity"),
        (axes[1], "payload", "Maximum payload"),
    ]:
        panel = changes[changes["dimension"] == dim]
        if panel.empty:
            ax.set_visible(False)
            continue
        for alg in ALGORITHMS:
            sub = panel[panel["algorithm"] == alg].sort_values("level")
            if sub.empty:
                continue
            ax.plot(sub["level"], sub["percent_change"], color=COLORS[alg],
                    marker=MARKERS[alg], markersize=4.5, label=alg,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.set_xlabel("Battery capacity (Wh)" if dim == "battery"
                      else "Maximum payload (kg)")
        ax.set_title(title)
        ax.set_ylabel("Makespan change vs baseline (%)")
    axes[0].legend(ncol=5, loc="best")
    fig.suptitle("Mean makespan change relative to 200 Wh / 3 kg", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, output_dir, "fig05_baseline_change", dpi)


def make_all_plots(summary, changes, rankings, config, output_dir, dpi=300):
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    jobs = [
        (lambda: _errorbar_plot(
            summary, "battery", "mean_makespan_s", "std_makespan_s",
            "Mean makespan (s)", "Battery capacity vs makespan (payload = %g kg)"
            % config["baseline"]["payload_kg"], "fig01_battery_makespan",
            output_dir, dpi)),
        (lambda: _errorbar_plot(
            summary, "payload", "mean_makespan_s", "std_makespan_s",
            "Mean makespan (s)", "Payload capacity vs makespan (battery = %g Wh)"
            % config["baseline"]["battery_wh"], "fig02_payload_makespan",
            output_dir, dpi)),
        (lambda: _errorbar_plot(
            summary, "battery", "mean_drone_share_pct", "std_makespan_s",
            "Drone-served customers (%)", "Battery capacity vs drone service share",
            "fig03_battery_drone_share", output_dir, dpi)),
        (lambda: _errorbar_plot(
            summary, "payload", "mean_drone_share_pct", "std_makespan_s",
            "Drone-served customers (%)", "Payload capacity vs drone service share",
            "fig04_payload_drone_share", output_dir, dpi)),
        (lambda: _change_plot(changes, output_dir, dpi)),
    ]
    # The drone-share plots should not use makespan std as an error bar.
    for index in (2, 3):
        jobs[index] = (lambda idx=index: _simple_line(
            summary, "battery" if idx == 2 else "payload", "mean_drone_share_pct",
            "Drone-served customers (%)",
            "Battery capacity vs drone service share" if idx == 2
            else "Payload capacity vs drone service share",
            "fig03_battery_drone_share" if idx == 2 else "fig04_payload_drone_share",
            output_dir, dpi))
    for job in jobs:
        result = job()
        if result:
            generated.extend(result)
    (output_dir / "figure_manifest.json").write_text(
        __import__("json").dumps({"generated": generated, "dpi": int(dpi)}, indent=2),
        encoding="utf-8",
    )
    return generated


def _simple_line(summary, dimension, value, ylabel, title, stem, output_dir, dpi):
    panel = _panel(summary, dimension)
    if panel.empty or panel[value].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for alg in ALGORITHMS:
        x, y, _ = _series(panel, alg, value)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=COLORS[alg], marker=MARKERS[alg], markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=alg)
    ax.set_xlabel("Battery capacity (Wh)" if dimension == "battery"
                  else "Maximum drone payload (kg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=5, loc="best")
    return _save(fig, output_dir, stem, dpi)


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate sensitivity figures from analysis CSVs.")
    p.add_argument("--analysis-dir", default=None,
                   help="default: config.analysis_dir (or analysis)")
    p.add_argument("--figures-dir", default=None,
                   help="default: config.figures_dir (or figures)")
    p.add_argument("--config", default=str(_HERE / "config" / "sensitivity_config.json"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    import json
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = Path(args.config).resolve().parent.parent

    def resolve(value, fallback):
        p = Path(str(value or fallback)).expanduser()
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    analysis_dir = resolve(args.analysis_dir, config.get("analysis_dir", "analysis"))
    figures_dir = resolve(args.figures_dir, config.get("figures_dir", "figures"))
    summary = _read(analysis_dir / "sensitivity_summary.csv")
    changes = _read(analysis_dir / "sensitivity_baseline_changes.csv")
    rankings = _read(analysis_dir / "sensitivity_rankings.csv")
    if summary.empty:
        print("No formal sensitivity summary found. Run analyze_sensitivity.py first.",
              file=sys.stderr)
        return 2
    generated = make_all_plots(
        summary, changes, rankings, config, figures_dir, args.dpi
    )
    print("Generated %d figure file(s) in %s" % (len(generated), figures_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
