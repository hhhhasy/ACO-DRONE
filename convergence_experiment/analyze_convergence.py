"""Analysis of the five-algorithm convergence experiment.

Reads only ``results/runs/*.json`` and ``history/*.csv``.  It never imports or
runs a solver.  Curves are aligned to the configured real-time grid by forward
holding the last known best feasible makespan; no improvement is interpolated.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ALGORITHMS = ("ACO", "GA", "ALNS", "PSO", "VND")
HISTORY_FIELDS = [
    "algorithm", "dataset", "seed", "time_budget_seconds",
    "elapsed_time_seconds", "best_makespan_s", "iteration_or_generation",
    "evaluations", "event_type",
]
KEY = ["algorithm", "dataset", "seed", "time_budget_seconds"]


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(v, d=3) -> str:
    x = _num(v)
    return "NA" if x is None else (("%." + str(d) + "f") % x)


def _fmt_pct(v, d=2) -> str:
    x = _num(v)
    return "NA" if x is None else (("%." + str(d) + "f") % x) + "%"


def _write_csv(df: pd.DataFrame, path: Path, fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if fields is not None:
        for f in fields:
            if f not in out.columns:
                out[f] = np.nan
        out = out[list(fields)]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _md_table(headers, rows) -> str:
    if not rows:
        return "_（无数据）_\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _latex_escape(v) -> str:
    s = str(v)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in s)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_project_path(value: str) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = _HERE / p
    return p.resolve()


def time_grid(config: Dict[str, Any]) -> np.ndarray:
    grid = config.get("time_grid", {})
    start = float(grid.get("start", 0.0))
    stop = float(grid.get("stop", config.get("time_budget_seconds", 60.0)))
    step = float(grid.get("step", 0.5))
    if step <= 0:
        raise ValueError("time grid step must be positive")
    return np.round(np.arange(start, stop + step * 0.5, step), 10)


def load_runs(input_dir: Path) -> pd.DataFrame:
    runs_dir = input_dir / "runs"
    records = []
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            payload["_json_path"] = str(path)
            records.append(payload)
    if not records:
        manifest = input_dir / "runs.csv"
        if manifest.is_file():
            df = pd.read_csv(manifest, encoding="utf-8-sig")
            df["_json_path"] = ""
            records = df.to_dict("records")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col, default in [
        ("status", "ok"), ("algorithm", ""), ("dataset", ""), ("seed", np.nan),
        ("time_budget_seconds", np.nan), ("best_makespan_s", np.nan),
        ("feasible", False), ("actual_runtime_seconds", np.nan),
        ("history_file", ""), ("history_events", np.nan),
        ("first_feasible_time_s", np.nan), ("last_improvement_time_s", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    df["algorithm"] = df["algorithm"].astype(str).str.upper()
    df["dataset"] = df["dataset"].astype(str)
    df["status"] = df["status"].astype(str).str.lower()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    for col in ["time_budget_seconds", "best_makespan_s", "actual_runtime_seconds",
                "history_events", "first_feasible_time_s", "last_improvement_time_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible_flag"] = df["feasible"].map(
        lambda x: bool(x) if isinstance(x, (bool, np.bool_))
        else str(x).lower() in {"1", "true", "yes", "y"}
    )
    return df


def _dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    dup = int(df.duplicated(subset=KEY, keep=False).sum())
    if dup:
        df = df.sort_values(KEY + ["status"]).drop_duplicates(subset=KEY, keep="last")
    return df, dup


def load_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if df.empty or not set(HISTORY_FIELDS).issubset(df.columns):
        return pd.DataFrame()
    for col in ["seed", "time_budget_seconds", "elapsed_time_seconds",
                "best_makespan_s", "iteration_or_generation", "evaluations"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        df["elapsed_time_seconds"].notna() &
        df["best_makespan_s"].notna() &
        np.isfinite(df["elapsed_time_seconds"]) &
        np.isfinite(df["best_makespan_s"]) &
        (df["best_makespan_s"] > 0)
    ].copy()
    if df.empty:
        return df
    df = df.sort_values("elapsed_time_seconds")
    df = df.drop_duplicates(subset=["elapsed_time_seconds"], keep="last")
    df["best_makespan_s"] = df["best_makespan_s"].cummin()
    return df


def align_curve(history: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Forward-hold the last best value; NaN before the first feasible event."""
    if history.empty:
        return np.full(len(grid), np.nan)
    times = history["elapsed_time_seconds"].to_numpy(float)
    values = history["best_makespan_s"].to_numpy(float)
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def _curve_stats(curves: np.ndarray) -> Dict[str, float]:
    """curves shape (n_runs, n_times); NaN where no feasible solution yet."""
    n_times = curves.shape[1]
    mean = np.full(n_times, np.nan)
    median = np.full(n_times, np.nan)
    std = np.full(n_times, np.nan)
    low = np.full(n_times, np.nan)
    high = np.full(n_times, np.nan)
    n = np.zeros(n_times, dtype=int)
    for j in range(n_times):
        vals = curves[:, j]
        vals = vals[np.isfinite(vals)]
        n[j] = len(vals)
        if not len(vals):
            continue
        mean[j] = float(np.mean(vals))
        median[j] = float(np.median(vals))
        if len(vals) >= 2:
            s = float(np.std(vals, ddof=1))
            std[j] = s
            half = float(scipy_stats.t.ppf(0.975, len(vals) - 1)) * s / math.sqrt(len(vals))
            low[j] = mean[j] - half
            high[j] = mean[j] + half
        else:
            std[j] = 0.0
            low[j] = high[j] = mean[j]
    return {"mean": mean, "median": median, "std": std, "ci95_low": low,
            "ci95_high": high, "n": n}


def _run_summary(history: pd.DataFrame, grid: np.ndarray, curve: np.ndarray,
                 nodes: Sequence[float]) -> Dict[str, Any]:
    first = float(history["elapsed_time_seconds"].iloc[0]) if not history.empty else np.nan
    if not history.empty:
        vals = history["best_makespan_s"].to_numpy(float)
        times = history["elapsed_time_seconds"].to_numpy(float)
        decreases = np.where(np.diff(vals) < -1e-9)[0] + 1
        improvements = int(len(decreases))
        last_imp = float(times[decreases[-1]]) if len(decreases) else np.nan
    else:
        improvements = 0
        last_imp = np.nan
    final_value = float(curve[-1]) if len(curve) and np.isfinite(curve[-1]) else None
    values = {}
    for t in nodes:
        idx = int(np.argmin(np.abs(grid - float(t))))
        values["best_at_%gs" % float(t)] = (
            float(curve[idx]) if np.isfinite(curve[idx]) else None
        )
    return {
        "first_feasible_time_s": first if math.isfinite(first) else None,
        "improvements": improvements,
        "last_improvement_time_s": last_imp if math.isfinite(last_imp) else None,
        "final_makespan_s": final_value,
        **values,
    }


def _final_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, alg), group in final_df.groupby(["dataset", "algorithm"]):
        vals = group["final_makespan_s"].dropna().to_numpy(float)
        firsts = group["first_feasible_time_s"].dropna().to_numpy(float)
        improvements = group["improvements"].dropna().to_numpy(float)
        last_imps = group["last_improvement_time_s"].dropna().to_numpy(float)
        if len(vals):
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
            half = (float(scipy_stats.t.ppf(0.975, len(vals) - 1)) * std /
                    math.sqrt(len(vals))) if len(vals) >= 2 else 0.0
            best = float(np.min(vals))
            worst = float(np.max(vals))
        else:
            mean = std = half = best = worst = np.nan
        rows.append({
            "dataset": dataset, "algorithm": alg,
            "n_runs": len(group), "n_final": len(vals),
            "mean_final_makespan_s": mean,
            "std_final_makespan_s": std,
            "ci95_low_final": mean - half if math.isfinite(mean) else np.nan,
            "ci95_high_final": mean + half if math.isfinite(mean) else np.nan,
            "best_final_makespan_s": best,
            "worst_final_makespan_s": worst,
            "mean_first_feasible_time_s": float(np.mean(firsts)) if len(firsts) else np.nan,
            "median_first_feasible_time_s": float(np.median(firsts)) if len(firsts) else np.nan,
            "mean_improvements": float(np.mean(improvements)) if len(improvements) else np.nan,
            "mean_last_improvement_time_s": float(np.mean(last_imps)) if len(last_imps) else np.nan,
        })
    return pd.DataFrame(rows)


def _target_times(final_df: pd.DataFrame, gap_curves: Dict[Tuple[str, str, int], np.ndarray],
                  grid: np.ndarray, targets: Sequence[float]) -> pd.DataFrame:
    bks = final_df.groupby("dataset")["final_makespan_s"].min().to_dict()
    rows = []
    for dataset in sorted(set(k[0] for k in gap_curves)):
        for alg in ALGORITHMS:
            runs = [k for k in gap_curves if k[0] == dataset and k[1] == alg]
            if not runs:
                continue
            for target in targets:
                times = []
                censored = 0
                for key in runs:
                    gaps = gap_curves[key]
                    hit = np.where(np.isfinite(gaps) & (gaps <= float(target) + 1e-9))[0]
                    if len(hit):
                        times.append(float(grid[hit[0]]))
                    else:
                        censored += 1
                rows.append({
                    "dataset": dataset, "algorithm": alg, "target_gap_pct": float(target),
                    "n_runs": len(runs), "n_reached": len(times), "n_censored": censored,
                    "mean_time_to_target_s": float(np.mean(times)) if times else np.nan,
                    "median_time_to_target_s": float(np.median(times)) if times else np.nan,
                    "min_time_to_target_s": float(np.min(times)) if times else np.nan,
                    "max_time_to_target_s": float(np.max(times)) if times else np.nan,
                    "bks_makespan_s": bks.get(dataset, np.nan),
                })
    return pd.DataFrame(rows)


def _missing(runs: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    expected = set()
    for ds in config["datasets"]:
        for alg in config["algorithms"]:
            for seed in config["seeds"]:
                expected.add((alg.upper(), str(ds["name"]), int(seed)))
    present = set()
    if not runs.empty:
        present = set(
            (str(r.algorithm), str(r.dataset), int(r.seed))
            for r in runs.itertuples()
            if pd.notna(r.seed)
        )
    for alg, dataset, seed in sorted(expected - present):
        rows.append({"algorithm": alg, "dataset": dataset, "seed": seed,
                     "reason": "missing"})
    if not runs.empty:
        for _, row in runs[~runs["status"].eq("ok")].iterrows():
            rows.append({"algorithm": row["algorithm"], "dataset": row["dataset"],
                         "seed": row["seed"], "reason": "failed",
                         "error": row.get("error", "")})
    return pd.DataFrame(rows)


def _latex_tables(final_summary: pd.DataFrame, stats_at_times: pd.DataFrame,
                  targets: pd.DataFrame, budget: float) -> str:
    lines = ["% Generated by analyze_convergence.py; requires booktabs.",
             "% \\input{convergence_tables.tex}", ""]
    if not final_summary.empty:
        datasets = sorted(final_summary["dataset"].unique())
        lines += [
            "\\begin{table}[htbp]", "\\centering",
            "\\caption{Mean makespan at %g s (mean $\\pm$ sample std).}"
            % float(budget),
            "\\label{tab:convergence-final}",
            "\\begin{tabular}{l%s}" % ("r" * len(datasets)), "\\toprule",
            "Algorithm & " + " & ".join(_latex_escape(d) for d in datasets) + " \\\\",
            "\\midrule",
        ]
        for alg in ALGORITHMS:
            cells = []
            for ds in datasets:
                r = final_summary[(final_summary["dataset"] == ds) &
                                  (final_summary["algorithm"] == alg)]
                if r.empty or pd.isna(r["mean_final_makespan_s"].iloc[0]):
                    cells.append("NA")
                else:
                    cells.append("%s $\\pm$ %s" % (
                        _fmt(r["mean_final_makespan_s"].iloc[0], 1),
                        _fmt(r["std_final_makespan_s"].iloc[0], 1),
                    ))
            lines.append("%s & %s \\\\" % (alg, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if not targets.empty:
        lines += [
            "\\begin{table}[htbp]", "\\centering",
            "\\caption{Mean time (s) to reach a gap target; censored runs are counted in the CSV.}",
            "\\label{tab:convergence-target-times}",
            "\\begin{tabular}{llrrr}", "\\toprule",
            "Dataset & Algorithm & 1\\% & 2\\% & 5\\% \\\\", "\\midrule",
        ]
        for (ds, alg), group in targets.groupby(["dataset", "algorithm"]):
            cells = []
            for target in (1.0, 2.0, 5.0):
                r = group[np.isclose(group["target_gap_pct"], target)]
                cells.append(_fmt(r["mean_time_to_target_s"].iloc[0], 1)
                             if not r.empty else "NA")
            lines.append("%s & %s & %s \\\\" % (_latex_escape(ds), alg, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines) + "\n"


def _report(config, runs, final_summary, stats_at_times, targets, missing,
            history_missing, duplicates) -> str:
    budget = float(config["time_budget_seconds"])
    n_seeds = len(config["seeds"])
    lines = ["# 五算法收敛性实验报告\n"]
    lines.append("生成时间（UTC）：%s\n" % datetime.now(timezone.utc).isoformat())
    lines.append("数据集：%s\n" % ", ".join(d["name"] for d in config["datasets"]))
    lines.append("算法：%s\n" % ", ".join(config["algorithms"]))
    lines.append("随机种子：%s\n" % ", ".join(str(s) for s in config["seeds"]))
    lines.append("时间预算：`%g s`；物理参数：%g Wh / %g kg。\n" % (
        float(config["time_budget_seconds"]),
        float(config.get("physics", {}).get("battery_wh", 200.0)),
        float(config.get("physics", {}).get("payload_kg", 3.0)),
    ))
    lines += ["## 数据完整性\n"]
    n_ok = int(runs["status"].eq("ok").sum()) if not runs.empty else 0
    lines.append("- 运行记录：%d" % len(runs))
    lines.append("- 成功运行：%d" % n_ok)
    lines.append("- 重复键：%d" % duplicates)
    lines.append("- 缺失/失败记录：%d" % len(missing))
    lines.append("- 缺少收敛历史的成功运行：%d\n" % history_missing)
    if runs.empty:
        lines += [
            "## 尚无正式结果\n",
            "`results/` 与 `history/` 中没有正式收敛实验记录。本程序没有运行任何算法，"
            "也没有生成任何曲线或统计结论。请先手动执行 `run_convergence.py --execute`。\n",
        ]
        return "\n".join(lines)

    lines += [
        "## %g 秒最终结果\n" % budget,
        "下表为每个数据集、每个算法在 %d 个种子上的 %g s 历史最优 Makespan。\n"
        % (n_seeds, budget),
    ]
    if not final_summary.empty:
        headers = ["数据集", "算法", "均值", "标准差", "95% CI", "首次可行 s", "改善次数"]
        rows = [[
            r.dataset, r.algorithm, _fmt(r.mean_final_makespan_s, 1),
            _fmt(r.std_final_makespan_s, 1),
            "[%s, %s]" % (_fmt(r.ci95_low_final, 1), _fmt(r.ci95_high_final, 1)),
            _fmt(r.mean_first_feasible_time_s, 2), _fmt(r.mean_improvements, 1),
        ] for r in final_summary.itertuples()]
        lines.append(_md_table(headers, rows))
    lines += [
        "\n## 收敛行为\n",
        "曲线使用真实墙钟时间，按 0.5 s 统一网格前向保持；首次可行解之前保留缺失状态，"
        "不做线性插值。归一化 Gap 的参考 BKS 是本次实验中该实例的 %g s 最佳已知值，"
        "不是数学最优值。\n" % budget,
    ]
    if not targets.empty:
        headers = ["数据集", "算法", "目标 Gap", "达到次数", "未达到", "平均时间 s"]
        rows = [[
            r.dataset, r.algorithm, _fmt_pct(r.target_gap_pct, 1),
            int(r.n_reached), int(r.n_censored), _fmt(r.mean_time_to_target_s, 2),
        ] for r in targets.itertuples()]
        lines.append(_md_table(headers, rows))
    lines += [
        "\n## 说明与限制\n",
        "- 所有历史点来自各算法内部的真实评价器调用；未使用不同预算终值拼接曲线。",
        "- 历史只在首次可行解和严格改善时写入；不可行候选不会进入历史。",
        "- 时间预算包含数据加载与初始化；各算法按各自剩余预算搜索。",
        "- 本报告只分析已完成的正式运行；缺失/失败运行见 convergence_missing_runs.csv。",
        "- 若某算法在 %g s 内没有可行解，其曲线在首次可行前保持缺失，统计时不计入均值。"
        % budget,
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    results_dir: str | Path,
    history_dir: str | Path,
    output_dir: str | Path,
    figures_dir: str | Path,
    config_path: str | Path,
    make_plots: bool = True,
    dpi: int = 300,
    quiet: bool = False,
):
    config = load_config(Path(config_path))
    results_dir = Path(results_dir).resolve()
    history_dir = Path(history_dir).resolve()
    output_dir = Path(output_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    grid = time_grid(config)
    nodes = [float(x) for x in config.get("report_time_nodes", [1, 5, 10, 20, 30, 40, 50, 60])]

    runs = load_runs(results_dir)
    duplicates = 0
    if not runs.empty:
        runs, duplicates = _dedupe(runs)
    missing = _missing(runs, config) if not runs.empty else pd.DataFrame(
        columns=["algorithm", "dataset", "seed", "reason", "error"]
    )

    curves_by_run: Dict[Tuple[str, str, int], np.ndarray] = {}
    run_rows = []
    history_missing = 0
    if not runs.empty:
        for row in runs.itertuples():
            if str(row.status) != "ok" or not bool(row.feasible_flag):
                continue
            history_path = Path(str(row.history_file)) if str(row.history_file) else None
            if history_path is None or not history_path.is_file():
                # Portable fallback if the project directory was moved after the run.
                history_path = history_dir / (str(getattr(row, "run_id", "")) + ".csv")
            hist = load_history(history_path) if history_path is not None else pd.DataFrame()
            if hist.empty:
                history_missing += 1
                continue
            curve = align_curve(hist, grid)
            key = (str(row.dataset), str(row.algorithm), int(row.seed))
            curves_by_run[key] = curve
            summary = _run_summary(hist, grid, curve, nodes)
            run_rows.append({
                "dataset": str(row.dataset), "algorithm": str(row.algorithm),
                "seed": int(row.seed), "status": str(row.status),
                "history_events": int(len(hist)),
                **summary,
            })
    run_summary = pd.DataFrame(run_rows)

    # Per dataset/algorithm curve statistics and gaps.
    curve_rows = []
    gap_curves: Dict[Tuple[str, str, int], np.ndarray] = {}
    final_df = run_summary.copy() if not run_summary.empty else pd.DataFrame()
    datasets = [str(d["name"]) for d in config["datasets"]]
    for dataset in datasets:
        keys = [k for k in curves_by_run if k[0] == dataset]
        if not keys:
            continue
        finals = [
            curves_by_run[k][-1] for k in keys
            if np.isfinite(curves_by_run[k][-1])
        ]
        bks = float(min(finals)) if finals else float("nan")
        for alg in ALGORITHMS:
            alg_keys = [k for k in keys if k[1] == alg]
            if not alg_keys:
                continue
            curves = np.vstack([curves_by_run[k] for k in alg_keys])
            stats = _curve_stats(curves)
            for j, t in enumerate(grid):
                curve_rows.append({
                    "dataset": dataset, "algorithm": alg, "time_s": float(t),
                    "n_runs": int(stats["n"][j]),
                    "mean_makespan_s": float(stats["mean"][j]),
                    "median_makespan_s": float(stats["median"][j]),
                    "std_makespan_s": float(stats["std"][j]),
                    "ci95_low_makespan_s": float(stats["ci95_low"][j]),
                    "ci95_high_makespan_s": float(stats["ci95_high"][j]),
                    "bks_makespan_s": bks,
                    "mean_gap_pct": (
                        100.0 * (stats["mean"][j] - bks) / bks
                        if math.isfinite(bks) and bks > 0 and math.isfinite(stats["mean"][j])
                        else np.nan
                    ),
                })
            if math.isfinite(bks) and bks > 0:
                for key in alg_keys:
                    gap_curves[key] = 100.0 * (curves_by_run[key] - bks) / bks

    curves = pd.DataFrame(curve_rows)
    final_summary = _final_summary(final_df) if not final_df.empty else pd.DataFrame()
    targets = _target_times(
        final_df, gap_curves, grid,
        [float(x) for x in config.get("target_gap_percent", [1.0, 2.0, 5.0])],
    ) if not final_df.empty else pd.DataFrame()
    stats_at_times = (
        curves[np.isclose(
            pd.to_numeric(curves["time_s"], errors="coerce").to_numpy()[:, None],
            np.asarray(nodes)[None, :]
        ).any(axis=1)] if not curves.empty else pd.DataFrame()
    )
    if not stats_at_times.empty:
        stats_at_times = stats_at_times.copy()
        stats_at_times["time_s"] = stats_at_times["time_s"].round(6)

    _write_csv(runs, output_dir / "convergence_runs.csv")
    _write_csv(run_summary, output_dir / "convergence_run_summary.csv")
    _write_csv(curves, output_dir / "convergence_curves.csv")
    _write_csv(stats_at_times, output_dir / "convergence_stats_at_times.csv")
    _write_csv(final_summary, output_dir / "convergence_final_summary.csv")
    _write_csv(targets, output_dir / "convergence_target_times.csv")
    _write_csv(missing, output_dir / "convergence_missing_runs.csv")

    report = _report(config, runs, final_summary, stats_at_times, targets,
                     missing, history_missing, duplicates)
    (output_dir / "convergence_report.md").write_text(report, encoding="utf-8")
    (output_dir / "convergence_tables.tex").write_text(
        _latex_tables(final_summary, stats_at_times, targets,
                      float(config["time_budget_seconds"]))
        if not final_summary.empty else "% No formal convergence results.\n",
        encoding="utf-8",
    )

    if not curves.empty and make_plots:
        try:
            import plot_convergence
            plot_convergence.make_all_plots(
                curves=curves, final_summary=final_summary,
                targets=targets, config=config,
                output_dir=figures_dir, dpi=dpi,
                run_summary=run_summary,
            )
        except Exception as exc:
            print("WARNING: plotting failed: %s" % exc, file=sys.stderr)
            (output_dir / "plot_errors.txt").write_text(str(exc), encoding="utf-8")

    if not quiet:
        if runs.empty:
            print("尚无正式结果：%s" % results_dir)
            print("未生成任何统计量或图表。请先手动运行正式收敛实验。")
        else:
            print("已写入收敛分析：%s" % output_dir)
            print("runs=%d, curves=%d, histories_missing=%d, missing=%d"
                  % (len(runs), len(curves), history_missing, len(missing)))
    return runs, curves, final_summary, targets, missing


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze five-algorithm convergence results (read-only; no solver calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default=str(_HERE / "results"))
    p.add_argument("--history-dir", default=str(_HERE / "history"))
    p.add_argument("--output-dir", default=str(_HERE / "analysis"))
    p.add_argument("--figures-dir", default=str(_HERE / "figures"))
    p.add_argument("--config", default=str(_HERE / "config" / "convergence_config.json"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        results_dir=args.results_dir, history_dir=args.history_dir,
        output_dir=args.output_dir, figures_dir=args.figures_dir,
        config_path=args.config, make_plots=not args.no_plots,
        dpi=args.dpi, quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
