"""Statistical analysis of the drone battery/payload sensitivity experiment.

Reads only ``results/runs/*.json`` written by ``run_sensitivity.py``.  It never
imports or runs a solver.  If no formal results exist it writes a clear report
and does not create figures.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ALGORITHMS = ("ACO", "GA", "ALNS", "PSO", "VND")
KEY = ["algorithm", "battery_wh", "payload_kg", "seed"]


def _as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(v, d: int = 3) -> str:
    x = _num(v)
    return "NA" if x is None else (("%." + str(d) + "f") % x)


def _fmt_pct(v, d: int = 2) -> str:
    x = _num(v)
    return "NA" if x is None else (("%." + str(d) + "f") % x) + "%"


def _latex_escape(value: Any) -> str:
    s = str(value)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in s)


def _write_csv(df: pd.DataFrame, path: Path, fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if fields is not None:
        for f in fields:
            if f not in out.columns:
                out[f] = np.nan
        out = out[list(fields)]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_（无数据）_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_project_path(value: str) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = _HERE / p
    return p.resolve()


def parameter_combinations(config: Dict[str, Any]) -> List[Tuple[float, float]]:
    b0 = float(config["baseline"]["battery_wh"])
    p0 = float(config["baseline"]["payload_kg"])
    combos = [(float(b), p0) for b in config["battery_wh_levels"]]
    for p in config["payload_kg_levels"]:
        if (b0, float(p)) not in combos:
            combos.append((b0, float(p)))
    out, seen = [], set()
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


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
        ("battery_wh", np.nan), ("payload_kg", np.nan),
        ("time_budget_seconds", np.nan), ("actual_runtime_seconds", np.nan),
        ("best_makespan_s", np.nan), ("feasible", False),
        ("initial_makespan_s", np.nan), ("iterations_or_generations", np.nan),
        ("evaluations", np.nan), ("drone_served_customers", np.nan),
        ("truck_served_customers", np.nan), ("drone_share_pct", np.nan),
        ("drone_sorties", np.nan), ("sorties_per_physical_drone", np.nan),
        ("mean_customers_per_sortie", np.nan),
        ("total_flight_energy_wh", np.nan), ("total_hover_energy_wh", np.nan),
        ("total_energy_wh", np.nan), ("battery_utilization_pct", np.nan),
        ("max_drone_battery_utilization_pct", np.nan),
        ("fleet_completion_cv", np.nan), ("max_minus_mean_completion_s", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    df["algorithm"] = df["algorithm"].astype(str).str.upper()
    df["dataset"] = df["dataset"].astype(str)
    df["status"] = df["status"].astype(str).str.lower()
    for col in ["seed", "battery_wh", "payload_kg", "time_budget_seconds",
                "actual_runtime_seconds", "best_makespan_s", "initial_makespan_s",
                "iterations_or_generations", "evaluations", "drone_served_customers",
                "truck_served_customers", "drone_share_pct", "drone_sorties",
                "sorties_per_physical_drone", "mean_customers_per_sortie",
                "total_flight_energy_wh", "total_hover_energy_wh", "total_energy_wh",
                "battery_utilization_pct", "max_drone_battery_utilization_pct",
                "fleet_completion_cv", "max_minus_mean_completion_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["seed"] = df["seed"].astype("Int64")
    df["feasible_flag"] = df["feasible"].map(_as_bool)
    df["valid_quality"] = (
        df["status"].eq("ok") & df["feasible_flag"] &
        df["best_makespan_s"].notna() & np.isfinite(df["best_makespan_s"]) &
        (df["best_makespan_s"] > 0)
    )
    df["cv_pct"] = np.where(
        (df["fleet_completion_cv"].notna()) & (df["fleet_completion_cv"] >= 0),
        100.0 * df["fleet_completion_cv"], np.nan,
    )
    return df


def _dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    dup = int(df.duplicated(subset=KEY, keep=False).sum())
    if dup:
        df = df.sort_values(KEY + ["status"]).drop_duplicates(subset=KEY, keep="last")
    return df, dup


def _aggregate(df: pd.DataFrame, dimension: str, config: Dict[str, Any]) -> pd.DataFrame:
    baseline_b = float(config["baseline"]["battery_wh"])
    baseline_p = float(config["baseline"]["payload_kg"])
    rows = []
    for alg in ALGORITHMS:
        sub_alg = df[df["algorithm"] == alg]
        if sub_alg.empty:
            continue
        levels = sorted(sub_alg["battery_wh" if dimension == "battery" else "payload_kg"].dropna().unique())
        for level in levels:
            if dimension == "battery":
                sub = sub_alg[
                    np.isclose(sub_alg["battery_wh"], level) &
                    np.isclose(sub_alg["payload_kg"], baseline_p)
                ]
            else:
                sub = sub_alg[
                    np.isclose(sub_alg["payload_kg"], level) &
                    np.isclose(sub_alg["battery_wh"], baseline_b)
                ]
            if sub.empty:
                continue
            valid = sub[sub["valid_quality"]]
            values = valid["best_makespan_s"].to_numpy(float)
            n = len(sub)
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) >= 2 else (0.0 if len(values) else np.nan)
            rows.append({
                "algorithm": alg,
                "dimension": dimension,
                "level": float(level),
                "unit": "Wh" if dimension == "battery" else "kg",
                "n_runs": n,
                "n_feasible": int(sub["feasible_flag"].sum()),
                "feasible_rate": float(sub["feasible_flag"].mean()) if n else np.nan,
                "n_quality": len(values),
                "mean_makespan_s": mean,
                "std_makespan_s": std,
                "median_makespan_s": float(np.median(values)) if len(values) else np.nan,
                "min_makespan_s": float(np.min(values)) if len(values) else np.nan,
                "max_makespan_s": float(np.max(values)) if len(values) else np.nan,
                "cv_pct_across_seeds": (100.0 * std / mean if mean and mean > 0 and np.isfinite(std) else np.nan),
                "mean_actual_runtime_seconds": float(sub["actual_runtime_seconds"].mean()),
                "mean_iterations": float(sub["iterations_or_generations"].mean()),
                "mean_evaluations": float(sub["evaluations"].mean()),
                "mean_drone_served_customers": float(sub["drone_served_customers"].mean()),
                "mean_drone_share_pct": float(sub["drone_share_pct"].mean()),
                "mean_drone_sorties": float(sub["drone_sorties"].mean()),
                "mean_sorties_per_physical_drone": float(sub["sorties_per_physical_drone"].mean()),
                "mean_customers_per_sortie": float(sub["mean_customers_per_sortie"].mean()),
                "mean_total_flight_energy_wh": float(sub["total_flight_energy_wh"].mean()),
                "mean_total_hover_energy_wh": float(sub["total_hover_energy_wh"].mean()),
                "mean_total_energy_wh": float(sub["total_energy_wh"].mean()),
                "mean_battery_utilization_pct": float(sub["battery_utilization_pct"].mean()),
                "mean_max_drone_battery_utilization_pct": float(sub["max_drone_battery_utilization_pct"].mean()),
                "mean_fleet_completion_cv_pct": float(sub["cv_pct"].mean()),
                "mean_max_minus_mean_completion_s": float(sub["max_minus_mean_completion_s"].mean()),
            })
    return pd.DataFrame(rows)


def _baseline_changes(summary: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    baseline_b = float(config["baseline"]["battery_wh"])
    baseline_p = float(config["baseline"]["payload_kg"])
    rows = []
    for _, row in summary.iterrows():
        baseline_level = baseline_b if row["dimension"] == "battery" else baseline_p
        base = summary[
            (summary["algorithm"] == row["algorithm"]) &
            (summary["dimension"] == row["dimension"]) &
            np.isclose(summary["level"], baseline_level)
        ]
        base_mean = float(base["mean_makespan_s"].iloc[0]) if not base.empty else np.nan
        mean = _num(row["mean_makespan_s"])
        diff = mean - base_mean if mean is not None and _num(base_mean) is not None else np.nan
        pct = 100.0 * diff / base_mean if diff is not None and base_mean and np.isfinite(base_mean) else np.nan
        rows.append({
            "algorithm": row["algorithm"], "dimension": row["dimension"],
            "level": row["level"], "baseline_level": baseline_level,
            "baseline_mean_makespan_s": base_mean, "mean_makespan_s": mean,
            "absolute_change_s": diff, "percent_change": pct,
        })
    return pd.DataFrame(rows)


def _rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dim, level), group in summary.groupby(["dimension", "level"]):
        valid = group[group["mean_makespan_s"].notna()].sort_values("mean_makespan_s")
        for rank, (_, row) in enumerate(valid.iterrows(), 1):
            rows.append({
                "dimension": dim, "level": float(level), "algorithm": row["algorithm"],
                "mean_makespan_s": row["mean_makespan_s"], "rank": rank,
                "n_quality": row["n_quality"],
            })
        for _, row in group[group["mean_makespan_s"].isna()].iterrows():
            rows.append({
                "dimension": dim, "level": float(level), "algorithm": row["algorithm"],
                "mean_makespan_s": np.nan, "rank": np.nan, "n_quality": row["n_quality"],
            })
    return pd.DataFrame(rows)


def _missing(df: pd.DataFrame, config: Dict[str, Any], algorithms: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    combos = parameter_combinations(config)
    present = set(map(tuple, df[KEY].itertuples(index=False, name=None))) if not df.empty else set()
    rows = []
    for alg in algorithms:
        for b, p in combos:
            for seed in seeds:
                if (alg, float(b), float(p), int(seed)) not in present:
                    rows.append({"algorithm": alg, "battery_wh": b, "payload_kg": p,
                                 "seed": int(seed), "reason": "missing"})
    for _, row in df[~df["status"].eq("ok")].iterrows():
        rows.append({"algorithm": row["algorithm"], "battery_wh": row["battery_wh"],
                     "payload_kg": row["payload_kg"], "seed": row["seed"],
                     "reason": "failed", "error": row.get("error", "")})
    return pd.DataFrame(rows)


def _latex_tables(summary: pd.DataFrame, changes: pd.DataFrame, rankings: pd.DataFrame) -> str:
    lines = ["% Generated by analyze_sensitivity.py; requires booktabs.",
             "% \\input{sensitivity_tables.tex}", ""]
    for dim, caption in [("battery", "Battery capacity"), ("payload", "Payload capacity")]:
        panel = summary[summary["dimension"] == dim]
        if panel.empty:
            continue
        lines += [
            "\\begin{table}[htbp]", "\\centering",
            "\\caption{%s sensitivity (mean $\\pm$ sample std of makespan, s).}" % caption,
            "\\label{tab:sensitivity-%s}" % dim,
            "\\begin{tabular}{lrrrrr}", "\\toprule",
            "Level & ACO & GA & ALNS & PSO & VND \\\\", "\\midrule",
        ]
        for level in sorted(panel["level"].unique()):
            cells = []
            for alg in ALGORITHMS:
                row = panel[(panel["algorithm"] == alg) & np.isclose(panel["level"], level)]
                if row.empty or pd.isna(row["mean_makespan_s"].iloc[0]):
                    cells.append("NA")
                else:
                    cells.append("%s $\\pm$ %s" % (
                        _fmt(row["mean_makespan_s"].iloc[0], 1),
                        _fmt(row["std_makespan_s"].iloc[0], 1),
                    ))
            unit = panel["unit"].iloc[0]
            lines.append("%g %s & %s \\\\" % (level, _latex_escape(unit), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    if not rankings.empty:
        lines += [
            "\\begin{table}[htbp]", "\\centering",
            "\\caption{Algorithm ranks by mean makespan at each resource level (1 = best).}",
            "\\label{tab:sensitivity-ranks}", "\\begin{tabular}{llrrrrr}", "\\toprule",
            "Dimension & Level & ACO & GA & ALNS & PSO & VND \\\\", "\\midrule",
        ]
        for (dim, level), group in rankings.groupby(["dimension", "level"]):
            rank_map = dict(zip(group["algorithm"], group["rank"]))
            cells = [_fmt(rank_map.get(alg, np.nan), 0) for alg in ALGORITHMS]
            lines.append("%s & %g & %s \\\\" % (dim, level, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines) + "\n"


def _report(
    config: Dict[str, Any], df: pd.DataFrame, summary: pd.DataFrame,
    changes: pd.DataFrame, rankings: pd.DataFrame, missing: pd.DataFrame,
    duplicates: int,
) -> str:
    lines = ["# 无人机资源灵敏度实验报告\n"]
    lines.append("生成时间（UTC）：%s\n" % datetime.now(timezone.utc).isoformat())
    lines.append("数据集：`%s`\n" % config.get("dataset_name", config.get("dataset")))
    lines.append("算法：%s\n" % ", ".join(config.get("algorithms", ALGORITHMS)))
    lines.append("随机种子：%s\n" % ", ".join(str(s) for s in config.get("seeds", [])))
    lines.append("时间预算：`%g s`\n" % float(config.get("time_budget_seconds", 0)))
    lines.append(
        "资源组合：电池 %s Wh × 载重 %s kg；基准为 %g Wh / %g kg。"
        "电池灵敏度固定载重 %g kg，载重灵敏度固定电池 %g Wh。\n"
        % (config["battery_wh_levels"], config["payload_kg_levels"],
           config["baseline"]["battery_wh"], config["baseline"]["payload_kg"],
           config["baseline"]["payload_kg"], config["baseline"]["battery_wh"])
    )
    lines += ["## 数据完整性\n"]
    n_ok = int(df["status"].eq("ok").sum()) if not df.empty else 0
    n_valid = int(df["valid_quality"].sum()) if not df.empty else 0
    lines.append("- 运行记录：%d" % len(df))
    lines.append("- 成功运行：%d" % n_ok)
    lines.append("- 可行运行：%d" % n_valid)
    lines.append("- 重复键：%d" % duplicates)
    lines.append("- 缺失/失败记录：%d\n" % len(missing))
    if df.empty:
        lines += [
            "## 尚无正式结果\n",
            "`results/` 中没有正式灵敏度实验记录。本程序没有运行任何求解器，"
            "也没有生成任何图表或统计结论。请先手动执行 `run_sensitivity.py --execute`。\n",
        ]
        return "\n".join(lines)

    for dim, title in [("battery", "电池容量灵敏度（载重 = %g kg）" % config["baseline"]["payload_kg"]),
                      ("payload", "最大载重灵敏度（电池 = %g Wh）" % config["baseline"]["battery_wh"])]:
        panel = summary[summary["dimension"] == dim]
        lines.append("## %s\n" % title)
        if panel.empty:
            lines.append("_无数据。_\n")
            continue
        headers = ["算法"] + ["%g" % x for x in sorted(panel["level"].unique())]
        rows = []
        for alg in ALGORITHMS:
            cells = []
            for level in sorted(panel["level"].unique()):
                r = panel[(panel["algorithm"] == alg) & np.isclose(panel["level"], level)]
                if r.empty or pd.isna(r["mean_makespan_s"].iloc[0]):
                    cells.append("NA")
                else:
                    cells.append("%s ± %s" % (
                        _fmt(r["mean_makespan_s"].iloc[0], 1),
                        _fmt(r["std_makespan_s"].iloc[0], 1),
                    ))
            rows.append([alg] + cells)
        unit = panel["unit"].iloc[0]
        lines.append("单位：Makespan s；水平为 %s。\n" % unit)
        lines.append(_md_table(headers, rows))
        ch = changes[changes["dimension"] == dim]
        if not ch.empty:
            lines.append("相对基准的平均变化：\n")
            hr = ["算法", "水平", "改变 %"]
            rr = [[r.algorithm, "%g" % r.level, _fmt_pct(r.percent_change, 2)]
                  for r in ch.itertuples()]
            lines.append(_md_table(hr, rr))
        rk = rankings[rankings["dimension"] == dim]
        if not rk.empty:
            lines.append("各水平排名（1 = 最佳）：\n")
            hr = ["水平"] + list(ALGORITHMS)
            rr = []
            for level in sorted(rk["level"].unique()):
                g = rk[np.isclose(rk["level"], level)]
                m = {r.algorithm: r["rank"] for _, r in g.iterrows()}
                rr.append(["%g" % level] + [
                    _fmt(m.get(a), 0) for a in ALGORITHMS
                ])
            lines.append(_md_table(hr, rr))
    lines += ["## 无人机使用与稳定性\n"]
    lines.append(
        "下表给出各算法在各水平上的无人机服务比例、平均任务数和种子间 CV；"
        "详细数值见 `sensitivity_summary.csv` 与 `sensitivity_runs.csv`。\n"
    )
    hr = ["算法", "维度", "水平", "无人机比例 %", "任务数", "CV %", "可行率"]
    rr = [[r.algorithm, r.dimension, "%g" % r.level,
           _fmt(r.mean_drone_share_pct, 2), _fmt(r.mean_drone_sorties, 2),
           _fmt(r.cv_pct_across_seeds, 2), _fmt_pct(100.0 * r.feasible_rate, 1)]
          for r in summary.itertuples()]
    lines.append(_md_table(hr, rr))
    lines += [
        "\n## 说明\n",
        "- 所有数值来自 `results/runs/*.json`；未运行求解器、未补造数据。",
        "- 可行率统计所有运行；均值/标准差/排名只使用可行且 Makespan 有限的运行。",
        "- 电池或载重改变后，同一参数被传入构造、解码与统一评价器；详见 README。",
        "- 若某算法在低资源下初始解不可行，记录为 feasible=false，而不是伪造 Makespan。",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    figures_dir: str | Path,
    make_plots: bool = True,
    dpi: int = 300,
    quiet: bool = False,
):
    config = load_config(Path(config_path))
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(input_dir)
    if not df.empty:
        df, duplicates = _dedupe(df)
    else:
        duplicates = 0

    algorithms = [str(a).upper() for a in config.get("algorithms", ALGORITHMS)]
    seeds = [int(s) for s in config.get("seeds", [])]
    missing = _missing(df, config, algorithms, seeds) if not df.empty else pd.DataFrame(
        columns=["algorithm", "battery_wh", "payload_kg", "seed", "reason", "error"]
    )
    summary = _aggregate(df, "battery", config) if not df.empty else pd.DataFrame()
    summary_payload = _aggregate(df, "payload", config) if not df.empty else pd.DataFrame()
    summary = pd.concat([summary, summary_payload], ignore_index=True) if not summary.empty or not summary_payload.empty else pd.DataFrame()
    changes = _baseline_changes(summary, config) if not summary.empty else pd.DataFrame()
    rankings = _rankings(summary) if not summary.empty else pd.DataFrame()

    _write_csv(df, output_dir / "sensitivity_runs.csv")
    _write_csv(summary, output_dir / "sensitivity_summary.csv")
    _write_csv(changes, output_dir / "sensitivity_baseline_changes.csv")
    _write_csv(rankings, output_dir / "sensitivity_rankings.csv")
    # Compact paper-facing table: mean, sample std and rank.
    paper_rows = []
    rank_map = {}
    if not rankings.empty:
        for r in rankings.itertuples():
            rank_map[(r.algorithm, r.dimension, float(r.level))] = r.rank
    if not summary.empty:
        for r in summary.itertuples():
            paper_rows.append({
                "algorithm": r.algorithm, "dimension": r.dimension,
                "level": r.level, "unit": r.unit,
                "mean_makespan_s": r.mean_makespan_s,
                "std_makespan_s": r.std_makespan_s,
                "rank": rank_map.get((r.algorithm, r.dimension, float(r.level)), np.nan),
                "n_quality": r.n_quality,
                "feasible_rate": r.feasible_rate,
            })
    _write_csv(pd.DataFrame(paper_rows), output_dir / "sensitivity_paper_table.csv")
    _write_csv(missing, output_dir / "sensitivity_missing_runs.csv")
    # Stability/drone views are explicit slices of the same real summary.
    _write_csv(
        summary[["algorithm", "dimension", "level", "n_runs", "n_quality",
                 "mean_makespan_s", "std_makespan_s", "cv_pct_across_seeds",
                 "min_makespan_s", "max_makespan_s"]]
        if not summary.empty else pd.DataFrame(),
        output_dir / "sensitivity_stability.csv",
    )
    _write_csv(
        summary[["algorithm", "dimension", "level", "mean_drone_served_customers",
                 "mean_drone_share_pct", "mean_drone_sorties",
                 "mean_sorties_per_physical_drone", "mean_customers_per_sortie",
                 "mean_total_flight_energy_wh", "mean_total_hover_energy_wh",
                 "mean_total_energy_wh", "mean_battery_utilization_pct",
                 "mean_max_drone_battery_utilization_pct"]]
        if not summary.empty else pd.DataFrame(),
        output_dir / "sensitivity_drone_usage.csv",
    )

    report = _report(config, df, summary, changes, rankings, missing, duplicates)
    (output_dir / "sensitivity_report.md").write_text(report, encoding="utf-8")
    (output_dir / "sensitivity_tables.tex").write_text(
        _latex_tables(summary, changes, rankings) if not summary.empty
        else "% No formal sensitivity results.\n",
        encoding="utf-8",
    )

    if not df.empty and make_plots:
        try:
            import plot_sensitivity
            plot_sensitivity.make_all_plots(
                summary=summary, changes=changes, rankings=rankings,
                config=config, output_dir=figures_dir, dpi=dpi,
            )
        except Exception as exc:  # plotting must not destroy tables
            print("WARNING: plotting failed: %s" % exc, file=sys.stderr)
            (output_dir / "plot_errors.txt").write_text(str(exc), encoding="utf-8")

    if not quiet:
        if df.empty:
            print("尚无正式结果：%s" % input_dir)
            print("未生成任何统计量或图表。请先手动运行正式灵敏度实验。")
        else:
            print("已写入灵敏度分析：%s" % output_dir)
            print("runs=%d, summary rows=%d, missing=%d"
                  % (len(df), len(summary), len(missing)))
    return df, summary, changes, rankings, missing


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze drone sensitivity results (read-only; no solver calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=None,
                   help="default: config.output_dir")
    p.add_argument("--output-dir", default=None,
                   help="default: config.analysis_dir (or analysis)")
    p.add_argument("--figures-dir", default=None,
                   help="default: config.figures_dir (or figures)")
    p.add_argument("--config", default=str(_HERE / "config" / "sensitivity_config.json"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    input_dir = _resolve_project_path(
        args.input_dir or config.get("output_dir", "results")
    )
    output_dir = _resolve_project_path(
        args.output_dir or config.get("analysis_dir", "analysis")
    )
    figures_dir = _resolve_project_path(
        args.figures_dir or config.get("figures_dir", "figures")
    )
    run_analysis(
        input_dir=input_dir, output_dir=output_dir,
        config_path=args.config, figures_dir=figures_dir,
        make_plots=not args.no_plots, dpi=args.dpi, quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
