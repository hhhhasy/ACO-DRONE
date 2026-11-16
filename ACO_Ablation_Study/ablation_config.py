"""Shared configuration for the ACO+Tabu ablation study.

Everything that defines a run -- the six variants, the formal seeds, the time
budgets, the output path convention and the configuration fingerprints -- lives
here so that the runner, the batch runner, the analyser and the tests all use
exactly the same definition.

The baseline is the current ACO algorithm::

    service_policy_mode = "fixed"          # always strategy B
    adaptive_service_fraction = 1.0        # 100% of ants
    service_proxy_strength = 1.0           # full workload-aware bias (kappa)

A variant is only a set of ACOConfig overrides:

* ``full``: the current algorithm, all four pheromone layers enabled;
* ``no_edge`` / ``no_assign`` / ``no_service`` / ``no_drone``: exactly one
  ``pheromone_*`` boolean disabled relative to ``full``;
* ``no_workload_bias``: exactly one field differs from ``full``,
  ``service_proxy_strength = 0.0`` (kappa=0).  The two service-mode
  desirabilities become ``eta_truck = eta_drone = 1`` while the service choice
  is still made by ACS from ``tau_service`` and local/global learning.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from aco_solver import ACOConfig

# ---------------------------------------------------------------------------
# Formal experiment parameters (configured here, never run automatically)
# ---------------------------------------------------------------------------
ANTS = 30
FORMAL_SEEDS: Tuple[int, ...] = (13, 21, 42, 50, 65, 77, 100, 123, 2024, 3407)
FORMAL_BUDGETS: Tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
MAIN_BUDGET: float = 50.0
#: 30 datasets x 10 seeds x 1 budget x 6 variants = 1800 runs.
MAIN_EXPERIMENT_RUNS = 1800

SCHEMA_VERSION = 1
RUN_FILENAME = "run.json"
SOLUTION_FILENAME = "solution.txt"
HISTORY_FILENAME = "history.csv"
STATUS_COMPLETED = "completed"
BATCH_MANIFEST_FILENAME = "_batch_manifest.json"

#: Fields that legitimately vary between runs of the same variant.
VOLATILE_CONFIG_FIELDS = ("seed", "time_limit_seconds", "verbose_every")


@dataclass(frozen=True)
class VariantSpec:
    key: str
    title_zh: str
    description_zh: str
    service_policy_mode: str
    adaptive_service_fraction: float = 1.0
    service_proxy_strength: float = 1.0
    pheromone_edge: bool = True
    pheromone_assign: bool = True
    pheromone_service: bool = True
    pheromone_drone: bool = True

    def config_overrides(self) -> Dict[str, object]:
        return {
            "service_policy_mode": self.service_policy_mode,
            "adaptive_service_fraction": float(self.adaptive_service_fraction),
            "service_proxy_strength": float(self.service_proxy_strength),
            "pheromone_edge": self.pheromone_edge,
            "pheromone_assign": self.pheromone_assign,
            "pheromone_service": self.pheromone_service,
            "pheromone_drone": self.pheromone_drone,
        }


VARIANTS: "OrderedDict[str, VariantSpec]" = OrderedDict([
    ("full", VariantSpec(
        key="full",
        title_zh="完整当前 ACO（100% 策略 B，κ=1）",
        description_zh=(
            "当前 ACO+Tabu：全部蚂蚁使用策略 B，保留完整工作量感知双向偏好"
            "（service_proxy_strength=1.0）；四层信息素、Tabu 与原始物理"
            "约束全部启用。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        service_proxy_strength=1.0,
    )),
    ("no_edge", VariantSpec(
        key="no_edge",
        title_zh="关闭路径信息素 tau_edge",
        description_zh=(
            "完整当前 ACO，仅关闭路线边信息素：不参与候选评分、不初始化强化、"
            "不局部/全局更新、不蒸发、不重启平滑；卡车路径决策仍保留。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        pheromone_edge=False,
    )),
    ("no_assign", VariantSpec(
        key="no_assign",
        title_zh="关闭卡车归属信息素 tau_assign",
        description_zh=(
            "完整当前 ACO，仅关闭客户-卡车归属信息素；客户仍被分配到卡车，"
            "其余三层信息素与策略 B 保持不变。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        pheromone_assign=False,
    )),
    ("no_service", VariantSpec(
        key="no_service",
        title_zh="关闭服务方式信息素 tau_service",
        description_zh=(
            "完整当前 ACO，仅关闭卡车/无人机服务方式信息素；服务方式选择"
            "及其启发式评分仍保留，无人机配送不被禁止。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        pheromone_service=False,
    )),
    ("no_drone", VariantSpec(
        key="no_drone",
        title_zh="关闭无人机归属信息素 tau_drone",
        description_zh=(
            "完整当前 ACO，仅关闭客户-物理无人机归属信息素；无人机仍可配送，"
            "物理无人机选择仍由其余启发式和随机逻辑决定。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        pheromone_drone=False,
    )),
    ("no_workload_bias", VariantSpec(
        key="no_workload_bias",
        title_zh="取消策略 B 工作量偏向（κ=0）",
        description_zh=(
            "完整当前 ACO，仅将 service_proxy_strength 设为 0.0："
            "eta_truck = eta_drone = 1，客户对两种服务方式吸引力相同；"
            "服务方式仍由 tau_service 与 ACS 选择，四层信息素全部启用。"
        ),
        service_policy_mode="fixed",
        adaptive_service_fraction=1.0,
        service_proxy_strength=0.0,
    )),
])

VARIANT_KEYS: Tuple[str, ...] = tuple(VARIANTS.keys())
REFERENCE_VARIANT = "full"


def get_variant(key: str) -> VariantSpec:
    try:
        return VARIANTS[key]
    except KeyError:
        raise KeyError(
            "unknown variant %r; choose from %s" % (key, ", ".join(VARIANT_KEYS))
        )


# ---------------------------------------------------------------------------
# Config construction and fingerprints
# ---------------------------------------------------------------------------
def build_config(
    variant: str,
    seed: int,
    time_limit_seconds: float,
    ants: int = ANTS,
    iterations: int = 0,
    verbose_every: int = 0,
) -> ACOConfig:
    """Build the ACOConfig of one ablation run.

    Only the variant overrides are applied; every other field keeps the
    official solver default.  ``time_limit_seconds`` is passed at construction
    time because ``AntColonySolver`` validates it; the runner later replaces it
    with the remaining budget after instance loading/initialisation, exactly
    like ``aco_time_main.py`` does.
    """
    overrides = get_variant(variant).config_overrides()
    return ACOConfig(
        ants=int(ants),
        iterations=int(iterations),
        seed=int(seed),
        time_limit_seconds=float(time_limit_seconds),
        verbose_every=int(verbose_every),
        **overrides
    )


def stable_config_dict(config: ACOConfig) -> Dict[str, object]:
    """Config values that define a variant, excluding per-run fields."""
    data = dict(vars(config))
    for key in VOLATILE_CONFIG_FIELDS:
        data.pop(key, None)
    return data


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def variant_fingerprint(variant: str) -> str:
    """Hash of all non-volatile config fields of a variant."""
    cfg = build_config(variant, seed=0, time_limit_seconds=0.0,
                       ants=ANTS, iterations=0, verbose_every=0)
    return _sha256_json(stable_config_dict(cfg))


def run_config_fingerprint(
    variant: str,
    seed: int,
    budget: float,
    ants: int = ANTS,
    iterations: int = 0,
) -> str:
    """Hash of everything that identifies one run's algorithm settings."""
    cfg = build_config(variant, seed=seed, time_limit_seconds=budget,
                       ants=ants, iterations=iterations, verbose_every=0)
    return _sha256_json(dict(vars(cfg)))


def config_diff(variant_a: str, variant_b: str) -> Dict[str, Tuple[object, object]]:
    """Fields whose values differ between two variants (volatile excluded)."""
    cfg_a = build_config(variant_a, seed=0, time_limit_seconds=0.0)
    cfg_b = build_config(variant_b, seed=0, time_limit_seconds=0.0)
    a = stable_config_dict(cfg_a)
    b = stable_config_dict(cfg_b)
    return {
        key: (a.get(key), b.get(key))
        for key in sorted(set(a) | set(b))
        if a.get(key) != b.get(key)
    }


def validate_variant_isolation() -> None:
    """Enforce the strict single-factor ablation design.

    * ``full`` is the shared baseline: current ACO, 100% strategy B, kappa=1.
    * every ``no_*`` differs from ``full`` in exactly one pheromone flag;
    * ``no_workload_bias`` differs from ``full`` only in
      ``service_proxy_strength`` (1.0 -> 0.0);
    * every variant keeps the same fixed service policy, fraction and all other
      ACOConfig fields.
    """
    if len(VARIANT_KEYS) != 6:
        raise AssertionError("expected six variants, got %r" % (VARIANT_KEYS,))
    if REFERENCE_VARIANT not in VARIANTS:
        raise AssertionError("reference variant %r is missing" % REFERENCE_VARIANT)

    for variant in VARIANT_KEYS:
        spec = get_variant(variant)
        if spec.service_policy_mode != "fixed":
            raise AssertionError("%s must use fixed service policy" % variant)
        if spec.adaptive_service_fraction != 1.0:
            raise AssertionError("%s must use 100%% strategy B" % variant)

    expected_flag = {
        "no_edge": "pheromone_edge",
        "no_assign": "pheromone_assign",
        "no_service": "pheromone_service",
        "no_drone": "pheromone_drone",
    }
    for variant, flag in expected_flag.items():
        diff = config_diff(variant, REFERENCE_VARIANT)
        if set(diff) != {flag}:
            raise AssertionError(
                "%s must differ from %s only in %s; got %r"
                % (variant, REFERENCE_VARIANT, flag, diff)
            )
        if diff[flag] != (False, True):
            raise AssertionError(
                "%s must disable %s; got %r" % (variant, flag, diff[flag])
            )

    bias_diff = config_diff("no_workload_bias", REFERENCE_VARIANT)
    if set(bias_diff) != {"service_proxy_strength"}:
        raise AssertionError(
            "no_workload_bias must differ only in service_proxy_strength; got %r"
            % (bias_diff,)
        )
    if bias_diff["service_proxy_strength"] != (0.0, 1.0):
        raise AssertionError(
            "no_workload_bias must set kappa=0; got %r"
            % (bias_diff["service_proxy_strength"],)
        )


# ---------------------------------------------------------------------------
# Source / dataset provenance
# ---------------------------------------------------------------------------
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

CODE_FILES: Tuple[str, ...] = (
    "aco_solver.py",
    "evaluator.py",
    "drone_physics.py",
    "fast_window_physics.py",
    "ablation_config.py",
)


def code_fingerprint() -> str:
    """Hash of every file that can change solver behaviour."""
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = os.path.join(STUDY_DIR, name)
        digest.update(name.encode("utf-8"))
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(path: str) -> Dict[str, object]:
    absolute = os.path.abspath(path)
    return {
        "dataset": os.path.splitext(os.path.basename(absolute))[0],
        "dataset_path": absolute,
        "dataset_sha256": file_sha256(absolute),
    }


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
def format_budget_tag(budget: float) -> str:
    return ("%g" % float(budget)) + "s"


def run_directory(
    results_root: str,
    variant: str,
    budget: float,
    dataset_name: str,
    seed: int,
) -> str:
    """``results/<variant>/<budget>s/<dataset>/seed<seed>/``.

    The four components are all part of the path, so different variants,
    budgets, datasets and seeds can never overwrite one another.
    """
    root = os.path.abspath(results_root)
    path = os.path.join(
        root, str(variant), format_budget_tag(budget),
        str(dataset_name), "seed%d" % int(seed),
    )
    # Defence in depth: the result must stay below its own results root.
    common = os.path.commonpath([root, os.path.abspath(path)])
    if common != root:
        raise ValueError("refusing output path outside results root: %s" % path)
    return path


def run_file_path(
    results_root: str,
    variant: str,
    budget: float,
    dataset_name: str,
    seed: int,
) -> str:
    return os.path.join(
        run_directory(results_root, variant, budget, dataset_name, seed),
        RUN_FILENAME,
    )


def discover_dataset_files(data_dir: str, pattern: str = "*.json") -> List[str]:
    """Sorted JSON files below ``data_dir``; basenames must be unique."""
    root = os.path.abspath(data_dir)
    if not os.path.isdir(root):
        raise ValueError("dataset directory does not exist: %s" % data_dir)
    paths = sorted(
        os.path.abspath(p)
        for p in glob.glob(os.path.join(root, pattern), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(".json")
    )
    if not paths:
        raise ValueError(
            "no JSON datasets matched %s/%s" % (data_dir, pattern)
        )
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset basenames in %s" % data_dir)
    return paths


def summarise_plan(
    variants: Sequence[str],
    budgets: Sequence[float],
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, object]:
    return {
        "variants": list(variants),
        "budgets": [float(x) for x in budgets],
        "datasets": [os.path.splitext(os.path.basename(x))[0] for x in datasets],
        "seeds": [int(x) for x in seeds],
        "total_runs": int(len(variants) * len(budgets) * len(datasets) * len(seeds)),
    }
