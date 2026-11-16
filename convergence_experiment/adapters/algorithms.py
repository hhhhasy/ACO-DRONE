"""Thin adapters around the five comparison algorithms.

No optimisation mechanism, objective or hyperparameter is reimplemented here.
The only sensitivity-specific operation is setting ``drone_battery_capacity_wh``
and ``drone_payload_capacity`` on the *same* problem/model objects that the
original solvers and the unified evaluator already use.

For ALNS/VND/PSO the battery capacity is passed through the new
``vrp_with_drones(..., drone_battery_capacity_wh=...)`` argument.  For ACO/GA/PSO the
two attributes on their ``ProblemData`` are overridden immediately after loading.
All final feasibility checks use ``evaluator.evaluate_solution``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

# This machine's threadpoolctl library scan crashes on import of sklearn
# (openblas_get_config() returns NULL).  Neutralize only that scan; BLAS thread
# counts are already pinned to 1 above.
try:
    import threadpoolctl
    threadpoolctl._ThreadpoolInfo._load_modules = lambda self: None
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_PROJECT_DIR = _HERE.parent
_RESEARCH_ROOT = _PROJECT_DIR.parent
for _path in (str(_RESEARCH_ROOT), str(_PROJECT_DIR), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluator import evaluate_solution
from load_data import build_model_kwargs, load_data_from_json
from model import vrp_with_drones
from aco_solver import ACOConfig, AntColonySolver, load_problem as aco_load_problem
from ga_solver import ClassicGA, GAConfig, decode as ga_decode, load_problem as ga_load_problem
from pso_solver import PSOConfig, ParticleSwarmSolver
from pso_decoder import load_problem as pso_load_problem
from vnd_solver import UserNeighborhoodVND


@dataclass
class RunSpec:
    algorithm: str
    dataset_path: str
    dataset_name: str
    seed: int
    time_budget_seconds: float
    battery_wh: float
    payload_kg: float
    params: Dict[str, Any] = field(default_factory=dict)


def apply_aco_physics(problem, battery_wh: float, payload_kg: float):
    problem.drone_battery_capacity_wh = float(battery_wh)
    problem.drone_payload_capacity = float(payload_kg)
    return problem


def load_aco_problem(dataset_path: str, battery_wh: float, payload_kg: float):
    return apply_aco_physics(aco_load_problem(str(dataset_path)), battery_wh, payload_kg)


def load_pso_problem(dataset_path: str, battery_wh: float, payload_kg: float):
    # Latest PSO uses its own ProblemData/decoder; only the physics overrides
    # are applied here, exactly like the other algorithms.
    problem = pso_load_problem(str(dataset_path))
    problem.drone_battery_capacity_wh = float(battery_wh)
    problem.drone_payload_capacity = float(payload_kg)
    return problem


def apply_ga_physics(problem, battery_wh: float, payload_kg: float):
    problem.drone_battery_capacity_wh = float(battery_wh)
    problem.drone_payload_capacity = float(payload_kg)
    return problem


def load_ga_problem(dataset_path: str, battery_wh: float, payload_kg: float):
    return apply_ga_physics(ga_load_problem(str(dataset_path)), battery_wh, payload_kg)


def build_alns_model(dataset_path: str, seed: int, battery_wh: float, payload_kg: float):
    data = load_data_from_json(str(dataset_path), verbose=False)
    if data is None:
        raise RuntimeError("failed to load dataset: %s" % dataset_path)
    kwargs = build_model_kwargs(data)
    kwargs["drone_battery_capacity_wh"] = float(battery_wh)
    kwargs["drone_payload_capacity"] = float(payload_kg)
    return vrp_with_drones(**kwargs, seed=int(seed))


def _fleet_cv(times) -> float:
    values = [float(x) for x in times if math.isfinite(float(x))]
    if not values:
        return float("nan")
    mean = statistics.fmean(values)
    if mean <= 0:
        return float("nan")
    return statistics.pstdev(values) / mean


def summarize_evaluation(
    ev, problem, battery_wh: float, payload_kg: float,
    vehicle_routes=None,
) -> Dict[str, Any]:
    """Extract the common result fields from the unified evaluator output."""
    n = int(getattr(problem, "num_customers", 0))
    if not ev.feasible:
        return {
            "feasible": False,
            "best_makespan_s": None,
            "failure_reason": str(ev.reason),
            "drone_served_customers": None,
            "truck_served_customers": None,
            "drone_share_pct": None,
            "drone_sorties": None,
            "sorties_per_physical_drone": None,
            "mean_customers_per_sortie": None,
            "total_flight_energy_wh": None,
            "total_hover_energy_wh": None,
            "total_energy_wh": None,
            "battery_utilization_pct": None,
            "max_drone_battery_utilization_pct": None,
            "drone_usage": [],
            "vehicle_completion_times_s": [],
            "fleet_completion_cv": None,
            "max_minus_mean_completion_s": None,
        }

    drone_customers = sum(len(report.order) for report in ev.sorties)
    if vehicle_routes is None:
        vehicle_routes = []
    truck_customers = sum(max(0, len(route) - 2) for route in vehicle_routes)
    n_physical = max(
        1, int(getattr(problem, "num_vehicles", 0)) *
        int(getattr(problem, "num_drones_per_vehicle", 0))
    )
    drone_usage = [
        {
            "vehicle": int(v) + 1,
            "label": str(label),
            "used_wh": float(state.energy_wh),
            "remaining_energy_wh": float(state.remaining_energy_wh),
            "sorties": len(state.sortie_keys),
        }
        for (v, label), state in sorted(ev.drones.items())
    ]
    total_energy = float(ev.total_flight_energy_wh) + float(ev.total_hover_energy_wh)
    max_util = max(
        (100.0 * float(x["used_wh"]) / float(battery_wh) for x in drone_usage),
        default=0.0,
    )
    times = [float(x) for x in ev.vehicle_completion_times]
    mean_time = statistics.fmean(times) if times else float("nan")
    return {
        "feasible": True,
        "best_makespan_s": float(ev.makespan),
        "failure_reason": "",
        "drone_served_customers": int(drone_customers),
        "truck_served_customers": int(truck_customers),
        "drone_share_pct": (100.0 * drone_customers / n) if n else float("nan"),
        "drone_sorties": int(len(ev.sorties)),
        "sorties_per_physical_drone": len(ev.sorties) / n_physical,
        "mean_customers_per_sortie": (
            drone_customers / len(ev.sorties) if ev.sorties else 0.0
        ),
        "total_flight_energy_wh": float(ev.total_flight_energy_wh),
        "total_hover_energy_wh": float(ev.total_hover_energy_wh),
        "total_energy_wh": total_energy,
        "battery_utilization_pct": (
            100.0 * total_energy / (n_physical * float(battery_wh))
            if battery_wh > 0 else float("nan")
        ),
        "max_drone_battery_utilization_pct": max_util,
        "drone_usage": drone_usage,
        "vehicle_completion_times_s": times,
        "fleet_completion_cv": _fleet_cv(times),
        "max_minus_mean_completion_s": (
            max(times) - mean_time if times else float("nan")
        ),
    }


def _base_result(spec: RunSpec, start: float) -> Dict[str, Any]:
    return {
        "status": "ok",
        "algorithm": spec.algorithm,
        "dataset": spec.dataset_name,
        "dataset_path": str(spec.dataset_path),
        "seed": int(spec.seed),
        "time_budget_seconds": float(spec.time_budget_seconds),
        "battery_wh": float(spec.battery_wh),
        "payload_kg": float(spec.payload_kg),
        "algorithm_parameters": dict(spec.params),
        "actual_runtime_seconds": float(time.perf_counter() - start),
    }


def _infeasible_result(spec: RunSpec, start: float, reason: str, **extra) -> Dict[str, Any]:
    result = _base_result(spec, start)
    result.update({
        "feasible": False,
        "best_makespan_s": None,
        "failure_reason": str(reason),
        "initial_makespan_s": None,
        "iterations_or_generations": 0,
        "evaluations": 0,
        "vehicle_routes": [],
        "drone_routes": [],
        "drone_served_customers": None,
        "truck_served_customers": None,
        "drone_share_pct": None,
        "drone_sorties": None,
        "sorties_per_physical_drone": None,
        "mean_customers_per_sortie": None,
        "total_flight_energy_wh": None,
        "total_hover_energy_wh": None,
        "total_energy_wh": None,
        "battery_utilization_pct": None,
        "max_drone_battery_utilization_pct": None,
        "drone_usage": [],
        "vehicle_completion_times_s": [],
        "fleet_completion_cv": None,
        "max_minus_mean_completion_s": None,
    })
    result.update(extra)
    return result


def _run_aco(spec: RunSpec, start: float, recorder=None) -> Dict[str, Any]:
    p = spec.params
    problem = load_aco_problem(spec.dataset_path, spec.battery_wh, spec.payload_kg)
    cfg = ACOConfig(
        ants=int(p.get("ants", 30)),
        iterations=0,
        alpha=float(p.get("alpha", 1.0)),
        beta=float(p.get("beta", 2.0)),
        assignment_alpha=float(p.get("assignment_alpha", 1.0)),
        service_alpha=float(p.get("service_alpha", 1.0)),
        service_beta=float(p.get("service_beta", 1.5)),
        # Pin the current ACO service policy explicitly: 100% strategy B with
        # the full workload-aware bias (kappa=1).  This keeps these experiments
        # on the current algorithm even if the solver defaults change later.
        service_policy_mode=str(p.get("service_policy_mode", "fixed")),
        adaptive_service_fraction=float(p.get("adaptive_service_fraction", 1.0)),
        service_proxy_strength=float(p.get("service_proxy_strength", 1.0)),
        q0=float(p.get("q0", 0.85)),
        rho=float(p.get("rho", 0.10)),
        xi=float(p.get("xi", 0.10)),
        sector_bias=float(p.get("sector_bias", 2.0)),
        candidate_list_size=int(p.get("candidate_list_size", 18)),
        restart_after=int(p.get("restart_after", 45)),
        seed=int(spec.seed),
        verbose_every=0,
        time_limit_seconds=float(spec.time_budget_seconds),
    )
    solver = AntColonySolver(problem, cfg)
    initial = float(solver.global_best.objective)
    if recorder is not None:
        recorder.initial(initial, 0, 0)
        solver.history_callback = recorder
    solver.cfg.time_limit_seconds = max(
        1e-9, float(spec.time_budget_seconds) - (time.perf_counter() - start)
    )
    best, history, solver_runtime = solver.solve()
    ev = evaluate_solution(best.vehicle_routes, best.drone_routes, problem, True)
    result = _base_result(spec, start)
    result.update(summarize_evaluation(
        ev, problem, spec.battery_wh, spec.payload_kg,
        vehicle_routes=best.vehicle_routes,
    ))
    result.update({
        "initial_makespan_s": initial,
        "iterations_or_generations": int(solver.iterations_used),
        "completed_iterations": int(len(history)),
        "evaluations": int(solver.constructed_ants),
        "solver_runtime_seconds": float(solver_runtime),
        "vehicle_routes": [[int(x) for x in r] for r in best.vehicle_routes],
        "drone_routes": best.drone_routes,
        "master_routes": [[int(x) for x in r] for r in best.code.master_routes],
    })
    if recorder is not None:
        recorder.final(
            result.get("best_makespan_s"), solver.iterations_used,
            solver.constructed_ants,
        )
    return result


def _run_ga(spec: RunSpec, start: float, recorder=None) -> Dict[str, Any]:
    p = spec.params
    problem = load_ga_problem(spec.dataset_path, spec.battery_wh, spec.payload_kg)
    cfg = GAConfig(
        population_size=int(p.get("population_size", 50)),
        generations=0,
        elite_size=int(p.get("elite_size", 6)),
        tournament_size=int(p.get("tournament_size", 4)),
        seed=int(spec.seed),
        time_limit_seconds=float(spec.time_budget_seconds),
    )
    ga = ClassicGA(problem, cfg)
    if recorder is not None:
        ga.history_callback = recorder
    cfg.time_limit_seconds = max(
        1e-9, float(spec.time_budget_seconds) - (time.perf_counter() - start)
    )
    best = ga.solve(verbose=False)
    sol = best.solution or ga_decode(problem, best)
    ev = evaluate_solution(sol.vehicle_routes, sol.drone_routes, problem, True)
    result = _base_result(spec, start)
    result.update(summarize_evaluation(
        ev, problem, spec.battery_wh, spec.payload_kg,
        vehicle_routes=sol.vehicle_routes,
    ))
    result.update({
        "initial_makespan_s": (
            float(getattr(ga, "initial_best", math.inf))
            if math.isfinite(float(getattr(ga, "initial_best", math.inf))) else None
        ),
        "iterations_or_generations": int(len(ga.history)),
        "generations_completed": int(len(ga.history)),
        "evaluations": int(getattr(ga, "evaluation_count", 0)),
        "solver_runtime_seconds": float(getattr(ga, "runtime_seconds", 0.0)),
        "vehicle_routes": [[int(x) for x in r] for r in sol.vehicle_routes],
        "drone_routes": sol.drone_routes,
    })
    if recorder is not None:
        recorder.final(
            result.get("best_makespan_s"), len(ga.history),
            getattr(ga, "evaluation_count", 0),
        )
    return result


def _run_alns(spec: RunSpec, start: float, recorder=None) -> Dict[str, Any]:
    p = spec.params
    model = build_alns_model(
        spec.dataset_path, spec.seed, spec.battery_wh, spec.payload_kg
    )
    initial_routes, initial_drones, initial_time = model.generate_initial_solution(
        initial_strategy=str(p.get("initial_strategy", "farthest")),
        clustering_method=str(p.get("clustering", "kmeans")),
        verbose=False,
    )
    initial_ev = model.evaluate_solution(initial_routes, initial_drones)
    if recorder is not None and initial_ev.feasible:
        recorder.initial(float(initial_ev.makespan), 0, 0)
    remaining = max(1e-9, float(spec.time_budget_seconds) - (time.perf_counter() - start))
    best_routes, best_drones, best_time, info = model.local_search(
        initial_routes, initial_drones,
        iterations=0,
        time_limit_seconds=remaining,
        verbose=False,
        return_info=True,
        history_callback=recorder,
    )
    ev = model.evaluate_solution(best_routes, best_drones)
    result = _base_result(spec, start)
    result.update(summarize_evaluation(
        ev, model, spec.battery_wh, spec.payload_kg,
        vehicle_routes=best_routes,
    ))
    result.update({
        "initial_makespan_s": float(initial_time) if math.isfinite(float(initial_time)) else None,
        "iterations_or_generations": int(info.get("iterations_used", 0)),
        "evaluations": int(info.get("evaluations", 0)),
        "solver_runtime_seconds": float(info.get("search_runtime_seconds", 0.0)),
        "vehicle_routes": [[int(x) for x in r] for r in best_routes],
        "drone_routes": best_drones,
    })
    if recorder is not None:
        recorder.final(
            result.get("best_makespan_s"), int(info.get("iterations_used", 0)),
            int(info.get("evaluations", 0)),
        )
    return result


def _run_pso(spec: RunSpec, start: float, recorder=None) -> Dict[str, Any]:
    p = spec.params
    problem = load_pso_problem(spec.dataset_path, spec.battery_wh, spec.payload_kg)
    remaining = max(1e-9, float(spec.time_budget_seconds) - (time.perf_counter() - start))
    cfg = PSOConfig(
        swarm_size=int(p.get("swarm_size", 30)),
        iterations=0,
        time_limit_seconds=remaining,
        seed=int(spec.seed),
        chi=float(p.get("chi", 0.7298437881283576)),
        phi1=float(p.get("phi1", 2.05)),
        phi2=float(p.get("phi2", 2.05)),
        velocity_limit=float(p.get("velocity_limit", 0.25)),
        seed_fraction=float(p.get("seed_fraction", 0.5)),
        perturbation_sigma=float(p.get("perturbation_sigma", 0.15)),
        cache_size=int(p.get("cache_size", 1024)),
    )
    solver = ParticleSwarmSolver(problem, cfg)
    if recorder is not None:
        solver.history_callback = recorder
    best = solver.solve()
    ev = evaluate_solution(best.vehicle_routes, best.drone_routes, problem, True)
    result = _base_result(spec, start)
    result.update(summarize_evaluation(
        ev, problem, spec.battery_wh, spec.payload_kg,
        vehicle_routes=best.vehicle_routes,
    ))
    result.update({
        "initial_makespan_s": float(solver.initial.objective),
        "initial_population_best_s": float(getattr(
            solver, "initial_population_best", solver.initial.objective)),
        "iterations_or_generations": int(solver.iterations_used),
        "completed_iterations": int(solver.iterations_completed),
        "evaluations": int(solver.evaluations),
        "solver_runtime_seconds": float(getattr(solver, "runtime_seconds", 0.0)),
        "swarm_size_used": int(getattr(solver, "swarm_size_used", 0)),
        "proposals": int(getattr(solver, "proposals", 0)),
        "cache_hits": int(getattr(solver, "cache_hits", 0)),
        "tabu_stats": dict(getattr(solver, "tabu_stats", {})),
        "algorithm_variant": "independent-random-key-greedy-insertion-tabu20-v3",
        "initialization_method": "pso_balanced_insertion_drone_hints",
        "vehicle_routes": [[int(x) for x in r] for r in best.vehicle_routes],
        "drone_routes": best.drone_routes,
        "master_routes": [[int(x) for x in r] for r in best.code.master_routes],
    })
    # The PSO solver emits its own final history event after exact validation.
    return result


def _run_vnd(spec: RunSpec, start: float, recorder=None) -> Dict[str, Any]:
    p = spec.params
    model = build_alns_model(
        spec.dataset_path, spec.seed, spec.battery_wh, spec.payload_kg
    )
    initial_routes, initial_drones, initial_time = model.generate_initial_solution(
        initial_strategy=str(p.get("initial_strategy", "farthest")),
        clustering_method=str(p.get("clustering", "sweep")),
        verbose=False,
    )
    initial_ev = model.evaluate_solution(initial_routes, initial_drones)
    if not initial_ev.feasible:
        return _infeasible_result(
            spec, start, "initial solution infeasible: " + initial_ev.reason,
            method_status="initial_infeasible",
        )
    remaining = max(1e-9, float(spec.time_budget_seconds) - (time.perf_counter() - start))
    solver = UserNeighborhoodVND(
        model=model,
        trials_per_neighborhood=int(p.get("trials_per_neighborhood", 20)),
        max_evaluations=0,
        time_limit_seconds=remaining,
        seed=int(spec.seed),
        verbose=False,
        repeat_until_budget=True,
    )
    if recorder is not None:
        solver.history_callback = recorder
    result_obj = solver.run(initial_routes, initial_drones)
    ev = model.evaluate_solution(result_obj.vehicle_routes, result_obj.drone_routes)
    result = _base_result(spec, start)
    result.update(summarize_evaluation(
        ev, model, spec.battery_wh, spec.payload_kg,
        vehicle_routes=result_obj.vehicle_routes,
    ))
    result.update({
        "initial_makespan_s": float(result_obj.initial_time),
        "iterations_or_generations": int(result_obj.neighborhood_passes),
        "improvements": int(result_obj.improvements),
        "evaluations": int(result_obj.evaluations),
        "solver_runtime_seconds": float(result_obj.search_runtime_seconds),
        "vehicle_routes": [[int(x) for x in r] for r in result_obj.vehicle_routes],
        "drone_routes": result_obj.drone_routes,
    })
    if recorder is not None:
        recorder.final(
            result.get("best_makespan_s"), result_obj.neighborhood_passes,
            result_obj.evaluations,
        )
    return result


def run_algorithm(spec: RunSpec, recorder=None) -> Dict[str, Any]:
    """Run one algorithm and return a JSON-safe result dict.

    ``recorder`` is optional.  When supplied it must expose the callable
    convergence interface used by the convergence experiment.
    """
    start = getattr(recorder, "start_time", None)
    if start is None:
        start = time.perf_counter()
    name = spec.algorithm.upper()
    if name == "ACO":
        return _run_aco(spec, start, recorder)
    if name == "GA":
        return _run_ga(spec, start, recorder)
    if name == "ALNS":
        return _run_alns(spec, start, recorder)
    if name == "PSO":
        return _run_pso(spec, start, recorder)
    if name == "VND":
        return _run_vnd(spec, start, recorder)
    raise ValueError("unknown algorithm %r" % spec.algorithm)
