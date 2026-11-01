"""VND/RVND-style descent using the *original ALNS neighborhood operators*.

This module deliberately does NOT define new Relocate/Swap/2-opt neighborhoods.
Instead it reuses ``model.operator_pairs`` exactly as they are defined by the
original ALNS model:

    d1-r1  random single-customer removal / reinsertion
    d2-r2  random two-customer swap
    d3-r3  route reversal
    d4-r4  drone launch/rendezvous reassignment
    d5-r5  generate a new drone sortie
    d6-r6  Shaw related removal + Regret-2 repair
    d7-r7  slack-aware drone launch/rendezvous window adjustment

The *search control* is VND descent:
  - neighborhoods are visited in a fixed order;
  - only strictly improving, feasible candidates are accepted;
  - after an improvement, search restarts at the first neighborhood;
  - otherwise it advances to the next neighborhood.

Important methodological note
-----------------------------
The user's neighborhoods are stochastic destroy/repair generators rather than
finite deterministic neighborhoods that can be exhaustively enumerated.  Thus
``trials_per_neighborhood`` samples each neighborhood several times.  When all
seven sampled neighborhoods fail to improve, the procedure stops.  This keeps
the user's neighborhood definitions intact, but the stopping condition is a
sampled/stochastic local optimum rather than an exhaustive deterministic one.

No simulated annealing acceptance, adaptive operator weights, roulette operator
selection, or ALNS score update is used here.  Feasibility and objective remain
those of the shared evaluator through ``model.calculate_solution_time_cost``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math
import random
import time

try:
    import numpy as np
except Exception:  # numpy is already required by the original project
    np = None

from utils import fast_copy_drone_routes

EPS = 1e-9


@dataclass
class VNDResult:
    vehicle_routes: List[List[int]]
    drone_routes: List[list]
    best_time: float
    initial_time: float
    evaluations: int
    improvements: int
    neighborhood_passes: int
    terminated_by_limit: bool
    terminated_by_time: bool
    search_runtime_seconds: float
    history: List[dict] = field(default_factory=list)
    neighborhood_stats: List[dict] = field(default_factory=list)


class UserNeighborhoodVND:
    """Strict-improvement VND using ``model.operator_pairs`` unchanged."""

    def __init__(
        self,
        model,
        trials_per_neighborhood: int = 20,
        max_evaluations: int = 0,
        time_limit_seconds: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        repeat_until_budget: bool = False,
    ):
        self.model = model
        self.trials_per_neighborhood = max(1, int(trials_per_neighborhood))
        self.max_evaluations = max(0, int(max_evaluations))
        self.time_limit_seconds = (
            None if time_limit_seconds is None or float(time_limit_seconds) <= 0
            else float(time_limit_seconds)
        )
        self.seed = int(model.seed if seed is None else seed)
        self.verbose = bool(verbose)
        self.repeat_until_budget = bool(repeat_until_budget)

        # Exact neighborhood/operator pairs already defined by the user's model.
        self.operator_pairs = list(model.operator_pairs)
        if not self.operator_pairs:
            raise ValueError("model.operator_pairs is empty")

        self.evaluations = 0
        self.improvements = 0
        self.limit_hit = False
        self.time_hit = False
        self.history_callback = None
        self._search_start = 0.0

    @staticmethod
    def _copy_vehicle_routes(routes: Sequence[Sequence[int]]) -> List[List[int]]:
        return [[int(x) for x in route] for route in routes]

    @staticmethod
    def _copy_drone_routes(drone_routes):
        return fast_copy_drone_routes(drone_routes)

    def _emit_history(self, event_type: str, iteration: int) -> None:
        callback = getattr(self, "history_callback", None)
        if callback is None:
            return
        callback({
            "best_makespan_s": float(self._last_emitted_best),
            "iteration_or_generation": int(iteration),
            "evaluations": int(self.evaluations),
            "event_type": str(event_type),
        })

    def _time_up(self) -> bool:
        if self.time_limit_seconds is None:
            return False
        if time.perf_counter() - self._search_start >= self.time_limit_seconds:
            self.time_hit = True
            return True
        return False

    def _budget_available(self) -> bool:
        if self.max_evaluations > 0 and self.evaluations >= self.max_evaluations:
            self.limit_hit = True
            return False
        if self._time_up():
            return False
        return True

    def _objective(self, vehicle_routes, drone_routes) -> float:
        if not self._budget_available():
            return float("inf")
        self.evaluations += 1
        return float(self.model.calculate_solution_time_cost(vehicle_routes, drone_routes))

    def _sample_neighborhood(self, current_routes, current_drones, op_idx):
        """Generate one candidate with one of the user's original operator pairs."""
        destroy_op, repair_op, op_name = self.operator_pairs[op_idx]

        # Defensive copies: some original operators return or touch nested objects.
        base_routes = self._copy_vehicle_routes(current_routes)
        base_drones = self._copy_drone_routes(current_drones)

        destroyed_routes, destroyed_drones, removed = destroy_op(base_routes, base_drones)
        new_routes, new_drones = repair_op(destroyed_routes, destroyed_drones, removed)

        # Normalize container types so accepted states never alias operator internals.
        new_routes = self._copy_vehicle_routes(new_routes)
        new_drones = self._copy_drone_routes(new_drones)
        return new_routes, new_drones, op_name

    def run(self, initial_routes, initial_drones) -> VNDResult:
        random.seed(self.seed)
        if np is not None:
            np.random.seed(self.seed)

        self.evaluations = 0
        self.improvements = 0
        self.limit_hit = False
        self.time_hit = False
        self._search_start = time.perf_counter()

        current_routes = self._copy_vehicle_routes(initial_routes)
        current_drones = self._copy_drone_routes(initial_drones)
        current_time = self._objective(current_routes, current_drones)
        if not math.isfinite(current_time):
            raise ValueError("initial VND solution is infeasible")
        initial_time = current_time
        self._last_emitted_best = float(current_time)
        self._emit_history("initial", 0)

        history = [{
            "step": 0,
            "neighborhood_index": 0,
            "neighborhood": "initial",
            "trial": 0,
            "makespan": current_time,
            "evaluations": self.evaluations,
        }]
        stats = [
            {"index": i + 1, "name": name, "samples": 0, "improvements": 0}
            for i, (_, _, name) in enumerate(self.operator_pairs)
        ]

        k = 0
        passes = 0
        completed_cycles = 0
        while True:
            if not self._budget_available():
                break
            if k >= len(self.operator_pairs):
                completed_cycles += 1
                if self.repeat_until_budget and (self.time_limit_seconds is not None or self.max_evaluations > 0):
                    # The seven neighborhoods are stochastic samplers.  In timed
                    # experiments, one unsuccessful sampled sweep is not treated
                    # as a hard local-optimum proof; restart at N1 and keep using
                    # the remaining wall-clock budget.
                    k = 0
                    continue
                break

            passes += 1
            improved = False
            _, _, op_name = self.operator_pairs[k]

            for trial in range(1, self.trials_per_neighborhood + 1):
                if not self._budget_available():
                    break

                cand_routes, cand_drones, _ = self._sample_neighborhood(
                    current_routes, current_drones, k
                )
                cand_time = self._objective(cand_routes, cand_drones)
                stats[k]["samples"] += 1

                # evaluator returns inf for malformed/physically infeasible candidates.
                if not math.isfinite(cand_time):
                    continue

                # Pure descent: no SA, no equal/worse acceptance.
                if cand_time + EPS < current_time:
                    old_time = current_time
                    current_routes = cand_routes
                    current_drones = cand_drones
                    current_time = cand_time
                    self.improvements += 1
                    stats[k]["improvements"] += 1
                    improved = True
                    self._last_emitted_best = float(current_time)
                    self._emit_history("improvement", self.improvements)

                    history.append({
                        "step": self.improvements,
                        "neighborhood_index": k + 1,
                        "neighborhood": op_name,
                        "trial": trial,
                        "makespan": current_time,
                        "improvement": old_time - current_time,
                        "evaluations": self.evaluations,
                    })
                    if self.verbose:
                        print(
                            "[VND] N%d improved %.4f -> %.4f  (%s, trial=%d, eval=%d)"
                            % (k + 1, old_time, current_time, op_name, trial, self.evaluations)
                        )
                    break

            if self.limit_hit or self.time_hit:
                break

            if improved:
                # Classical VND control: restart from the first neighborhood.
                k = 0
            else:
                k += 1

        runtime = time.perf_counter() - self._search_start
        return VNDResult(
            vehicle_routes=self._copy_vehicle_routes(current_routes),
            drone_routes=self._copy_drone_routes(current_drones),
            best_time=current_time,
            initial_time=initial_time,
            evaluations=self.evaluations,
            improvements=self.improvements,
            neighborhood_passes=passes,
            terminated_by_limit=self.limit_hit,
            terminated_by_time=self.time_hit,
            search_runtime_seconds=runtime,
            history=history,
            neighborhood_stats=stats,
        )


# Backward-friendly name for users replacing the previous vnd_solver.py.
ClassicalVND = UserNeighborhoodVND
