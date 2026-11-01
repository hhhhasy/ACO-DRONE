"""General Variable Neighborhood Search (GVNS) using the user's 7 ALNS neighborhoods.

The neighborhood definitions are NOT reimplemented here.  They are taken directly
from ``model.operator_pairs`` so VNS, VND and ALNS can share exactly the same move
machinery and the same physical feasibility logic.

Search framework
----------------
Outer VNS:
    k = 1..K
    x_shake = Shake_k(x)              # may be worse than x
    x_local = RVND(x_shake)           # strict descent
    if f(x_local) < f(x):
        x = x_local; k = 1
    else:
        k = k + 1

Because the user's seven operators are stochastic destroy/repair generators rather
than finite deterministic neighborhoods, both shaking and RVND sample each selected
operator several times.  RVND randomizes the order in which neighborhoods are
visited, which prevents N1 from permanently dominating merely because it appears
first in ``model.operator_pairs``.

No simulated-annealing acceptance, ALNS adaptive weights, score updates or roulette
operator selection are used.  A worsened *shaken* solution is allowed only as an
intermediate starting point for RVND; the incumbent is replaced only by a strictly
better local optimum.  Objective and feasibility are always evaluated by the shared
model/evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
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
class VNSResult:
    vehicle_routes: List[List[int]]
    drone_routes: List[list]
    best_time: float
    initial_time: float
    evaluations: int
    accepted_improvements: int
    vns_iterations: int
    completed_no_improve_cycles: int
    terminated_by_evaluation_limit: bool
    terminated_by_time_limit: bool
    terminated_by_iteration_limit: bool
    search_runtime_seconds: float
    history: List[dict] = field(default_factory=list)
    shaking_stats: List[dict] = field(default_factory=list)
    rvnd_stats: List[dict] = field(default_factory=list)


class UserNeighborhoodVNS:
    """GVNS + RVND using ``model.operator_pairs`` unchanged."""

    def __init__(
        self,
        model,
        shake_trials: int = 5,
        local_trials: int = 10,
        max_no_improve_cycles: int = 2,
        max_vns_iterations: int = 0,
        max_evaluations: int = 0,
        time_limit_seconds: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.shake_trials = max(1, int(shake_trials))
        self.local_trials = max(1, int(local_trials))
        # 0 disables the natural no-improvement stop.  This is useful for
        # equal-wall-clock comparisons, where VNS must keep shaking until the
        # common deadline instead of terminating early.
        self.max_no_improve_cycles = max(0, int(max_no_improve_cycles))
        self.max_vns_iterations = max(0, int(max_vns_iterations))
        self.max_evaluations = max(0, int(max_evaluations))
        self.time_limit_seconds = (
            None
            if time_limit_seconds is None or float(time_limit_seconds) <= 0
            else float(time_limit_seconds)
        )
        self.seed = int(model.seed if seed is None else seed)
        self.verbose = bool(verbose)

        self.operator_pairs = list(model.operator_pairs)
        if not self.operator_pairs:
            raise ValueError("model.operator_pairs is empty")

        self.evaluations = 0
        self.accepted_improvements = 0
        self.vns_iterations = 0
        self.limit_hit = False
        self.time_hit = False
        self.iteration_hit = False
        self.history_callback = None
        self._search_start = 0.0

    @staticmethod
    def _copy_vehicle_routes(routes: Sequence[Sequence[int]]) -> List[List[int]]:
        return [[int(x) for x in route] for route in routes]

    @staticmethod
    def _copy_drone_routes(drone_routes):
        return fast_copy_drone_routes(drone_routes)

    @staticmethod
    def _solution_signature(vehicle_routes, drone_routes) -> Tuple:
        veh = tuple(tuple(int(x) for x in r) for r in vehicle_routes)
        dro = []
        for group in drone_routes:
            g = []
            for s in group:
                if not isinstance(s, (list, tuple)) or len(s) != 4:
                    g.append(repr(s))
                    continue
                label, launch, delivery, rendez = s
                g.append((str(label), int(launch), tuple(int(x) for x in delivery), int(rendez)))
            dro.append(tuple(g))
        return veh, tuple(dro)

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

    def _iteration_available(self) -> bool:
        if self.max_vns_iterations > 0 and self.vns_iterations >= self.max_vns_iterations:
            self.iteration_hit = True
            return False
        return True

    def _objective(self, vehicle_routes, drone_routes) -> float:
        if not self._budget_available():
            return float("inf")
        self.evaluations += 1
        return float(self.model.calculate_solution_time_cost(vehicle_routes, drone_routes))

    def _apply_operator(self, vehicle_routes, drone_routes, op_idx):
        """One call to one original ALNS destroy/repair pair."""
        destroy_op, repair_op, op_name = self.operator_pairs[op_idx]
        base_routes = self._copy_vehicle_routes(vehicle_routes)
        base_drones = self._copy_drone_routes(drone_routes)

        destroyed_routes, destroyed_drones, removed = destroy_op(base_routes, base_drones)
        new_routes, new_drones = repair_op(destroyed_routes, destroyed_drones, removed)
        return (
            self._copy_vehicle_routes(new_routes),
            self._copy_drone_routes(new_drones),
            op_name,
        )

    def _shake(self, current_routes, current_drones, current_time, op_idx, shake_stats):
        """Sample N_k and return the first changed, feasible shaken solution.

        The shaken solution is allowed to have a worse objective than the incumbent.
        This is the mechanism that lets VNS leave a local optimum without using SA.
        """
        base_sig = self._solution_signature(current_routes, current_drones)
        _, _, op_name = self.operator_pairs[op_idx]
        best_changed = None

        for trial in range(1, self.shake_trials + 1):
            if not self._budget_available():
                break
            cand_routes, cand_drones, _ = self._apply_operator(
                current_routes, current_drones, op_idx
            )
            shake_stats[op_idx]["samples"] += 1
            cand_sig = self._solution_signature(cand_routes, cand_drones)
            if cand_sig == base_sig:
                shake_stats[op_idx]["unchanged"] += 1
                continue

            cand_time = self._objective(cand_routes, cand_drones)
            if not math.isfinite(cand_time):
                shake_stats[op_idx]["infeasible"] += 1
                continue

            # First valid changed point is the canonical VNS random shake.
            best_changed = (cand_routes, cand_drones, cand_time, trial, op_name)
            shake_stats[op_idx]["feasible_changed"] += 1
            if cand_time > current_time + EPS:
                shake_stats[op_idx]["worse_shakes"] += 1
            elif cand_time + EPS < current_time:
                shake_stats[op_idx]["better_shakes"] += 1
            else:
                shake_stats[op_idx]["equal_shakes"] += 1
            break

        if best_changed is None:
            return (
                self._copy_vehicle_routes(current_routes),
                self._copy_drone_routes(current_drones),
                current_time,
                0,
                op_name,
                False,
            )
        r, d, t, trial, name = best_changed
        return r, d, t, trial, name, True

    def _rvnd(self, start_routes, start_drones, start_time, rvnd_stats):
        """Randomized VND local search over all seven original neighborhoods."""
        current_routes = self._copy_vehicle_routes(start_routes)
        current_drones = self._copy_drone_routes(start_drones)
        current_time = float(start_time)
        current_sig = self._solution_signature(current_routes, current_drones)
        local_improvements = 0

        active = list(range(len(self.operator_pairs)))
        random.shuffle(active)

        while active and self._budget_available():
            # Random selection from the remaining neighborhoods is standard RVND.
            pos = random.randrange(len(active))
            op_idx = active[pos]
            _, _, op_name = self.operator_pairs[op_idx]
            rvnd_stats[op_idx]["visits"] += 1
            improved = False

            for trial in range(1, self.local_trials + 1):
                if not self._budget_available():
                    break
                cand_routes, cand_drones, _ = self._apply_operator(
                    current_routes, current_drones, op_idx
                )
                rvnd_stats[op_idx]["samples"] += 1

                # Avoid paying evaluator cost for exact no-op returns.
                cand_sig = self._solution_signature(cand_routes, cand_drones)
                if cand_sig == current_sig:
                    rvnd_stats[op_idx]["unchanged"] += 1
                    continue

                cand_time = self._objective(cand_routes, cand_drones)
                if not math.isfinite(cand_time):
                    rvnd_stats[op_idx]["infeasible"] += 1
                    continue

                # Strict descent inside RVND; no SA and no equal/worse move acceptance.
                if cand_time + EPS < current_time:
                    current_routes = cand_routes
                    current_drones = cand_drones
                    current_time = cand_time
                    current_sig = cand_sig
                    local_improvements += 1
                    rvnd_stats[op_idx]["improvements"] += 1
                    improved = True
                    if self.verbose:
                        print(
                            "    [RVND] N%d improved -> %.4f  (%s, trial=%d, eval=%d)"
                            % (op_idx + 1, current_time, op_name, trial, self.evaluations)
                        )
                    break

            if not self._budget_available():
                break

            if improved:
                # An improvement changes the local landscape: restore all neighborhoods.
                active = list(range(len(self.operator_pairs)))
                random.shuffle(active)
            else:
                # No sampled improvement: remove this neighborhood for this RVND pass.
                active.pop(pos)

        return current_routes, current_drones, current_time, local_improvements

    def run(self, initial_routes, initial_drones) -> VNSResult:
        random.seed(self.seed)
        if np is not None:
            np.random.seed(self.seed)

        self.evaluations = 0
        self.accepted_improvements = 0
        self.vns_iterations = 0
        self.limit_hit = False
        self.time_hit = False
        self.iteration_hit = False
        self._search_start = time.perf_counter()

        current_routes = self._copy_vehicle_routes(initial_routes)
        current_drones = self._copy_drone_routes(initial_drones)
        current_time = self._objective(current_routes, current_drones)
        if not math.isfinite(current_time):
            raise ValueError("initial VNS solution is infeasible")
        initial_time = current_time

        best_routes = self._copy_vehicle_routes(current_routes)
        best_drones = self._copy_drone_routes(current_drones)
        best_time = current_time
        self._last_emitted_best = float(best_time)
        self._emit_history("initial", 0)

        history = [{
            "iteration": 0,
            "cycle": 0,
            "k": 0,
            "shake_neighborhood": "initial",
            "shake_time": current_time,
            "local_time": current_time,
            "incumbent_time": current_time,
            "best_time": best_time,
            "accepted": True,
            "evaluations": self.evaluations,
        }]

        shaking_stats = [
            {
                "index": i + 1,
                "name": name,
                "samples": 0,
                "unchanged": 0,
                "infeasible": 0,
                "feasible_changed": 0,
                "better_shakes": 0,
                "equal_shakes": 0,
                "worse_shakes": 0,
            }
            for i, (_, _, name) in enumerate(self.operator_pairs)
        ]
        rvnd_stats = [
            {
                "index": i + 1,
                "name": name,
                "visits": 0,
                "samples": 0,
                "unchanged": 0,
                "infeasible": 0,
                "improvements": 0,
            }
            for i, (_, _, name) in enumerate(self.operator_pairs)
        ]

        no_improve_cycles = 0
        cycle = 0
        k = 0

        while self._budget_available() and self._iteration_available():
            # Because every accepted improvement immediately resets k to N1, reaching
            # k > K means that a complete N1..NK sweep since the last improvement
            # produced no accepted solution.  That is one natural VNS no-improve cycle.
            if k >= len(self.operator_pairs):
                cycle += 1
                no_improve_cycles += 1
                if self.verbose:
                    print(
                        "[VNS] completed no-improve cycle %d/%d at %.4f"
                        % (no_improve_cycles, self.max_no_improve_cycles, current_time)
                    )
                if (
                    self.max_no_improve_cycles > 0
                    and no_improve_cycles >= self.max_no_improve_cycles
                ):
                    break
                k = 0
                continue

            self.vns_iterations += 1
            selected_k = k
            if self.verbose:
                print(
                    "[VNS] iter=%d k=N%d incumbent=%.4f"
                    % (self.vns_iterations, k + 1, current_time)
                )

            shake_routes, shake_drones, shake_time, shake_trial, op_name, changed = self._shake(
                current_routes, current_drones, current_time, k, shaking_stats
            )
            if not self._budget_available() and not changed:
                break

            local_routes, local_drones, local_time, local_improvements = self._rvnd(
                shake_routes, shake_drones, shake_time, rvnd_stats
            )

            accepted = local_time + EPS < current_time
            old_incumbent = current_time
            if accepted:
                current_routes = self._copy_vehicle_routes(local_routes)
                current_drones = self._copy_drone_routes(local_drones)
                current_time = float(local_time)
                self.accepted_improvements += 1
                no_improve_cycles = 0

                if current_time + EPS < best_time:
                    best_routes = self._copy_vehicle_routes(current_routes)
                    best_drones = self._copy_drone_routes(current_drones)
                    best_time = current_time
                    self._last_emitted_best = float(best_time)
                    self._emit_history("improvement", self.vns_iterations)

                if self.verbose:
                    print(
                        "[VNS] ACCEPT N%d %.4f -> %.4f  "
                        "(shake=%.4f, RVND improvements=%d, eval=%d)"
                        % (
                            k + 1, old_incumbent, current_time, shake_time,
                            local_improvements, self.evaluations,
                        )
                    )
                # Classical VNS neighborhood change: improvement => restart at N1.
                k = 0
            else:
                if self.verbose:
                    relation = "no-change" if not changed else "rejected"
                    print(
                        "[VNS] %s N%d shake=%.4f local=%.4f incumbent=%.4f"
                        % (relation, k + 1, shake_time, local_time, current_time)
                    )
                k += 1

            history.append({
                "iteration": self.vns_iterations,
                "cycle": cycle,
                "k": selected_k + 1,
                "shake_neighborhood": op_name,
                "shake_trial": shake_trial,
                "shake_changed": bool(changed),
                "shake_time": float(shake_time),
                "local_time": float(local_time),
                "local_improvements": int(local_improvements),
                "incumbent_time": float(current_time),
                "best_time": float(best_time),
                "accepted": bool(accepted),
                "evaluations": int(self.evaluations),
            })

        runtime = time.perf_counter() - self._search_start
        return VNSResult(
            vehicle_routes=self._copy_vehicle_routes(best_routes),
            drone_routes=self._copy_drone_routes(best_drones),
            best_time=float(best_time),
            initial_time=float(initial_time),
            evaluations=int(self.evaluations),
            accepted_improvements=int(self.accepted_improvements),
            vns_iterations=int(self.vns_iterations),
            completed_no_improve_cycles=int(no_improve_cycles),
            terminated_by_evaluation_limit=bool(self.limit_hit),
            terminated_by_time_limit=bool(self.time_hit),
            terminated_by_iteration_limit=bool(self.iteration_hit),
            search_runtime_seconds=float(runtime),
            history=history,
            shaking_stats=shaking_stats,
            rvnd_stats=rvnd_stats,
        )


# Short aliases for convenience.
GeneralVNS = UserNeighborhoodVNS
GVNS = UserNeighborhoodVNS
