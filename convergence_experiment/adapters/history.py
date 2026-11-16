"""Real wall-clock best-so-far recorder for the five comparison algorithms.

The recorder is deliberately independent of every optimisation mechanism.  It is
passed to a solver as an optional callback; when it is not passed, the solver
behaves exactly as before.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

FIELDS = [
    "algorithm",
    "dataset",
    "seed",
    "time_budget_seconds",
    "elapsed_time_seconds",
    "best_makespan_s",
    "iteration_or_generation",
    "evaluations",
    "event_type",
]

_EPS = 1e-9


class HistoryRecorder:
    """Collect first-feasible and every strict best-so-far improvement."""

    def __init__(
        self,
        algorithm: str,
        dataset: str,
        seed: int,
        time_budget_seconds: float,
        start_time: Optional[float] = None,
    ):
        self.algorithm = str(algorithm)
        self.dataset = str(dataset)
        self.seed = int(seed)
        self.time_budget_seconds = float(time_budget_seconds)
        self.start_time = (
            float(start_time) if start_time is not None else time.perf_counter()
        )
        self.events: List[Dict[str, Any]] = []

    @property
    def last_best(self) -> float:
        return float(self.events[-1]["best_makespan_s"]) if self.events else math.inf

    @property
    def last_time(self) -> float:
        return float(self.events[-1]["elapsed_time_seconds"]) if self.events else 0.0

    def _append(
        self,
        best_makespan_s,
        iteration_or_generation,
        evaluations,
        event_type: str,
    ) -> None:
        try:
            best = float(best_makespan_s)
        except (TypeError, ValueError):
            return
        if not math.isfinite(best) or best <= 0.0:
            return
        requested = str(event_type)
        if not self.events:
            event_type = "initial"
        else:
            if best > self.last_best + _EPS:
                return
            if abs(best - self.last_best) <= _EPS and requested != "final":
                return
            event_type = "final" if requested == "final" else "improvement"

        elapsed = time.perf_counter() - self.start_time
        elapsed = max(0.0, float(elapsed))
        if self.events:
            # Strictly increasing timestamps, even if the clock has low resolution.
            elapsed = max(elapsed, self.last_time + _EPS)
        self.events.append({
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "seed": self.seed,
            "time_budget_seconds": self.time_budget_seconds,
            "elapsed_time_seconds": float(elapsed),
            "best_makespan_s": float(best),
            "iteration_or_generation": int(iteration_or_generation or 0),
            "evaluations": int(evaluations or 0),
            "event_type": event_type,
        })

    def __call__(self, event: Dict[str, Any]) -> None:
        self._append(
            event.get("best_makespan_s"),
            event.get("iteration_or_generation", 0),
            event.get("evaluations", 0),
            event.get("event_type", "improvement"),
        )

    def initial(self, best_makespan_s, iteration_or_generation=0, evaluations=0) -> None:
        self._append(best_makespan_s, iteration_or_generation, evaluations, "initial")

    def final(self, best_makespan_s, iteration_or_generation=0, evaluations=0) -> None:
        self._append(best_makespan_s, iteration_or_generation, evaluations, "final")

    def rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.events]
