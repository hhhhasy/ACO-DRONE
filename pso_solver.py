"""Independent random-key PSO with balanced-insertion seed and FIFO tabu.
No other optimization solver is imported. Common physical evaluation is retained.
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
import time
import numpy as np
from pso_decoder import ParticleCode, decode, load_problem, build_initial_code
from evaluator import evaluate_solution


_TABU_SIZE = 20  # FIFO solution memory; 0 disables it.


@dataclass
class PSOConfig:
    swarm_size: int = 30
    iterations: int = 500
    time_limit_seconds: float = 0.0
    seed: int = 13
    chi: float = 0.7298437881283576
    phi1: float = 2.05
    phi2: float = 2.05
    velocity_limit: float = 0.25
    seed_fraction: float = 0.5
    perturbation_sigma: float = 0.15
    cache_size: int = 1024


class ParticleSwarmSolver:
    def __init__(self, problem, config=None):
        self.problem = problem
        self.cfg = config or PSOConfig()
        c = self.cfg
        if c.swarm_size < 2 or c.cache_size < 0:
            raise ValueError('swarm_size >= 2 and cache_size >= 0 required')
        if not math.isfinite(c.time_limit_seconds) or c.time_limit_seconds < 0:
            raise ValueError('invalid time limit')
        if c.iterations <= 0 and c.time_limit_seconds <= 0:
            raise ValueError('unlimited iterations require a positive time limit')
        for value in (c.chi, c.phi1, c.phi2, c.velocity_limit, c.perturbation_sigma):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('PSO coefficients must be finite and positive')
        if not 0 <= c.seed_fraction <= 1 or c.velocity_limit > 1:
            raise ValueError('invalid seed fraction or velocity limit')
        if problem.num_vehicles <= 0 or problem.num_customers < 1 or problem.num_drones_per_vehicle < 0:
            raise ValueError('requires customers, trucks, and nonnegative drone count')
        # Optional convergence callback; observation only, never affects search.
        self.history_callback = None

    def encode(self, code):
        n, k = self.problem.num_customers, self.problem.num_vehicles
        d = max(1, self.problem.num_drones_per_vehicle)
        x = np.zeros((4, n), dtype=float)
        for vehicle, route in enumerate(code.master_routes):
            for rank, customer in enumerate(route):
                j = customer - 1
                x[0, j] = (vehicle + 0.5) / k
                x[1, j] = (rank + 1) / (len(route) + 1)
                x[2, j] = 0.75 if code.service[customer] else 0.25
                x[3, j] = (code.drone[customer] % d + 0.5) / d
        return x

    def to_code(self, x):
        p = self.problem
        k, d = p.num_vehicles, p.num_drones_per_vehicle
        groups = [[] for _ in range(k)]
        assignment = np.minimum((x[0] * k).astype(int), k - 1)
        for j in range(p.num_customers):
            groups[int(assignment[j])].append(j + 1)
        for route in groups:
            route.sort(key=lambda c: (float(x[1, c-1]), c))
        service = [0] + [int(d > 0 and x[2, j] >= 0.5 and
            p.customer_demands[j+1] <= p.drone_payload_capacity + 1e-9)
            for j in range(p.num_customers)]
        labels = [0] + [min(int(x[3, j] * max(1, d)), max(1, d)-1)
                        for j in range(p.num_customers)]
        return ParticleCode(groups, service, labels)

    @staticmethod
    def signature(code):
        return (tuple(tuple(r) for r in code.master_routes), tuple(code.service),
                tuple(di if mode else -1 for di, mode in zip(code.drone, code.service)))

    def _evaluate(self, x, use_tabu=True):
        code = self.to_code(x)
        key = self.signature(code)
        self.proposals += 1
        if use_tabu and _TABU_SIZE > 0 and key in self._tabu:
            remembered = self._tabu[key]
            if remembered.objective < self._tabu_best - 1e-9:
                self.tabu_stats["aspirations"] += 1
                return remembered.clone()
            self._tabu_streak += 1
            if self._tabu_streak < _TABU_SIZE:
                self.tabu_stats["rejections"] += 1
                return None
            del self._tabu[key]
            self.tabu_stats["relaxations"] += 1
        self._tabu_streak = 0
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            result = self.cache[key].clone()
            self._remember_tabu(key, result)
            return result
        result = decode(self.problem, code)
        self.evaluations += 1
        if not result.feasible or not math.isfinite(result.objective):
            raise RuntimeError('PSO decoder failed: ' + result.evaluation.reason)
        if self.cfg.cache_size:
            self.cache[key] = result.clone()
            while len(self.cache) > self.cfg.cache_size:
                self.cache.popitem(last=False)
        self._remember_tabu(key, result)
        return result

    def _remember_tabu(self, key, result):
        if _TABU_SIZE > 0:
            self._tabu[key] = result.clone()
            while len(self._tabu) > _TABU_SIZE:
                self._tabu.popitem(last=False)
            self.tabu_stats["admissions"] += 1

    def _emit_history(self, best, event_type, iteration):
        """Emit one optional best-so-far event; does not affect the algorithm."""
        callback = getattr(self, "history_callback", None)
        if callback is None:
            return
        value = float(best.objective)
        if not math.isfinite(value):
            return
        callback({
            "best_makespan_s": value,
            "iteration_or_generation": int(iteration),
            "evaluations": int(getattr(self, "evaluations", 0)),
            "event_type": str(event_type),
        })

    def solve(self):
        started = time.perf_counter()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.cache = OrderedDict()
        self._tabu = OrderedDict()
        self._tabu_streak = 0
        self.tabu_stats = dict(rejections=0, aspirations=0, admissions=0, relaxations=0)
        self.proposals = self.evaluations = self.cache_hits = 0
        self.iterations_used = self.iterations_completed = 0
        self.history = []
        def expired():
            return c.time_limit_seconds > 0 and time.perf_counter()-started >= c.time_limit_seconds
        # Own deterministic route construction and drone-mode hints.
        # This baseline construction/decoding is inside the total budget.
        initial_code = build_initial_code(self.problem)
        self.initial = decode(self.problem, initial_code)
        best = self.initial.clone()
        self._tabu_best = best.objective
        self._remember_tabu(self.signature(best.code), best)
        seed_x = self.encode(best.code)
        # No injected fitness for a different genotype: exact code round-trip.
        if self.signature(self.to_code(seed_x)) != self.signature(best.code):
            raise RuntimeError('PSO seed cannot round-trip through particle encoding')
        best_x = seed_x.copy()
        self.initialization_runtime_seconds = time.perf_counter()-started
        self.history.append(dict(iteration=0, elapsed_seconds=time.perf_counter()-started,
                                 global_best=best.objective))
        self._emit_history(best, "initial", 0)
        positions = [seed_x.copy()]
        pbest = [seed_x.copy()]
        fitness = [best.objective]
        self.cache[self.signature(best.code)] = best.clone()
        if c.cache_size == 0:
            self.cache.clear()
        # Seed perturbations and uniform particles; only seed has a precomputed fitness.
        for i in range(1, c.swarm_size):
            if expired():
                break
            if i < max(1, round(c.seed_fraction*c.swarm_size)):
                x = np.clip(seed_x+self.rng.normal(0,c.perturbation_sigma,seed_x.shape),0,1)
            else:
                x = self.rng.random(seed_x.shape)
            sol = self._evaluate(x, use_tabu=False)
            if sol is None:
                continue
            positions.append(x); pbest.append(x.copy()); fitness.append(sol.objective)
            if sol.objective < best.objective - 1e-9:
                best, best_x = sol.clone(), x.copy()
                self._tabu_best = best.objective
                self._emit_history(best, "improvement", 0)
        self.initial_population_best = best.objective
        self.swarm_size_used = len(positions)
        x = np.asarray(positions)
        pb = np.asarray(pbest)
        pf = np.asarray(fitness)
        velocity = self.rng.uniform(-c.velocity_limit,c.velocity_limit,x.shape)
        it = 0
        while not expired() and (c.iterations <= 0 or it < c.iterations):
            it += 1
            self.iterations_used = it
            guide = best_x.copy()  # synchronous global guide for this iteration
            completed = True
            for i in range(len(x)):
                if expired():
                    completed = False
                    break
                r1, r2 = self.rng.random(x[i].shape), self.rng.random(x[i].shape)
                velocity[i] = c.chi*(velocity[i]+c.phi1*r1*(pb[i]-x[i])+c.phi2*r2*(guide-x[i]))
                velocity[i] = np.clip(velocity[i],-c.velocity_limit,c.velocity_limit)
                proposed = x[i]+velocity[i]
                hit = (proposed < 0) | (proposed > 1)
                x[i] = np.clip(proposed,0,1)
                velocity[i][hit] = 0.0
                sol = self._evaluate(x[i])
                if sol is None:
                    # Keep the exploratory position, but never attach a fake
                    # fitness or update pbest/gbest from an unevaluated point.
                    velocity[i] = self.rng.uniform(-c.velocity_limit, c.velocity_limit, x[i].shape)
                    continue
                if sol.objective < pf[i] - 1e-9:
                    pf[i], pb[i] = sol.objective, x[i].copy()
                if sol.objective < best.objective - 1e-9:
                    best, best_x = sol.clone(), x[i].copy()
                    self._tabu_best = best.objective
                    self._emit_history(best, "improvement", it)
            if completed:
                self.iterations_completed += 1
            self.history.append(dict(iteration=it, elapsed_seconds=time.perf_counter()-started,
                                     global_best=best.objective))
            if not completed:
                break
        # Same physical evaluator as all other algorithms, not a proxy fitness.
        ev = evaluate_solution(best.vehicle_routes,best.drone_routes,self.problem,True)
        if not ev.feasible or not math.isclose(ev.objective,best.objective,rel_tol=1e-10,abs_tol=1e-7):
            raise RuntimeError('PSO final validation failed: '+ev.reason)
        best.evaluation = ev
        self.time_limit_hit = expired()
        self.runtime_seconds = time.perf_counter()-started
        self._emit_history(best, "final", self.iterations_used)
        return best
