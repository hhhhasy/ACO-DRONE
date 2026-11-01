import math
import random
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from utils import (
    nearest_neighbor_insertion, drone_clusters, dp_energy_fixed_lr,
    fast_copy_drone_routes, farthest_insertion, partition_to_node_ids,
)
import sweep_clustering
from operators import ALNSOperators
from drone_physics import (
    DRONE_BATTERY_CAPACITY_WH, DRONE_PAYLOAD_CAPACITY_KG, DRONE_SPEED_M_S,
    TRUCK_SPEED_M_S, best_sortie_energy_wh, energy_optimal_sortie,
    payload_within_capacity, sortie_payload_kg,
)
from evaluator import evaluate_solution

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class vrp_with_drones:
    """纯 ALNS 模型：单起点初始化 + 自适应算子选择 + SA 接受机制。

    单位：距离 m，时间 s，速度 m/s，质量 kg，功率 W，能量 Wh。
    坐标与需求由 ``physicalize_datasets.py`` 一次性物理化，运行时代码不再做
    任何 ×100 或 demand normalization。
    无人机可行性由 payload(<=3 kg) + 真实飞行能耗 + 悬停能耗 + 剩余电量(200 Wh)
    决定；**不存在最大飞行距离约束**。
    """

    def __init__(self,
                     num_customers=None,
                     num_vehicles=None,
                     num_drones_per_vehicle=None,
                     vehicle_speed=None,
                     drone_speed=None,
                     drone_payload_capacity=None,
                     drone_battery_capacity_wh=None,
                     drone_max_distance=None,   # deprecated, ignored
                     depot_coords=None,
                     customer_coords=None,
                     customer_demands=None,
                     seed=42,
                     score_weights=None):
            self.num_customers = int(num_customers)
            self.num_vehicles = int(num_vehicles)
            self.num_drones_per_vehicle = int(num_drones_per_vehicle)
            # Fixed physical speeds (m/s).  JSON values are only a fallback.
            self.vehicle_speed = float(vehicle_speed) if vehicle_speed else TRUCK_SPEED_M_S
            self.drone_speed = float(drone_speed) if drone_speed else DRONE_SPEED_M_S
            self.drone_payload_capacity = (
                float(drone_payload_capacity)
                if drone_payload_capacity else DRONE_PAYLOAD_CAPACITY_KG
            )
            self.drone_battery_capacity_wh = (
                float(drone_battery_capacity_wh)
                if drone_battery_capacity_wh is not None
                else DRONE_BATTERY_CAPACITY_WH
            )
            # Deprecated: the maximum drone flight distance is no longer part of
            # the problem.  The attribute is kept only so that old callers do not
            # crash; nothing in the search or the evaluator reads it.
            self.drone_max_distance = None
            if drone_max_distance is not None:
                self.drone_max_distance_legacy = float(drone_max_distance)
            if self.num_customers < 0 or self.num_vehicles <= 0 or self.num_drones_per_vehicle < 0:
                raise ValueError("Invalid customer/vehicle/drone counts")
            if self.vehicle_speed <= 0 or self.drone_speed <= 0:
                raise ValueError("vehicle_speed and drone_speed must be positive")
            if self.drone_payload_capacity < 0:
                raise ValueError("drone payload capacity must be non-negative")
            if self.drone_battery_capacity_wh <= 0:
                raise ValueError("drone battery capacity must be positive")

            self.depot_coords = np.asarray(depot_coords, dtype=float).reshape(1, 2)
            self.customer_coords = np.asarray(customer_coords, dtype=float).reshape(-1, 2)
            if len(self.customer_coords) != self.num_customers:
                raise ValueError("num_customers does not match customer_coords")
            if len(customer_demands) != self.num_customers:
                raise ValueError("num_customers does not match customer_demands")
            self.customer_demands = [0.0] + [float(x) for x in customer_demands]
            self.seed = int(seed)
            random.seed(self.seed)

            self.all_coords = np.vstack([self.depot_coords, self.customer_coords])
            # Vectorized pairwise matrices replace the O(n^2) Python double loop.
            delta = self.all_coords[:, None, :] - self.all_coords[None, :, :]
            self.veh_dist_matrix = np.abs(delta).sum(axis=2)
            self.dro_dist_matrix = np.sqrt(np.square(delta).sum(axis=2))
            self._drone_distance_normalizer = float(self.dro_dist_matrix.max()) if self.dro_dist_matrix.size else 0.0
            self._max_customer_demand = max(self.customer_demands) if self.customer_demands else 0.0

            self.alns_operators = ALNSOperators(self)
            self.operator_pairs = self._build_operator_pairs()

            # Keep current experiment defaults.  Formal runners can still override
            # score_interval (e.g. 50) explicitly for reproducibility.
            self.score_interval = 30
            self.Tmin = 0.8
            self.gamma = 0.98
            self.reaction_param = 0.5
            self.operator_rho = [1.0] * len(self.operator_pairs)
            self.operator_score_accum = [0.0] * len(self.operator_pairs)
            self.operator_uses = [0] * len(self.operator_pairs)
            self.score_weights = score_weights or {'w1': 20, 'w2': 7, 'w3': 3, 'w4': 0}

            # Diagnostics of the most recent generate_initial_solution() call:
            # the sweep log lines and the winning rotation.  Both stay None when
            # the legacy KMeans partition ran.
            self.last_sweep_result = None
            self.last_initial_solution_log = None

    def _build_operator_pairs(self):
            return [
                (self.destroy_operator_1, self.repair_operator_1, "d1-r1: 随机移除并随机插入单个服务点"),
                (self.destroy_operator_2, self.repair_operator_2, "d2-r2: 随机交换两个点"),
                # (self.destroy_operator_3, self.repair_operator_3, "d3-r3: 路径翻转"),
                (self.destroy_operator_4, self.repair_operator_4, "d4-r4: 重新安排无人机起降点"),
                (self.destroy_operator_5, self.repair_operator_5, "d5-r5: 随机生成无人机任务"),
                (self.destroy_operator_6, self.repair_operator_6, "d6-r6: shaw移除"),
                # (self.destroy_operator_7, self.repair_operator_7, "d7-r7: slack感知的无人机起降窗口调整"),
            ]

    @staticmethod
    def _resolve_route_builder(initial_strategy):
            """Map a strategy name onto the existing truck route constructor."""
            if initial_strategy == "farthest":
                return farthest_insertion
            if initial_strategy == "nearest":
                return nearest_neighbor_insertion
            raise ValueError(
                f"Unsupported initial_strategy={initial_strategy!r}; use 'farthest' or 'nearest'."
            )

    def generate_initial_solution(self, initial_strategy="farthest", verbose=False,
                                  clustering_method=None, rotation=None):
            """Build a truck reference solution and derive energy-feasible sorties.

            The truck customer partition comes from the balanced angular sweep
            (``sweep_clustering``): each truck owns one continuous wedge around
            the depot, which keeps the initial truck routes from crossing each
            other all over the map.  Every sweep rotation is pushed through the
            *whole* pipeline -- truck route constructor, drone sortie
            initialisation, schedule / hover / SOC evaluation -- and the rotation
            whose fully evaluated solution has the smallest makespan wins.  The
            geometry is never the selection criterion.

            ``clustering_method='kmeans'`` restores the legacy KMeans partition
            (single start) for comparison runs; ``rotation`` forces one specific
            sweep start instead of the multi-start search.

            A sortie is only kept when it satisfies the payload capacity *and*
            the real flight energy of the owning drone's cumulative battery.
            After the sorties are built the whole solution is re-evaluated
            (schedule + hover + SOC); if any drone has run out of battery the
            most expensive sortie of that drone is dissolved and its customers
            are handed back to the truck, until the solution is feasible.
            """
            route_builder = self._resolve_route_builder(initial_strategy)
            method = sweep_clustering.get_initial_clustering_method(clustering_method)

            if method == 'kmeans':
                # Legacy single-start path: the route builder applies KMeans.
                self.last_sweep_result = None
                self.last_initial_solution_log = None
                veh_routes, final_drone_routes, total_time = (
                    self._build_initial_solution_from_partition(None, route_builder)
                )
            elif method == 'sweep':
                rotations = (
                    [float(rotation) % sweep_clustering.TWO_PI] if rotation is not None else None
                )
                self.last_sweep_result = sweep_clustering.build_best_sweep_initial_solution(
                    customer_coords=self.customer_coords[:self.num_customers],
                    depot=self.depot_coords[0],
                    num_trucks=self.num_vehicles,
                    drone_payload_capacity=self.drone_payload_capacity,
                    customer_demands=self.customer_demands[1:],
                    build_solution=lambda partition: self._build_initial_solution_from_partition(
                        partition, route_builder),
                    rotations=rotations,
                )
                self.last_initial_solution_log = self.last_sweep_result.log_lines
                veh_routes, final_drone_routes, total_time = self.last_sweep_result
            else:
                raise ValueError("unknown initial clustering method %r" % (method,))

            if verbose:
                print(veh_routes, final_drone_routes)
                print("Total time cost (makespan):", total_time)
                for line in (self.last_initial_solution_log or []):
                    print(line)
            return veh_routes, final_drone_routes, total_time

    def _build_initial_solution_from_partition(self, partition, route_builder):
            """One complete initial solution from a truck customer partition.

            ``partition`` holds 0-based indices into the customer array (as the
            sweep module produces them) or ``None`` to let the route builder
            apply the configured clustering method itself.  The visiting order
            inside each cluster and the whole drone side come from the existing
            constructors -- the partition only decides which truck's delivery
            system owns each customer.
            """
            node_partition = None if partition is None else partition_to_node_ids(partition)

            veh_routes, _ = route_builder(
                customer_coords=self.customer_coords[:self.num_customers],
                num_vehicles=self.num_vehicles,
                depot_coord=self.depot_coords[0],
                dist_matrix=self.veh_dist_matrix,
                seed=self.seed,
                partition=node_partition,
                customer_demands=self.customer_demands[1:],
                drone_payload_capacity=self.drone_payload_capacity,
            )

            if self.num_customers <= 0:
                return veh_routes, [[] for _ in veh_routes], 0.0

            veh_routes, final_drone_routes = self._construct_drone_sorties(veh_routes)
            total_time = self.calculate_solution_time_cost(veh_routes, final_drone_routes)
            return veh_routes, final_drone_routes, total_time

    def _construct_drone_sorties(self, veh_routes):
            """Derive validated, energy-feasible sorties for an existing truck plan."""
            veh_routes = [list(route) for route in veh_routes]
            if self.num_customers <= 0:
                return veh_routes, [[] for _ in veh_routes]

            avg_demand = sum(self.customer_demands[1:]) / self.num_customers
            avg_drone_service = (
                max(1, int(self.drone_payload_capacity / avg_demand))
                if avg_demand > 1e-12 else max(1, self.num_customers)
            )

            final_drone_routes = []
            for vehicle_idx, original_route in enumerate(veh_routes):
                route = list(original_route)
                customers = [node for node in route if node != 0]
                if customers and self.num_drones_per_vehicle > 0:
                    coords = self.customer_coords[np.asarray(customers, dtype=int) - 1]
                    clusters = drone_clusters(
                        route, self.num_drones_per_vehicle, coords, avg_drone_service, self.seed
                    )
                else:
                    clusters = []

                # Important: sorties are vehicle-local.  The previous implementation
                # kept this list outside the vehicle loop and could leak earlier-vehicle
                # sorties into later vehicles during validation.
                drone_sorties = []
                blocked_nodes = set()
                drone_served = set()
                # Battery already committed by each physical drone of this truck.
                drone_energy_used = defaultdict(float)

                for cluster_original in clusters:
                    cluster = list(cluster_original)
                    if not cluster:
                        continue
                    launch_node = int(cluster[0])
                    for drone in range(self.num_drones_per_vehicle):
                        available = [c for c in cluster if c not in blocked_nodes]
                        if not available:
                            break
                        drone_id = f"{vehicle_idx + 1}-D{drone + 1}"
                        remaining_wh = (
                            self.drone_battery_capacity_wh - drone_energy_used[drone_id]
                        )
                        drone_task, selected_nodes = self.create_drone_task(
                            launch_node, route, available,
                            remaining_energy_wh=remaining_wh,
                        )
                        if not drone_task or not drone_task[1]:
                            continue
                        blocked_nodes.update(int(n) for n in selected_nodes)
                        # Energy-aware delivery order (minimises flight energy).
                        l, d, r = dp_energy_fixed_lr(
                            l_idx=int(drone_task[0]),
                            r_idx=int(drone_task[2]),
                            d_indices=[int(x) for x in drone_task[1]],
                            dro_dist_matrix=self.dro_dist_matrix,
                            demands=self.customer_demands,
                        )
                        drone_energy_used[drone_id] += best_sortie_energy_wh(
                            l, r, d, self.dro_dist_matrix, self.customer_demands
                        )
                        drone_served.update(d)
                        delivery_set = set(drone_served)
                        route = [node for node in route if node not in delivery_set]
                        drone_sorties.append([drone_id, l, d, r])

                provisional_route = [node for node in original_route if node not in drone_served]
                validated_sorties = self._validate_drone_sorties(drone_sorties, provisional_route)
                valid_served = {int(d) for s in validated_sorties for d in s[2]}
                veh_routes[vehicle_idx] = [node for node in original_route if int(node) not in valid_served]
                final_drone_routes.append(validated_sorties)

            # Full schedule + hover + SOC feasibility pass; dissolve sorties whose
            # drone cannot afford them (real energy, not distance).
            return self._repair_energy_feasibility(veh_routes, final_drone_routes)

    def evaluate_solution(self, vehicle_routes, drone_routes, check_coverage=True):
            """Full evaluation (makespan + hover + per-drone SOC) -- see evaluator."""
            return evaluate_solution(
                vehicle_routes, drone_routes, self, check_coverage=check_coverage
            )

    def _repair_energy_feasibility(self, veh_routes, drone_routes, max_rounds=200):
            """Dissolve sorties until the solution is physically feasible.

            Only the *battery* / payload dimension is repaired here: a sortie is
            removed (and its customers handed back to the truck route) whenever
            its drone's cumulative consumption would exceed 200 Wh.  Returns the
            best (possibly still infeasible) attempt after ``max_rounds``.
            """
            veh_routes = [list(route) for route in veh_routes]
            drone_routes = fast_copy_drone_routes(drone_routes)

            for _ in range(max_rounds):
                evaluation = self.evaluate_solution(veh_routes, drone_routes)
                if evaluation.feasible:
                    return veh_routes, drone_routes
                if 'battery' not in evaluation.reason and 'payload' not in evaluation.reason:
                    # Structural problems are not repairable here.
                    return veh_routes, drone_routes

                # Drop the most energy-expensive sortie of the worst drone.
                worst = None
                for report in evaluation.sorties:
                    if worst is None or report.total_energy_wh > worst.total_energy_wh:
                        worst = report
                if worst is None:
                    return veh_routes, drone_routes

                v_idx = worst.vehicle_idx
                sorties = drone_routes[v_idx]
                keep = []
                for sortie in sorties:
                    if (int(sortie[1]) == worst.launch
                            and int(sortie[3]) == worst.rendezvous
                            and [int(c) for c in sortie[2]] == list(worst.order)):
                        continue
                    keep.append(sortie)
                if len(keep) == len(sorties):
                    return veh_routes, drone_routes
                drone_routes[v_idx] = keep

                # Hand the customers back to the truck route.
                for customer in worst.order:
                    self._insert_customer_into_route(veh_routes[v_idx], int(customer))

            return veh_routes, drone_routes

    def _insert_customer_into_route(self, route, customer):
            """Cheapest feasible insertion of one customer into a truck route."""
            if customer in route:
                return route
            best_pos = 1
            best_cost = float('inf')
            for pos in range(1, len(route)):
                prev_node = int(route[pos - 1])
                next_node = int(route[pos])
                cost = (
                    float(self.veh_dist_matrix[prev_node, customer])
                    + float(self.veh_dist_matrix[customer, next_node])
                    - float(self.veh_dist_matrix[prev_node, next_node])
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            route.insert(best_pos, customer)
            return route

    def optimize_drone_task(self,drone_routes):
            # 为每个无人机任务执行 energy-aware 动态规划优化（最小化 flight energy，而非距离）
            optimized_routes = []
            for vehicle_drone_routes in drone_routes:
                optimized_route = []
                for sortie in vehicle_drone_routes:
                    drone_id, launch, delivery, rendezvous = sortie
                    l, d, r = dp_energy_fixed_lr(
                        l_idx=int(launch),
                        r_idx=int(rendezvous),
                        d_indices=[int(c) for c in delivery],
                        dro_dist_matrix=self.dro_dist_matrix,
                        demands=self.customer_demands,
                    )
                    optimized_route.append([drone_id, l, d, r])
                optimized_routes.append(optimized_route)
            return optimized_routes

    def _reset_adaptive_state(self):
            n = len(self.operator_pairs)
            self.operator_rho = [1.0] * n
            self.operator_score_accum = [0.0] * n
            self.operator_uses = [0] * n

    def _update_operator_weights(self, reaction_param):
            """按累计 uses 更新；本窗口未使用的算子不衰减。"""
            for j, uses in enumerate(self.operator_uses):
                if uses <= 0:
                    continue
                avg_score = self.operator_score_accum[j] / uses
                new_weight = (
                    (1.0 - reaction_param) * self.operator_rho[j]
                    + reaction_param * avg_score
                )
                self.operator_rho[j] = max(float(new_weight), 1e-6)

            self.operator_score_accum = [0.0] * len(self.operator_pairs)
            self.operator_uses = [0] * len(self.operator_pairs)

    def local_search(
            self,
            initial_routes,
            initial_drones,
            iterations=100,
            score_interval=None,
            gamma=None,
            Tmin=None,
            reaction_param=None,
            time_limit_seconds=None,
            reset_adaptive=True,
            reset_rng=True,
            verbose=False,
            return_info=False,
            target_time=None,
            history_callback=None,
        ):
            search_start = time.perf_counter()

            if reset_rng:
                random.seed(self.seed)
            if reset_adaptive:
                self._reset_adaptive_state()

            score_interval = self.score_interval if score_interval is None else max(1, int(score_interval))
            gamma = self.gamma if gamma is None else gamma
            Tmin = self.Tmin if Tmin is None else Tmin
            reaction_param = self.reaction_param if reaction_param is None else reaction_param

            current_routes = [list(route) for route in initial_routes]
            current_drones = fast_copy_drone_routes(initial_drones)
            current_time = self.calculate_solution_time_cost(current_routes, current_drones)
            evaluations = 1

            best_routes = [list(route) for route in current_routes]
            best_drones = fast_copy_drone_routes(current_drones)
            best_time = current_time

            time_to_best_seconds = 0.0
            best_iteration = 0
            time_to_target_seconds = (
                0.0 if target_time is not None and best_time <= target_time + 1e-9 else None
            )

            # The initial solution can be energy-infeasible (e.g. a drone without
            # enough battery).  The annealing schedule still needs a finite
            # reference temperature, so fall back to the truck-only travel time.
            reference_time = current_time
            if not math.isfinite(reference_time):
                reference_time = self._truck_only_reference(current_routes)
            self.T0 = -(max(reference_time, 1e-9) * 0.25) / math.log(0.5)
            T = self.T0
            max_attempts = 20


            def time_up():
                return (
                    time_limit_seconds is not None
                    and time.perf_counter() - search_start >= time_limit_seconds
                )

            def emit(event_type, best_value, iteration):
                if history_callback is None:
                    return
                history_callback({
                    "best_makespan_s": float(best_value),
                    "iteration_or_generation": int(iteration),
                    "evaluations": int(evaluations),
                    "event_type": str(event_type),
                })

            iterations_used = 0
            iteration_limit = None if iterations is None else int(iterations)
            if iteration_limit is not None and iteration_limit <= 0:
                iteration_limit = None
            if iteration_limit is None and (time_limit_seconds is None or time_limit_seconds <= 0):
                raise ValueError(
                    "time_limit_seconds must be positive when iterations is unlimited"
                )

            i = 0
            while iteration_limit is None or i < iteration_limit:
                if time_up():
                    break
                iterations_used = i + 1

                op_idx = self.choose_operator_index()
                destroy_op, repair_op, op_name = self.operator_pairs[op_idx]
                score = self.score_weights["w4"]

                for _ in range(max_attempts):
                    if time_up():
                        break

                    destroyed_routes, destroyed_drones, removed_nodes = destroy_op(
                        current_routes, current_drones
                    )
                    new_routes, new_drones = repair_op(
                        destroyed_routes, destroyed_drones, removed_nodes
                    )
                    evaluations += 1
                    new_time = self.calculate_solution_time_cost(new_routes, new_drones)

                    # Physical feasibility barrier: a candidate that violates the
                    # payload / flight-energy / hover / battery constraints has
                    # objective ``inf`` and is rejected outright -- the simulated
                    # annealing temperature can never make it acceptable.
                    accept, delta = self._acceptance_decision(new_time, current_time, T)
                    if delta is not None and delta >= 0:
                        # Cooling happens on every non-improving evaluation.
                        T = max(T * gamma, Tmin)
                    if not accept:
                        continue

                    # delta is never None here (infeasible candidates are rejected
                    # above): delta < 0 is an improving move, delta >= 0 was
                    # accepted by the annealing criterion.
                    score = (
                        self.score_weights["w2"] if delta < 0
                        else self.score_weights["w3"]
                    )

                    current_routes = [list(route) for route in new_routes]
                    current_drones = fast_copy_drone_routes(new_drones)
                    current_time = new_time

                    if new_time < best_time - 1e-12:
                        best_routes = [list(route) for route in new_routes]
                        best_drones = fast_copy_drone_routes(new_drones)
                        best_time = new_time
                        elapsed = time.perf_counter() - search_start
                        time_to_best_seconds = elapsed
                        best_iteration = i + 1
                        emit("improvement", best_time, i + 1)
                        score = self.score_weights["w1"]

                        if (
                            target_time is not None
                            and time_to_target_seconds is None
                            and best_time <= target_time + 1e-9
                        ):
                            time_to_target_seconds = elapsed

                        if verbose:
                            print(
                                f"[ALNS] iter={i + 1}, op={op_name}, "
                                f"best={best_time:.4f}, t_best={elapsed:.3f}s"
                            )
                    break

                self.operator_score_accum[op_idx] += score
                self.operator_uses[op_idx] += 1
                if sum(self.operator_uses) >= score_interval:
                    self._update_operator_weights(reaction_param)
                i += 1

            # Keep the wall-clock budget strict: deterministic post-processing is
            # only attempted when there is still time left in the static-search
            # budget.  This prevents a nominal 30 s search from doing extra work
            # after the deadline.
            if not time_up():
                optimized_drones = self.optimize_drone_task(best_drones)
                evaluations += 1
                optimized_time = self.calculate_solution_time_cost(best_routes, optimized_drones)
            else:
                optimized_drones = best_drones
                optimized_time = best_time
            if optimized_time < best_time - 1e-12:
                best_drones = optimized_drones
                best_time = optimized_time
                elapsed = time.perf_counter() - search_start
                time_to_best_seconds = elapsed
                best_iteration = iterations_used
                emit("improvement", best_time, iterations_used)
                if (
                    target_time is not None
                    and time_to_target_seconds is None
                    and best_time <= target_time + 1e-9
                ):
                    time_to_target_seconds = elapsed

            info = {
                "search_runtime_seconds": time.perf_counter() - search_start,
                "time_to_best_seconds": time_to_best_seconds,
                "time_to_target_seconds": time_to_target_seconds,
                "best_iteration": best_iteration,
                "iterations_used": iterations_used,
                "evaluations": evaluations,
            }
            if return_info:
                return best_routes, best_drones, best_time, info
            return best_routes, best_drones, best_time

    def _acceptance_decision(self, new_time, current_time, temperature):
            """Simulated-annealing acceptance rule with a physical-feasibility gate.

            Returns ``(accept, delta)``.  A candidate whose objective is not
            finite is *physically infeasible* (payload > 3 kg, sortie energy
            beyond the remaining battery, cumulative SOC < 0, malformed
            solution, ...) and is rejected outright -- no temperature, however
            large, can make it acceptable.  Simulated annealing only ever
            compares physically feasible candidates.
            """
            if new_time is None or not math.isfinite(new_time):
                return False, None
            if current_time is None or not math.isfinite(current_time):
                # Any feasible candidate beats an infeasible incumbent.
                return True, -float('inf')
            delta = new_time - current_time
            if delta < 0:
                return True, delta
            accept = random.random() < math.exp(-delta / max(temperature, 1e-12))
            return accept, delta

    def _truck_only_reference(self, vehicle_routes):
            """Finite surrogate time used only to set the initial temperature."""
            total = 0.0
            speed = float(self.vehicle_speed) if self.vehicle_speed > 0 else 1.0
            for route in vehicle_routes or []:
                try:
                    nodes = [int(x) for x in route]
                except (TypeError, ValueError):
                    continue
                for idx in range(1, len(nodes)):
                    total += float(self.veh_dist_matrix[nodes[idx - 1], nodes[idx]]) / speed
            return max(total, 1.0)

    def choose_operator_index(self):
            # 步骤1：预处理权重（轮盘赌的前提：权重必须是「非负、有限值」）
            valid_weights = []
            for w in self.operator_rho:
                # 过滤非有限值（inf/NaN），负权重转为0（无效权重视为0）
                if math.isfinite(w):
                    valid_weights.append(max(w, 0.0))
                else:
                    valid_weights.append(0.0)
        
            # 步骤2：计算有效权重总和（轮盘赌的「轮盘周长」）
            total_weight = sum(valid_weights)
            operator_count = len(self.operator_rho)
        
            # 退化情况：总权重无效或为0（所有权重都无效/为负），均匀随机选择
            if not math.isfinite(total_weight) or total_weight <= 1e-12:  # 加微小阈值避免浮点数误差
                return random.randrange(operator_count)
        
            # 步骤3：生成0~总权重的随机采样点（轮盘上的随机指针）
            pick = random.uniform(0.0, total_weight)
        
            # 步骤4：累加权重，找到采样点对应的算子索引（指针落在哪个「扇形区域」）
            cumulative_weight = 0.0
            for idx, weight in enumerate(valid_weights):
                cumulative_weight += weight
                # 当累加和超过采样点时，当前索引即为选中的算子
                if cumulative_weight >= pick - 1e-12:  # 微小阈值处理浮点数精度问题
                    return idx

    def create_drone_task(self, launch_node, veh_route, customer, remaining_energy_wh=None):
            """Create one energy-feasible sortie from a launch node and a customer set.

            Feasibility = payload <= 3 kg AND the real flight energy of the
            sortie (in its energy-optimal order) fits in the remaining battery of
            the drone.  There is **no** maximum flight distance test.
            """
            launch_node = int(launch_node)
            filtered = [int(c) for c in customer if int(c) not in (0, launch_node)]
            sorted_customers = sorted(filtered, key=lambda x: self.dro_dist_matrix[launch_node, x])

            deliveries = []
            payload = 0.0
            for c in sorted_customers:
                demand = self.customer_demands[c]
                if payload + demand > self.drone_payload_capacity + 1e-12:
                    break
                deliveries.append(c)
                payload += demand

            if remaining_energy_wh is None:
                remaining_energy_wh = self.drone_battery_capacity_wh

            try:
                launch_idx = list(veh_route).index(launch_node)
            except ValueError:
                return None, None

            for idx in range(launch_idx + 1, len(veh_route)):
                candidate = int(veh_route[idx])
                if candidate in deliveries:
                    continue
                while deliveries:
                    flight_energy = best_sortie_energy_wh(
                        launch_node, candidate, deliveries,
                        self.dro_dist_matrix, self.customer_demands,
                    )
                    if flight_energy <= remaining_energy_wh + 1e-9:
                        selected_nodes = [launch_node, candidate] + deliveries.copy()
                        return [launch_node, deliveries.copy(), candidate], selected_nodes
                    deliveries.pop()
            return None, None

    def _validate_drone_sorties(self, drone_sorties, final_route):
            """Filter sorties that violate endpoint order, sequencing or battery.

            The battery test is a real-energy test: for every physical drone the
            flight energies of its sorties accumulate, and a sortie that would
            push the cumulative consumption above 200 Wh is dissolved (its
            customers stay with the truck).
            """
            route = [int(x) for x in final_route]
            customer_pos = {node: idx for idx, node in enumerate(route) if node != 0}
            last_rendezvous_index_per_drone = {}
            energy_used_per_drone = defaultdict(float)
            validated = []

            for sortie in drone_sorties:
                if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                    continue
                drone_id, launch, delivery, rendezvous = sortie
                try:
                    launch = int(launch)
                    rendezvous = int(rendezvous)
                except (TypeError, ValueError):
                    continue
                deliveries = [int(c) for c in delivery]
                if not deliveries:
                    continue
                if not payload_within_capacity(deliveries, self.customer_demands,
                                               self.drone_payload_capacity):
                    continue
                launch_idx = 0 if launch == 0 else customer_pos.get(launch, -1)
                rendezvous_idx = len(route) - 1 if rendezvous == 0 else customer_pos.get(rendezvous, -1)
                if launch_idx < 0 or rendezvous_idx < 0 or launch_idx >= rendezvous_idx:
                    continue
                prev_end = last_rendezvous_index_per_drone.get(drone_id, -1)
                if launch_idx < prev_end:
                    continue
                sortie_energy = best_sortie_energy_wh(
                    launch, rendezvous, deliveries,
                    self.dro_dist_matrix, self.customer_demands,
                )
                if (energy_used_per_drone[drone_id] + sortie_energy
                        > self.drone_battery_capacity_wh + 1e-9):
                    continue
                # Store the sortie in its energy-optimal delivery order.
                _, ordered, _ = dp_energy_fixed_lr(
                    launch, rendezvous, deliveries, self.dro_dist_matrix,
                    self.customer_demands,
                )
                validated.append([drone_id, launch, list(ordered), rendezvous])
                energy_used_per_drone[drone_id] += sortie_energy
                last_rendezvous_index_per_drone[drone_id] = rendezvous_idx
            return validated

    def calculate_solution_time_cost(self, vehicle_routes, drone_routes):
            """Return the makespan of a complete feasible solution.

            The single source of truth is :func:`evaluator.evaluate_solution`,
            which handles truck-drone synchronization, hover energy, real flight
            energy and the cumulative battery/SOC of every physical drone.
            Any malformed, incomplete, duplicated, payload-infeasible or
            energy-infeasible candidate receives ``inf`` (so it can never be
            accepted by the simulated-annealing criterion).

            There is no maximum-drone-distance test: range is not a constraint
            any more.  Depot node 0 is role-aware (start for launch, final depot
            for rendezvous).
            """
            return evaluate_solution(vehicle_routes, drone_routes, self).objective

    def destroy_operator_1(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_1(vehicle_routes, drone_routes)

    def destroy_operator_2(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_2(vehicle_routes, drone_routes)

    def destroy_operator_3(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_3(vehicle_routes, drone_routes)

    def destroy_operator_4(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_4(vehicle_routes, drone_routes)

    def destroy_operator_5(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_5(vehicle_routes, drone_routes)

    def destroy_operator_6(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_6(vehicle_routes, drone_routes)

    def repair_operator_1(self, vehicle_routes, drone_routes, removed_nodes):
            return self.alns_operators.repair_operator_1(vehicle_routes, drone_routes, removed_nodes)

    def repair_operator_2(self, vehicle_routes, drone_routes, removed_info):
            return self.alns_operators.repair_operator_2(vehicle_routes, drone_routes, removed_info)

    def repair_operator_3(self, vehicle_routes, drone_routes, removed_info):
            return self.alns_operators.repair_operator_3(vehicle_routes, drone_routes, removed_info)

    def repair_operator_4(self, vehicle_routes, drone_routes, removed_info):
            return self.alns_operators.repair_operator_4(vehicle_routes, drone_routes, removed_info)

    def repair_operator_5(self, vehicle_routes, drone_routes, removed_info):
            return self.alns_operators.repair_operator_5(vehicle_routes, drone_routes, removed_info)

    def repair_operator_6(self, destroyed_vehicle_routes, destroyed_drone_routes, removed_nodes):
            return self.alns_operators.repair_operator_6(destroyed_vehicle_routes, destroyed_drone_routes, removed_nodes)

    def destroy_operator_7(self, vehicle_routes, drone_routes):
            return self.alns_operators.destroy_operator_7(vehicle_routes, drone_routes)

    def repair_operator_7(self, vehicle_routes, drone_routes, removed_nodes):
            return self.alns_operators.repair_operator_7(vehicle_routes, drone_routes, removed_nodes)

    def plot_customer_distribution(self, customers: list, labels: list = None, title: str = "Customer Distribution", figsize: tuple = (8, 6), save_path: str = None):
            """仅绘制客户节点的空间分布图, 可选择保存到文件"""
            fig = plt.figure(figsize=(15, 10))

            # 绘制仓库
            depot_x, depot_y = self.depot_coords[0]
            plt.scatter(depot_x, depot_y, c='red', s=250, marker='s', label='Depot (0)', zorder=5)
            # plt.text(depot_x + 1, depot_y + 1, "Depot", fontsize=12, fontweight='bold')

            # 绘制客户点
            customer_coords = np.array(customers)
            plt.scatter(
                customer_coords[:, 0],
                customer_coords[:, 1],
                c='#1f77b4',
                s=36,
                marker='o',
                label='客户',
                zorder=5
            )

        

            # # 图例与画布设置
            # legend_handles = [
            #     mpatches.Patch(color='red', label='仓库'),
            #     mpatches.Patch(color='#1f77b4', label='客户')
            # ]
            # plt.legend(handles=legend_handles, loc='upper left')
            # plt.title(title, fontsize=16)
            # plt.xlabel('X 坐标', fontsize=14)
            # plt.ylabel('Y 坐标', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('equal')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
        
            return fig

    def plot_sweep_clusters(self, partition, title="Angular Sweep 客户分组", save_path=None):
            """Show the angular sweep partition: one colour per truck sector.

            ``partition`` holds 0-based customer indices, one list per truck --
            exactly what ``sweep_clustering`` returns and what
            ``last_sweep_result.partition`` carries.  The depot is drawn once and
            each sector gets an auxiliary ray from the depot spanning its
            angular range, so a glance shows whether the sectors really are
            continuous wedges rather than interleaved (KMeans) blobs.  No cluster
            centroid is drawn -- the sweep has no centroid.

            ``partition=None`` (e.g. a KMeans run) is a no-op.
            """
            if not partition:
                return None

            fig = plt.figure(figsize=(15, 10))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            depot = self.depot_coords[0]

            plt.scatter(depot[0], depot[1], c='red', s=250, marker='s',
                        label='Depot (0)', zorder=6)

            # Rays reach past the farthest customer so the sector is visible.
            farthest = 0.0
            for cluster in partition:
                for index in cluster:
                    farthest = max(farthest, float(np.hypot(
                        self.customer_coords[index][0] - depot[0],
                        self.customer_coords[index][1] - depot[1])))
            ray_length = farthest * 1.15 if farthest > 0 else 1.0

            for truck_index, cluster in enumerate(partition):
                if not cluster:
                    continue
                color = colors[truck_index % len(colors)]
                coords = self.customer_coords[np.asarray(cluster, dtype=int)]
                plt.scatter(coords[:, 0], coords[:, 1], c=color, s=70,
                            marker='o', zorder=5,
                            label='卡车 %d 系统' % (truck_index + 1))

                angles = [math.atan2(float(self.customer_coords[i][1]) - depot[1],
                                     float(self.customer_coords[i][0]) - depot[0])
                          for i in cluster]
                # Draw the angular span as a filled wedge to make continuity obvious.
                span = (max(angles) - min(angles)) % (2.0 * math.pi)
                if span > 0.0:
                    wedge = mpatches.Wedge(
                        (depot[0], depot[1]), ray_length,
                        math.degrees(min(angles)), math.degrees(min(angles) + span),
                        alpha=0.10, color=color, zorder=1)
                    plt.gca().add_patch(wedge)
                for boundary in (min(angles), min(angles) + span):
                    plt.plot([depot[0], depot[0] + ray_length * math.cos(boundary)],
                             [depot[1], depot[1] + ray_length * math.sin(boundary)],
                             c=color, linestyle=':', linewidth=1.2, alpha=0.8, zorder=2)

            plt.title(title, fontsize=16)
            plt.xlabel('X 坐标', fontsize=14)
            plt.ylabel('Y 坐标', fontsize=14)
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('equal')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return fig

    def plot_solution(self, vehicle_routes, drone_routes, save_path=None):
            """绘制解决方案的可视化图，并可选择保存到文件"""
            fig = plt.figure(figsize=(15, 10))
            vehicle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
            drone_cmap = cm.get_cmap('tab20')
            drone_color_map = {}
            drone_counter = 0

            # ... (保持原有的绘图逻辑不变)

            # -------------------------- 1. 绘制仓库 --------------------------
            depot_x = self.depot_coords[0, 0]
            depot_y = self.depot_coords[0, 1]
            plt.scatter(depot_x, depot_y, c='red', s=250, marker='s', label='配送中心', zorder=5)
            # plt.text(depot_x + 1, depot_y + 1, "配送中心", fontsize=12, fontweight='bold')

            # -------------------------- 2. 绘制客户点 --------------------------
            customer_ys = []
            if self.num_customers is not None:
                for i in range(self.num_customers):
                    x, y = self.customer_coords[i]
                    customer_ys.append(y)
                    customer_id = i + 1
                    #plt.text(x + 0.5, y + 0.5, str(customer_id), fontsize=7, ha='left', va='bottom', color='black', fontweight='normal', zorder=10)

            # -------------------------- 3. 绘制车辆路径 --------------------------
            vehicle_patches = []
            vehicle_ys = []
            for vehicle_id, route in enumerate(vehicle_routes):
                if not route or len(route) < 2: 
                    continue
                color = vehicle_colors[vehicle_id % len(vehicle_colors)]
                x = [self.all_coords[int(node), 0] for node in route]
                y = [self.all_coords[int(node), 1] for node in route]
                vehicle_ys.extend(y)
                plt.plot(x, y, c=color, linewidth=2, alpha=0.8, marker='o', markersize=6, zorder=3)
                vehicle_patches.append(mpatches.Patch(color=color, label=f'卡车 {vehicle_id+1}'))

            # -------------------------- 4. 绘制无人机路径 --------------------------
            drone_patches = []
            used_drone_labels = set()
            drone_ys = []
            for vehicle_id, sorties in enumerate(drone_routes):
                if not sorties:
                    continue
                for sortie in sorties:
                    drone_id, launch, delivery, rendezvous = sortie
                    if drone_id not in drone_color_map:
                        color = drone_cmap(drone_counter / 20)
                        drone_color_map[drone_id] = color
                        drone_counter = (drone_counter + 1) % 20
                    drone_path = [int(launch)] + [int(d) for d in delivery] + [int(rendezvous)]
                    x = [self.all_coords[node, 0] for node in drone_path]
                    y = [self.all_coords[node, 1] for node in drone_path]
                    drone_ys.extend(y)
                    plt.plot(x, y, c=drone_color_map[drone_id], linestyle='--', linewidth=1.5, alpha=0.9, marker='d', markersize=4, zorder=4)
                    if drone_id not in used_drone_labels:
                        drone_patches.append(mpatches.Patch(color=drone_color_map[drone_id], linestyle='--', label=f'无人机 {drone_id}'))
                        used_drone_labels.add(drone_id)

            # -------------------------- 5. 调整纵轴范围 --------------------------
            all_ys = [depot_y] + customer_ys + vehicle_ys + drone_ys
            if all_ys:
                min_y, max_y = min(all_ys), max(all_ys)
                y_range = max_y - min_y
                if y_range == 0:
                    y_mid, new_half_range = min_y, 5
                else:
                    y_mid = (min_y + max_y) / 2
                    new_half_range = y_range / 6
                plt.ylim(y_mid - new_half_range, y_mid + new_half_range)

            # -------------------------- 6. 图例与画布 --------------------------
            # legend_handles = [mpatches.Patch(color='red', label='配送中心'), mpatches.Patch(color='black', label='客户')] + vehicle_patches + drone_patches
            # plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
            # plt.title('配送路线', fontsize=16)
            # plt.xlabel('X 坐标', fontsize=14)
            # plt.ylabel('Y 坐标', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('auto')
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

    def print_solution_details(self, vehicle_routes, drone_routes, file_path=None,
                               total_time=None, verbose=True):
            """按照指定格式打印解决方案详情，仅对起飞点和降落点计算索引，配送点保留原始编号

            ``verbose=False`` 时方案详情照常写入 ``file_path``，只是不往控制台
            输出任何一行（main.py 用它保持控制台只有两个时间成本）。
            """
            # 所有控制台输出都走 emit：verbose=False 时结果仍然完整落盘。
            emit = print if verbose else (lambda *args, **kwargs: None)
            emit("\n===== 配送方案详情 =====")
            output_lines = []
            # 1. 打印车辆路径详情
            output_lines.append("truck_route=")
            emit("truck_route=")
            truck_routes_list = []
            for route in vehicle_routes:
                cleaned = []
                for node in route:
                    if not cleaned or node != cleaned[-1]:
                        cleaned.append(int(node))
                truck_routes_list.append(cleaned)
            truck_routes_str = "[" + ",\n".join([str(route) for route in truck_routes_list]) + "]"
            output_lines.append(truck_routes_str)
            emit(truck_routes_str)
            # 2. 打印每架无人机的路径（全局编号）
            drone_task_map = defaultdict(list)  # key: drone_full_id, value: list of paths

            for vehicle_id, drone_sorties in enumerate(drone_routes, 1):
                for sortie in drone_sorties:
                    drone_full_id, launch, delivery, rendezvous = sortie
                    # 获取当前车辆的完整路径
                    vehicle_full_route = vehicle_routes[vehicle_id - 1]  # vehicle_id 从1开始
                    # 构建两个映射：
                    # 1. 节点第一次出现的索引（用于起飞点）
                    # 2. 节点最后一次出现的索引（用于降落点）
                    first_occurrence = {}
                    last_occurrence = {}
                    for idx, node in enumerate(vehicle_full_route):
                        if node not in first_occurrence:
                            first_occurrence[node] = idx
                        last_occurrence[node] = idx  # 最终保留最后一次出现的索引
                    
                    # 处理起飞点：使用第一次出现的索引
                    launch_node = int(launch)
                    launch_idx = first_occurrence.get(launch_node, launch_node)
                    # 处理配送点：不计算索引，直接使用原始客户编号（关键修改）
                    delivery_nodes = [int(d) for d in delivery]  # 仅转换为整数，不映射为索引
                    # 处理降落点：使用最后一次出现的索引
                    rendezvous_node = int(rendezvous)
                    rendezvous_idx = last_occurrence.get(rendezvous_node, rendezvous_node)
                
                    # 路径组成：[起飞点索引] + [配送点原始编号] + [降落点索引]
                    path = [launch_idx] + delivery_nodes + [rendezvous_idx]
                    drone_task_map[drone_full_id].append(path)
        
            # 按无人机ID排序输出
            # sorted_drones = sorted(drone_task_map.items(), key=lambda x: x[0])
            # print(sorted_drones)
            for idx, (drone_id, paths) in enumerate(drone_task_map.items(), start=1):
                drone_title = f"drone_routes{idx}="
                output_lines.append(drone_title)
                emit(drone_title)
                paths_str = str(paths)
                output_lines.append(paths_str)
                emit(paths_str)

            # 3. 打印总时间成本
            total_time = self.calculate_solution_time_cost(vehicle_routes, drone_routes)
            time_str = f"\n总时间成本: {total_time:.2f}"
            emit(time_str)

            energy_lines = self.build_energy_report(vehicle_routes, drone_routes)
            for line in energy_lines:
                output_lines.append(line)
                emit(line)

            if file_path:
                if total_time is not None:
                    output_lines.append(f"\n总时长: {total_time:.2f} ")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))
                emit(f"\n配送方案已保存至: {file_path}")

    def build_energy_report(self, vehicle_routes, drone_routes):
            """Human-readable per-sortie / per-drone energy and SOC report."""
            evaluation = self.evaluate_solution(vehicle_routes, drone_routes)
            lines = ["", "===== 无人机能耗与电池 (Wh, 单位: m/s/kg/Wh) ====="]
            if not evaluation.feasible:
                lines.append(f"solution_infeasible={evaluation.reason}")
                return lines

            lines.append(f"battery_capacity_wh={self.drone_battery_capacity_wh:.1f}")
            lines.append(f"payload_capacity_kg={self.drone_payload_capacity:.2f}")
            lines.append(f"total_flight_energy_wh={evaluation.total_flight_energy_wh:.4f}")
            lines.append(f"total_hover_energy_wh={evaluation.total_hover_energy_wh:.4f}")
            for report in evaluation.sorties:
                lines.append(
                    "sortie vehicle={} drone={} launch={} rendezvous={} order={} "
                    "start_payload_kg={:.4f} flight_time_s={:.2f} "
                    "flight_energy_wh={:.4f} hover_time_s={:.2f} hover_energy_wh={:.4f} "
                    "total_energy_wh={:.4f}".format(
                        report.vehicle_idx + 1, report.drone_label, report.launch,
                        report.rendezvous, report.order, report.start_payload_kg,
                        report.flight_time_s, report.flight_energy_wh,
                        report.hover_time_s, report.hover_energy_wh,
                        report.total_energy_wh,
                    )
                )
            for (vehicle_idx, label), state in sorted(evaluation.drones.items()):
                lines.append(
                    "drone vehicle={} label={} sorties={} used_wh={:.4f} "
                    "remaining_energy_wh={:.4f}".format(
                        vehicle_idx + 1, label, len(state.sortie_keys),
                        state.energy_wh, state.remaining_energy_wh,
                    )
                )
            lines.append(f"makespan_s={evaluation.makespan:.4f}")
            return lines