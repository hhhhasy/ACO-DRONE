# ALNS drop-in low-risk acceleration.
# Based on the supplied operators(20261017-014800).py.
# Changes: bounded exact-input memoization of deterministic sortie physics;
# once-per-instance Manhattan-distance maximum.
# All move loops, candidate choices, checks, acceptance, and scoring are untouched.
# Validate with alns_fixed_iter_verify.py before using in benchmarks.
import random
from copy import deepcopy
from functools import lru_cache
import numpy as np
import heapq
from utils import (
    insert_two_nodes_safely, global_check, find_occurrence_between,
    get_all_customers_with_details, fast_copy_drone_routes,
)
from drone_physics import (
    best_sortie_energy_wh,
    energy_optimal_sortie,
    energy_per_meter_wh,
    evaluate_sortie_flight,
    sortie_energy_wh,
    sortie_payload_kg,
)

EPS = 1e-9

class ALNSOperators:
    """Destroy / repair operators.

    Every operator that can create or modify a drone sortie evaluates the
    *real* energy of the resulting sortie (payload-aware power on every leg)
    and the cumulative energy of the physical drone that would fly it.  A
    candidate is discarded when

      * the launch payload exceeds 3 kg, or
      * the drone's cumulative flight energy would exceed its 200 Wh battery.

    There is no maximum-flight-distance test anywhere: range has been removed
    from the problem.
    """

    def __init__(self, model):
        self.model = model
        self.veh_path = None
        self.drone_path = None
        self._drone_labels = [
            f"{v + 1}-D{d + 1}"
            for v in range(model.num_vehicles)
            for d in range(model.num_drones_per_vehicle)
        ]
        # Per-instance, bounded memoization; exact ordered customer tuple is the
        # key, so no distinct candidate lists are merged. All geometry, demands,
        # and physical constants are immutable for the duration of a solve.
        self._energy_memo = lru_cache(maxsize=50000)(self._uncached_best_energy)
        self._order_memo = lru_cache(maxsize=25000)(self._uncached_best_order)
        # Constant across all Shaw destruction calls for this immutable instance.
        veh_max = float(np.max(model.veh_dist_matrix)) if model.veh_dist_matrix.size else 1.0
        self._cached_max_veh_dist = max(1e-12, veh_max)

    def _uncached_best_energy(self, launch, rendezvous, deliveries):
        return best_sortie_energy_wh(
            launch, rendezvous, list(deliveries),
            self.model.dro_dist_matrix, self.model.customer_demands,
        )

    def _uncached_best_order(self, launch, rendezvous, deliveries):
        # Store immutable output, return a fresh list to each caller.
        result = energy_optimal_sortie(
            launch, rendezvous, list(deliveries),
            self.model.dro_dist_matrix, self.model.customer_demands,
        )['order']
        return tuple(result)

    def memo_stats(self):
        """Inspect performance only; not used by the search algorithm."""
        return {'energy': self._energy_memo.cache_info()._asdict(),
                'order': self._order_memo.cache_info()._asdict()}

    # ------------------------------------------------------------------
    # Shared energy helpers (used by every destroy / repair operator)
    # ------------------------------------------------------------------
    def _payload_within_capacity(self, deliveries):
        """载重约束: sum(demand) <= 3.0 kg (恰好 3 kg 可行)"""
        return sortie_payload_kg(deliveries, self.model.customer_demands) \
            <= self.model.drone_payload_capacity + EPS

    def _payload_kg(self, deliveries):
        return sortie_payload_kg(deliveries, self.model.customer_demands)

    def _sortie_energy_wh(self, launch_node, deliveries, rendezvous_node, order=None):
        """真实飞行能耗 (Wh) -- 逐段 payload 下降, 不是距离。"""
        if not deliveries:
            return 0.0
        return sortie_energy_wh(
            int(launch_node), [int(c) for c in deliveries], int(rendezvous_node),
            self.model.dro_dist_matrix, self.model.customer_demands,
            order=order, speed_m_s=self.model.drone_speed,
        )

    def _best_sortie_energy_wh(self, launch_node, deliveries, rendezvous_node):
        """Exact original physics, memoized by the *ordered* request tuple."""
        if not deliveries:
            return 0.0
        return self._energy_memo(
            int(launch_node), int(rendezvous_node),
            tuple(int(c) for c in deliveries),
        )

    def _best_sortie_order(self, launch_node, deliveries, rendezvous_node):
        """Exact optimal order; return a fresh list to avoid cache mutation."""
        return list(self._order_memo(
            int(launch_node), int(rendezvous_node),
            tuple(int(c) for c in deliveries),
        ))

    def _drone_energy_used_wh(self, drone_routes, vehicle_idx, drone_label,
                              skip_sortie_idx=None):
        """Energy already committed by one physical drone (flight energy only).

        Hover energy is always >= 0, so ``used <= 200 Wh`` is a *necessary*
        condition; the exact battery/hover feasibility is confirmed by the full
        evaluator once the candidate solution is assembled.
        """
        used = 0.0
        for s_idx, sortie in enumerate(drone_routes[vehicle_idx] or []):
            if skip_sortie_idx is not None and s_idx == skip_sortie_idx:
                continue
            if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                continue
            if str(sortie[0]) != str(drone_label):
                continue
            deliveries = [int(c) for c in sortie[2]]
            if not deliveries:
                continue
            used += self._best_sortie_energy_wh(
                int(sortie[1]), deliveries, int(sortie[3])
            )
        return used

    def _drone_can_fly(self, drone_routes, vehicle_idx, drone_label, deliveries,
                       launch_node=None, rendezvous_node=None, skip_sortie_idx=None):
        """Feasibility of one candidate sortie for one physical drone.

        Step 1: payload <= 3 kg (reject immediately when exceeded).
        Step 2/3: energy of the candidate sortie in its energy-optimal order.
        Step 4 (partial): cumulative flight energy of the drone <= 200 Wh.
        The full schedule (hover + battery) is verified by ``global_check`` on
        the assembled candidate solution.
        """
        deliveries = [int(c) for c in deliveries]
        if not deliveries:
            return False
        if not self._payload_within_capacity(deliveries):
            return False
        if launch_node is None or rendezvous_node is None:
            return False

        candidate_energy = self._best_sortie_energy_wh(
            launch_node, deliveries, rendezvous_node
        )
        if candidate_energy > self.model.drone_battery_capacity_wh + EPS:
            return False
        if vehicle_idx is None or drone_label is None:
            return True
        if not (0 <= int(vehicle_idx) < len(drone_routes)):
            return False
        used = self._drone_energy_used_wh(
            drone_routes, int(vehicle_idx), drone_label,
            skip_sortie_idx=skip_sortie_idx,
        )
        return used + candidate_energy <= self.model.drone_battery_capacity_wh + EPS

    def _energy_delta_wh(self, launch_node, rendezvous_node, old_deliveries,
                         new_deliveries):
        """``new_sortie_energy_wh - old_sortie_energy_wh`` (both energy-optimal)."""
        return (
            self._best_sortie_energy_wh(launch_node, new_deliveries, rendezvous_node)
            - self._best_sortie_energy_wh(launch_node, old_deliveries, rendezvous_node)
        )

    def _energy_equivalent_distance(self, energy_wh, payload_kg):
        """Express a drone insertion cost on the same scale as a distance.

        The conversion uses the payload-aware energy per metre, so ranking drone
        options by this value is *identical* to ranking them by real energy
        consumption: no extra objective weight is introduced, it is a pure unit
        bridge that keeps the regret comparison between a truck insertion
        (metres) and a drone insertion (Wh) meaningful.
        """
        per_meter = energy_per_meter_wh(payload_kg, self.model.drone_speed)
        if per_meter <= 0.0:
            return float('inf')
        return float(energy_wh) / per_meter

    def _compute_drone_path_distance(self, launch_node, deliveries, rendezvous_node):
        """Feasibility shim kept for external callers.

        It no longer computes a path distance at all: it returns the *energy*
        feasibility of the sortie (payload <= 3 kg and battery headroom).  There
        is deliberately no ``distance <= drone_max_distance`` comparison left in
        the code base.
        """
        return self._drone_can_fly(
            [], None, None, deliveries,
            launch_node=launch_node, rendezvous_node=rendezvous_node,
        )

    def save_path(self, vehicle_routes, drone_routes):
        self.veh_path=[list(route) for route in vehicle_routes]
        self.drone_path=fast_copy_drone_routes(drone_routes)

    def get_path(self):
        return self.veh_path,self.drone_path

    # ------------------------------------------------------------------
    # Solution-level safety net
    # ------------------------------------------------------------------
    def _served_customers(self, vehicle_routes, drone_routes):
        served = set()
        for route in vehicle_routes:
            served.update(int(n) for n in route if int(n) != 0)
        for group in drone_routes:
            for sortie in group:
                if isinstance(sortie, (list, tuple)) and len(sortie) == 4:
                    served.update(int(c) for c in sortie[2])
        return served

    def _forced_feasible_solution(self, vehicle_routes, drone_routes, missing_nodes):
        """Last-resort repair: every unassigned customer goes back on a truck.

        Truck capacity is never binding in this benchmark, so this always
        restores a complete solution.  Battery-infeasible sorties are then
        dissolved by the shared model-side repair, and the result is verified by
        ``global_check`` (payload + flight energy + hover + SOC).
        """
        veh = [list(route) for route in vehicle_routes]
        dro = fast_copy_drone_routes(drone_routes)
        for route in veh:
            if len(route) < 2:
                route[:] = [0, 0]

        served = self._served_customers(veh, dro)
        for node in missing_nodes:
            node = int(node)
            if node <= 0 or node in served:
                continue
            best_v_idx = 0
            best_cost = float('inf')
            for v_idx, route in enumerate(veh):
                for pos in range(1, len(route)):
                    prev_node, next_node = int(route[pos - 1]), int(route[pos])
                    cost = (
                        float(self.model.veh_dist_matrix[prev_node, node])
                        + float(self.model.veh_dist_matrix[node, next_node])
                        - float(self.model.veh_dist_matrix[prev_node, next_node])
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_v_idx = v_idx
            self.model._insert_customer_into_route(veh[best_v_idx], node)
            served.add(node)

        return self.model._repair_energy_feasibility(veh, dro)

    def _normalize_drone_orders(self, drone_routes):
        """Put every sortie in its energy-optimal visiting order (in place).

        The visiting order determines the remaining payload of each leg and
        therefore the power and the energy of the sortie; every operator that
        touches a sortie re-optimises it so the search never compares orders
        that are needlessly expensive.
        """
        for group in drone_routes:
            for sortie in group:
                if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                    continue
                deliveries = [int(c) for c in sortie[2]]
                if len(deliveries) < 2:
                    continue
                sortie[2] = self._best_sortie_order(
                    int(sortie[1]), deliveries, int(sortie[3])
                )
        return drone_routes

    def _finalize(self, base_vehicle_routes, base_drone_routes, missing_nodes,
                  candidate_vehicle_routes, candidate_drone_routes):
        """Verify a candidate and never return a physically infeasible solution.

        1. energy-optimal delivery order for every sortie;
        2. dissolve sorties whose drone would exceed 200 Wh (cumulative SOC);
        3. full ``global_check`` (payload / energy / hover / SOC / coverage);
        4. otherwise fall back to a guaranteed-feasible truck-only repair.
        """
        self._normalize_drone_orders(candidate_drone_routes)
        veh, dro = self.model._repair_energy_feasibility(
            candidate_vehicle_routes, candidate_drone_routes
        )
        if global_check(veh, dro, self.model):
            return veh, dro
        return self._forced_feasible_solution(
            base_vehicle_routes, base_drone_routes, missing_nodes
        )

    
    



    def destroy_operator_1(self, vehicle_routes, drone_routes):
        """
        破坏算子1: 随机移除单个客户。
        """
        # 使用深拷贝以避免修改原始解
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)

        # 步骤1: 构建所有已服务客户的列表
        served_customers = []
        # 从车辆路径中收集客户
        for v_idx, route in enumerate(new_vehicle_routes):
            # 忽略仓库 (节点0)
            for node_idx, node in enumerate(route[1:-1]):
                # (客户ID, 来源类型, 车辆索引, 在路径中的索引)
                served_customers.append((node, 'vehicle', v_idx, node_idx + 1))
        
        # 从无人机路径中收集客户
        for v_idx, sorties in enumerate(new_drone_routes):
            for s_idx, sortie in enumerate(sorties):
                delivery_nodes = sortie[2]
                for delivery_idx, node in enumerate(delivery_nodes):
                    # (客户ID, 来源类型, 车辆索引, 任务索引, 在配送列表中的索引)
                    served_customers.append((node, 'drone', v_idx, s_idx, delivery_idx))

        # 步骤2: 随机选择一个客户
        if not served_customers:
            return new_vehicle_routes, new_drone_routes, []
        customer_to_remove_info = random.choice(served_customers)
        node_to_remove = customer_to_remove_info[0]
        source_type = customer_to_remove_info[1]
        removed_nodes = [node_to_remove]

        # 步骤3 & 4: 根据来源执行移除和特殊情况处理
        if source_type == 'vehicle':
            _, _, v_idx, node_idx_in_route = customer_to_remove_info
            
            # 从车辆路径中移除该节点
            # 注意：customer_to_remove_info[0] 和 new_vehicle_routes[v_idx][node_idx_in_route] 是同一个客户
            del new_vehicle_routes[v_idx][node_idx_in_route]
            
            # --- 特殊情况检查 ---
            # 检查被移除的节点是否是任何无人机任务的起降点
            affected_sorties_indices = []
            if v_idx < len(new_drone_routes):
                for s_idx, sortie in enumerate(new_drone_routes[v_idx]):
                    _, launch_node, _, rendezvous_node = sortie
                    if launch_node == node_to_remove or rendezvous_node == node_to_remove:
                        affected_sorties_indices.append(s_idx)
                        # 将这个无人机任务的所有客户加入待移除列表
                        removed_nodes.extend(sortie[2])
                        # print(f"  [Destroy-1] 移除车辆节点 {node_to_remove} 导致无人机任务 {s_idx} (L:{launch_node}, R:{rendezvous_node}) 解散。")

            # 从后往前删除，避免索引错乱
            if affected_sorties_indices:
                for s_idx in sorted(affected_sorties_indices, reverse=True):
                    del new_drone_routes[v_idx][s_idx]

        elif source_type == 'drone':
            _, _, v_idx, s_idx, delivery_idx = customer_to_remove_info
            
            # 从无人机配送列表中移除
            del new_drone_routes[v_idx][s_idx][2][delivery_idx]
            
            # 如果移除后无人机配送列表为空，则整个任务也应被删除
            if not new_drone_routes[v_idx][s_idx][2]:
                del new_drone_routes[v_idx][s_idx]
                # print(f"  [Destroy-1] 移除无人机节点 {node_to_remove} 导致其所在任务被清空并删除。")

        return new_vehicle_routes, new_drone_routes, list(set(removed_nodes))
    


    def repair_operator_1(self, vehicle_routes, drone_routes, removed_nodes):
        # 深拷贝路径数据，避免修改原始数据
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)

        # 随机打乱待插入节点顺序，避免顺序偏差
        removed_nodes = [int(n) for n in removed_nodes]
        random.shuffle(removed_nodes)
        remaining_nodes = []

        # 遍历所有需要插入的节点
        for node_to_insert in removed_nodes:
            # 存储所有可能的插入位置
            possible_moves = []

            # 评估插入到车辆路径的所有可能性
            for v_idx, route in enumerate(new_vehicle_routes):
                # 可以插入到仓库点(0)之后到路径末尾仓库点(0)之前的任何位置
                for insert_pos in range(1, len(route)):
                    # 记录可能的移动：(类型, 车辆索引, 插入位置)
                    possible_moves.append(('vehicle', v_idx, insert_pos))

            # 评估插入到现有无人机任务的所有可能性
            # 每个无人机候选插入都必须经过: 载重 -> 重新优化配送顺序 ->
            # 重算逐段能耗/时间 -> 该无人机累计 SOC 检查
            for v_idx, sorties in enumerate(new_drone_routes):
                for s_idx, sortie in enumerate(sorties):
                    # 提取任务的起飞点、配送点列表和降落点
                    launch_node, delivery_nodes, rendezvous_node = sortie[1], sortie[2], sortie[3]
                    delivery_nodes = [int(c) for c in delivery_nodes]
                    drone_label = sortie[0]

                    # 检查所有可能的插入位置
                    for insert_pos in range(len(delivery_nodes) + 1):
                        # 构造临时配送列表（顺序稍后会按能耗重新优化）
                        temp_delivery = delivery_nodes[:insert_pos] + [node_to_insert] + delivery_nodes[insert_pos:]
                        # 可行性检查（真实能耗 + 载重 + 电池 SOC）
                        if self._drone_can_fly(
                            new_drone_routes, v_idx, drone_label, temp_delivery,
                            launch_node=launch_node,
                            rendezvous_node=rendezvous_node,
                            skip_sortie_idx=s_idx,
                        ):
                            # 记录可能的移动：(类型, 车辆索引, 任务索引, 在配送列表中的插入位置)
                            possible_moves.append(('drone', v_idx, s_idx, insert_pos))

            # 从所有可行移动中随机选择一个执行
            if possible_moves:
                chosen_move = random.choice(possible_moves)
                move_type = chosen_move[0]

                if move_type == 'vehicle':
                    # 插入到车辆路径
                    _, v_idx, insert_pos = chosen_move
                    new_vehicle_routes[v_idx].insert(insert_pos, node_to_insert)

                elif move_type == 'drone':
                    # 插入到无人机配送列表，并按最小能耗重新排列配送顺序
                    _, v_idx, s_idx, insert_pos = chosen_move
                    sortie = new_drone_routes[v_idx][s_idx]
                    deliveries = [int(c) for c in sortie[2]]
                    deliveries.insert(insert_pos, node_to_insert)
                    sortie[2] = self._best_sortie_order(
                        sortie[1], deliveries, sortie[3]
                    )
            else:
                remaining_nodes.append(node_to_insert)

        # 放回完整解后重新计算 schedule + hover + 该无人机累计 SOC
        if not remaining_nodes:
            return self._finalize(
                vehicle_routes, drone_routes, removed_nodes,
                new_vehicle_routes, new_drone_routes,
            )
        return self._finalize(
            vehicle_routes, drone_routes, removed_nodes + remaining_nodes,
            new_vehicle_routes, new_drone_routes,
        )
    

    def destroy_operator_2(self, vehicle_routes, drone_routes):
        """Randomly remove two served customers for a swap move."""
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)
        customer_pool = []
        for v_idx, route in enumerate(new_vehicle_routes):
            for node_idx, node in enumerate(route[1:-1], start=1):
                if int(node) == 0:
                    continue
                customer_pool.append({'node': int(node), 'loc': ('vehicle', v_idx, node_idx)})
        for v_idx, sorties in enumerate(new_drone_routes):
            for s_idx, sortie in enumerate(sorties):
                for delivery_idx, node in enumerate(sortie[2]):
                    if int(node) == 0:
                        # Treat malformed candidates as a failed move; never terminate
                        # the entire experiment with sys.exit().
                        return new_vehicle_routes, new_drone_routes, []
                    customer_pool.append({'node': int(node), 'loc': ('drone', v_idx, s_idx, delivery_idx)})
        if len(customer_pool) < 2:
            return new_vehicle_routes, new_drone_routes, []

        nodes_to_swap_info = random.sample(customer_pool, 2)
        nodes_to_swap_info.sort(key=lambda x: x['loc'][1:], reverse=True)
        for info in nodes_to_swap_info:
            loc = info['loc']
            if loc[0] == 'vehicle':
                _, v_idx, node_idx = loc
                del new_vehicle_routes[v_idx][node_idx]
            else:
                _, v_idx, s_idx, delivery_idx = loc
                del new_drone_routes[v_idx][s_idx][2][delivery_idx]
        return new_vehicle_routes, new_drone_routes, nodes_to_swap_info
    

    def repair_operator_2(self, vehicle_routes, drone_routes, removed_info):
        """Swap two customers; retry with fresh pairs when the swap is infeasible."""
        if not removed_info or len(removed_info) != 2:
            return vehicle_routes, drone_routes

        destroyed_routes = [list(r) for r in vehicle_routes]
        destroyed_drones = fast_copy_drone_routes(drone_routes)
        current_info = removed_info

        for attempt in range(11):
            if not current_info or len(current_info) != 2:
                return vehicle_routes, drone_routes
            info1, info2 = current_info
            node1, loc1 = int(info1['node']), info1['loc']
            node2, loc2 = int(info2['node']), info2['loc']

            candidate_routes = [list(r) for r in destroyed_routes]
            candidate_drones = fast_copy_drone_routes(destroyed_drones)
            insert_two_nodes_safely(
                candidate_routes, candidate_drones,
                node2, loc1, node1, loc2,
            )
            # 被改动的 sortie 重新做 energy-aware 顺序优化（payload 逐段下降）
            self._normalize_drone_orders(candidate_drones)
            if global_check(candidate_routes, candidate_drones, self.model):
                return self._finalize(
                    vehicle_routes, drone_routes,
                    [node1, node2], candidate_routes, candidate_drones,
                )

            # Restore the exact pre-destroy solution for a safe fallback / retry.
            restored_routes = [list(r) for r in destroyed_routes]
            restored_drones = fast_copy_drone_routes(destroyed_drones)
            insert_two_nodes_safely(
                restored_routes, restored_drones,
                node1, loc1, node2, loc2,
            )
            if attempt >= 10:
                return restored_routes, restored_drones

            destroyed_routes, destroyed_drones, current_info = self.destroy_operator_2(
                restored_routes, restored_drones
            )

        return vehicle_routes, drone_routes

    def destroy_operator_3(self, vehicle_routes, drone_routes):
        """
        破坏算子3: 从一条车辆路径中选择并移除一个片段。
        """
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        # 步骤1: 筛选出足够长的路径
        # 路径至少需要4个节点 [0, C1, C2, 0] 才能翻转一个长度为2的片段
        eligible_routes_indices = [
            i for i, r in enumerate(new_vehicle_routes) if len(r) >= 4
        ]

        # 当前解不存在可执行路径翻转的车辆路径时，返回空操作，避免 random.choice([]) 报错
        if not eligible_routes_indices:
            return new_vehicle_routes, fast_copy_drone_routes(drone_routes), {}
        
        # 步骤2: 随机选择一条路径
        v_idx = random.choice(eligible_routes_indices)
        route = new_vehicle_routes[v_idx]

        # 步骤3: 随机选择片段的起止点
        # 索引范围是 [1, len(route)-2]，避开首尾的仓库0
        # 确保 i < j
        n_customers = len(route) - 2
        start_idx = random.randint(1, n_customers)
        end_idx = random.randint(start_idx, n_customers)
        
        # 保证片段长度至少为2
        if start_idx == end_idx:
            if start_idx > 1:
                start_idx -= 1
            elif end_idx < n_customers:
                end_idx += 1
            else: # 路径只有两个客户，无法翻转
                 return new_vehicle_routes, drone_routes, {}
        
        # 步骤4: 切掉片段
        segment = route[start_idx : end_idx + 1]
        new_vehicle_routes[v_idx] = route[:start_idx] + route[end_idx + 1:]
        
        removed_segment_info = {
            'v_idx': v_idx,
            'start_idx': start_idx,
            'segment': segment
        }

        return new_vehicle_routes, drone_routes, removed_segment_info 

    def repair_operator_3(self, vehicle_routes, drone_routes, removed_segment):
        """Reverse a truck-route segment while preserving sortie endpoint order."""
        if not removed_segment:
            return vehicle_routes, drone_routes
        final_routes = [list(route) for route in vehicle_routes]
        current_removed = removed_segment
        # Exact pre-destroy fallback for a failed reversal.
        fallback_routes = [list(route) for route in vehicle_routes]
        fv = removed_segment['v_idx']
        fs = removed_segment['start_idx']
        fallback_routes[fv] = (
            fallback_routes[fv][:fs]
            + list(removed_segment['segment'])
            + fallback_routes[fv][fs:]
        )

        for _ in range(11):
            v_idx = current_removed['v_idx']
            start_idx = current_removed['start_idx']
            segment = current_removed['segment']
            if not 0 <= v_idx < len(final_routes):
                return fallback_routes, drone_routes

            base_route = final_routes[v_idx]
            candidate_route = base_route[:start_idx] + list(reversed(segment)) + base_route[start_idx:]
            pos = {int(node): idx for idx, node in enumerate(candidate_route) if int(node) != 0}
            # Depot is role-sensitive: start for launch, end for rendezvous.
            feasible = True
            for sortie in (drone_routes[v_idx] if v_idx < len(drone_routes) else []):
                launch, rendez = int(sortie[1]), int(sortie[3])
                lp = 0 if launch == 0 else pos.get(launch)
                rp = len(candidate_route) - 1 if rendez == 0 else pos.get(rendez)
                if lp is None or rp is None or lp >= rp:
                    feasible = False
                    break
            if feasible:
                trial = [list(r) for r in final_routes]
                trial[v_idx] = candidate_route
                if global_check(trial, drone_routes, self.model):
                    return trial, drone_routes

            # Retry with a fresh random segment, matching the original operator's
            # retry semantics without repeated deep copies and unused maps.
            restored = [list(r) for r in final_routes]
            restored[v_idx] = base_route[:start_idx] + list(segment) + base_route[start_idx:]
            new_routes, _, current_removed = self.destroy_operator_3(restored, drone_routes)
            if not current_removed:
                return fallback_routes, drone_routes
            final_routes = [list(r) for r in new_routes]
        return fallback_routes, drone_routes
          
    

    def destroy_operator_4(self, vehicle_routes, drone_routes):
        """
        破坏算子3: 随机选择一个无人机任务，将起降点置空。
        """
        nonempty = [i for i, s in enumerate(drone_routes) if len(s) > 0]
        if not nonempty:
            return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes), []

        v_idx = random.choice(nonempty)
        s_idx = random.randrange(len(drone_routes[v_idx]))
        original_sortie = drone_routes[v_idx][s_idx]
        field = random.choice(['launch', 'rendez'])

        old_launch, old_rendez = original_sortie[1], original_sortie[3]
        old_value = old_launch if field == 'launch' else old_rendez

        removed_info = {
            'v_idx': v_idx,
            'sortie_idx': s_idx,
            'field': field,
            'old_value': old_value,
            'original_sortie': original_sortie
        }
        
        return vehicle_routes, drone_routes, [removed_info]
    
    def repair_operator_4(self, vehicle_routes, drone_routes, removed_info):
        # 深拷贝路径数据
        jump = True
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)

        # 将无人机任务映射到对应的车辆
        mapped = fast_copy_drone_routes(new_drone_routes)
        
        if not removed_info:
            return new_vehicle_routes, new_drone_routes


        rem = removed_info[0]
        v_idx = rem.get('v_idx')
        s_idx = rem.get('sortie_idx')
        field = rem.get('field')
        count=0

        while jump:

            # 验证索引和字段有效性
            if v_idx is None or s_idx is None or field not in ('launch', 'rendez'):
                return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)

            if not (0 <= v_idx < len(mapped)) or not (0 <= s_idx < len(mapped[v_idx])):
                return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)

            # 为该车构建 vehicle_route 与 node_positions（node -> list of indices）
            vehicle_route = new_vehicle_routes[v_idx]
            node_positions = {}
            for idx, node in enumerate(vehicle_route):
                node_positions.setdefault(int(node), []).append(idx)

            sorties = [list(s) for s in mapped[v_idx]]  # 可变视图

            # 构建无人机标签序列映射：label -> list of positions (indices in mapped[v_idx])
            drone_seq = {}
            labels = []
            for idx_pos, s in enumerate(mapped[v_idx]):
                label = None
                try:
                    if isinstance(s, (tuple, list)) and len(s) > 0:
                        label = str(s[0])
                except Exception:
                    label = None
                if label is None:
                    label = f"unknown_{idx_pos}"
                labels.append(label)
                drone_seq.setdefault(label, []).append(idx_pos)

            # 获取本次操作的任务的无人机标签和其在该无人机序列中的位置
            this_label = labels[s_idx]
            seq_positions = drone_seq.get(this_label, [])

            try:
                pos_in_seq = seq_positions.index(s_idx)
            except ValueError:
                # 找不到，退回
                return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)

            # 同一无人机的前一个/后一个任务索引（如果存在）
            prev_idx_same_drone = seq_positions[pos_in_seq - 1] if pos_in_seq - 1 >= 0 else None
            next_idx_same_drone = seq_positions[pos_in_seq + 1] if pos_in_seq + 1 < len(seq_positions) else None

            # 当前任务的起飞点/降落点（可能为 None）
            cur_launch, cur_rendez = sorties[s_idx][1], sorties[s_idx][3]

            # 从原始任务中获取起飞点和降落点信息
            orig_sortie = rem.get('original_sortie')
            if cur_launch is None and orig_sortie is not None:
                cur_launch = orig_sortie[1]
            if cur_rendez is None and orig_sortie is not None:
                cur_rendez = orig_sortie[3]

            # 临时配送列表（第三项）取自当前或原始任务
            temp_delivery = None
            try:
                tmp = sorties[s_idx]
                if len(tmp) >= 3 and isinstance(tmp[2], (list, tuple)):
                    temp_delivery = list(tmp[2])
            except Exception:
                temp_delivery = None
            if temp_delivery is None and orig_sortie is not None:
                try:
                    if len(orig_sortie) >= 3 and isinstance(orig_sortie[2], (list, tuple)):
                        temp_delivery = list(orig_sortie[2])
                except Exception:
                    temp_delivery = None
            if temp_delivery is None:
                temp_delivery = []

            
            # 计算上下界索引（exclusive）
            lower_idx = None
            upper_idx = None

            if field == 'launch':
                # 下界：前一个同无人机任务的降落点的某个出现位置（取最后一个出现位置）
                if prev_idx_same_drone is not None:
                    prev_r = mapped[v_idx][prev_idx_same_drone][3]
                    if prev_r is not None:
                        # 选择 prev_r 在 vehicle_route 中最后一个出现位置作为下界（靠后那个）
                        prev_pos = find_occurrence_between(node_positions,prev_r, min_exclusive=None, max_exclusive=None, prefer='last')
                        if prev_pos is None:
                            return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                        lower_idx = prev_pos
                # 上界：当前任务的降落点必须存在且要选取一个出现位置（尽量选择第一个在 lower_idx 之后的）
                if cur_rendez is None:
                    return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                # 寻找第一个大于 lower_idx 的降落点出现位置（若 lower_idx 为 None 则选第一个出现位置）
                if lower_idx is None:
                    rendez_pos = find_occurrence_between(node_positions,cur_rendez, min_exclusive=None, max_exclusive=None, prefer='first')
                else:
                    rendez_pos = find_occurrence_between(node_positions,cur_rendez, min_exclusive=lower_idx, max_exclusive=None, prefer='first')
                if rendez_pos is None:
                    return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                upper_idx = rendez_pos

            else:  # field == 'rendez'
                # 下界：当前任务的起飞点必须存在，选择一个靠后的起飞点出现位置（尽量靠后）
                if cur_launch is None:
                    return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                # 选择最后一个出现位置作为下界（保证降落点在其之后）
                launch_pos = find_occurrence_between(node_positions,cur_launch, min_exclusive=None, max_exclusive=None, prefer='last')
                if launch_pos is None:
                    return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                lower_idx = launch_pos
                # 上界：下一个同无人机任务的起飞点（若存在），选第一个出现位置且必须在 lower_idx 之后
                if next_idx_same_drone is not None:
                    next_l = mapped[v_idx][next_idx_same_drone][1]
                    if next_l is not None:
                        next_pos = find_occurrence_between(node_positions,next_l, min_exclusive=lower_idx, max_exclusive=None, prefer='first')
                        if next_pos is None:
                            return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)
                        upper_idx = next_pos

            # 筛选候选节点（满足位置顺序与飞行距离约束）
            candidates = []
            for idx, node in enumerate(vehicle_route):
                # 我们以具体位置 idx 为单位遍历，因此可以区分两个 0（起仓库、回仓库）
                if node == None:
                    continue

                # 排除掉不在 (lower_idx, upper_idx) 范围内的位置（注意 exclusive）
                if lower_idx is not None and not (idx > lower_idx):
                    continue
                if upper_idx is not None and not (idx < upper_idx):
                    continue

                # 位置上是否满足起飞点/降落点的内部顺序（起飞点必须在降落点之前）
                if field == 'launch':
                    # 候选节点的位置 idx 必须小于选定的降落点位置（upper_idx），但我们已把 upper_idx 设为具体降落点出现位置
                    rendez_idx = upper_idx
                    if rendez_idx is None or not (idx < rendez_idx):
                        continue
                else:
                    # field == 'rendez': 候选节点的位置 idx 必须大于选定的起飞点位置（lower_idx）
                    launch_idx = lower_idx
                    if launch_idx is None or not (idx > launch_idx):
                        continue

                # 飞行距离检查（构建临时无人机路径），注意使用节点值（node）和 temp_delivery 中的节点值
                if field == 'launch':
                    launch_node = node
                    rendez_node = cur_rendez
                else:
                    launch_node = cur_launch
                    rendez_node = node

                # 若起飞点或降落点是 None，跳过
                if launch_node is None or rendez_node is None:
                    continue

                # 端点改变 -> 配送顺序与逐段能耗都要重新计算：
                # 载重 <= 3 kg，且该无人机累计飞行能耗不超过 200 Wh。
                if not self._drone_can_fly(
                    mapped, v_idx, this_label, temp_delivery,
                    launch_node=launch_node,
                    rendezvous_node=rendez_node,
                    skip_sortie_idx=s_idx,
                ):
                    continue
                # 此处 candidates 存放节点的"值"和它在 vehicle_route 的具体位置索引（需区别两个0）
                candidates.append((node, idx))

            if not candidates:
                return [list(route) for route in vehicle_routes], fast_copy_drone_routes(drone_routes)

            # 随机从候选位置中选择一个（保留节点值）
            chosen_node, chosen_idx = random.choice(candidates)

            # 写回 mapped[v_idx][s_idx]（保持原来任务的结构，只修改起飞点或降落点的节点值）
            new_sortie = list(mapped[v_idx][s_idx])
            if field == 'launch':
                # 起飞点在任务的索引位置通常是 [label, launch, ...] ，我们只写节点值（不要写成索引）
                if len(new_sortie) < 2:
                    while len(new_sortie) < 4:
                        new_sortie.append(None)
                new_sortie[1] = int(chosen_node)
            else:
                # 降落点通常在第4位 (index=3) ，同上写节点值
                if len(new_sortie) < 4:
                    while len(new_sortie) < 4:
                        new_sortie.append(None)
                new_sortie[3] = int(chosen_node)

            # 端点变化后重新做 energy-aware 配送顺序优化（payload 逐段下降，
            # 不同顺序的能耗不同）。
            deliveries = [int(c) for c in new_sortie[2]] if len(new_sortie) > 2 else []
            if deliveries:
                new_sortie[2] = self._best_sortie_order(
                    int(new_sortie[1]), deliveries, int(new_sortie[3])
                )
            mapped[v_idx][s_idx] = tuple(new_sortie)
            count+=1

            # 验证修复后的路径是否有效（含真实能耗、悬停能耗与累计 SOC）
            if global_check([list(route) for route in vehicle_routes], mapped,self.model):
                return new_vehicle_routes, mapped
            else:
                new_vehicle_routes,new_drone_routes, removed_info = self.destroy_operator_4(vehicle_routes, drone_routes)
                rem = removed_info[0]
                v_idx = rem.get('v_idx')
                s_idx = rem.get('sortie_idx')
                field = rem.get('field')
            
            if count>10:
                return vehicle_routes, drone_routes



    def destroy_operator_5(self, vehicle_routes, drone_routes):
        """Select a drone label that will receive a newly generated sortie."""
        if not self._drone_labels:
            return vehicle_routes, drone_routes, {}
        chosen_label = random.choice(self._drone_labels)
        v_idx = int(chosen_label.split('-')[0]) - 1
        return vehicle_routes, drone_routes, {'drone_label': chosen_label, 'v_idx': v_idx}
    
    def repair_operator_5(self, vehicle_routes, drone_routes, removed_info):
        """Generate one feasible new drone sortie without repeated full deep copies."""
        if not removed_info or 'v_idx' not in removed_info:
            return vehicle_routes, drone_routes
        v_idx = int(removed_info['v_idx'])
        if not 0 <= v_idx < len(vehicle_routes):
            return vehicle_routes, drone_routes

        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)
        vehicle_route = new_vehicle_routes[v_idx]
        if len(vehicle_route) < 3:
            return vehicle_routes, drone_routes

        blocked_nodes, drone_served = set(), set()
        for group in new_drone_routes:
            for s in group:
                if s[1] is not None:
                    blocked_nodes.add(int(s[1]))
                if s[3] is not None:
                    blocked_nodes.add(int(s[3]))
                drone_served.update(int(d) for d in s[2])

        # Cache endpoint positions for existing sorties once.
        node_pos = {int(node): idx for idx, node in enumerate(vehicle_route) if int(node) != 0}
        node_pos.setdefault(0, 0)
        fallback_attempt = None
        # The input drone routes do not change while scanning candidate windows.
        # Recomputing the same physical drone's committed energy inside every
        # candidate check is unnecessary (especially on long truck routes).
        label = removed_info.get('drone_label', f'NEW-{v_idx + 1}')
        if v_idx < len(new_drone_routes) and label is not None:
            committed_wh = self._drone_energy_used_wh(new_drone_routes, v_idx, label)
        else:
            committed_wh = None  # Preserve rare malformed-input behavior.

        for launch_pos in range(len(vehicle_route) - 1):
            launch_node = int(vehicle_route[launch_pos])
            if launch_node in blocked_nodes or launch_node in drone_served:
                continue
            for rendez_pos in range(launch_pos + 1, len(vehicle_route)):
                rendez_node = int(vehicle_route[rendez_pos])
                if rendez_node == launch_node:
                    continue
                if rendez_node in blocked_nodes or rendez_node in drone_served:
                    continue

                candidates = [
                    int(vehicle_route[idx])
                    for idx in range(launch_pos + 1, rendez_pos)
                    if int(vehicle_route[idx]) not in blocked_nodes
                    and int(vehicle_route[idx]) not in drone_served
                    and int(vehicle_route[idx]) != 0
                ]
                if not candidates:
                    continue

                deliveries = []
                for cand in candidates:
                    trial_deliveries = deliveries + [cand]
                    # Identical payload and candidate-flight-energy checks as
                    # _drone_can_fly(), with committed energy factored out.
                    if not self._payload_within_capacity(trial_deliveries):
                        break
                    if committed_wh is None:
                        # Rare malformed input: preserve the existing helper path.
                        viable = self._drone_can_fly(
                            new_drone_routes, v_idx, label, trial_deliveries,
                            launch_node=launch_node, rendezvous_node=rendez_node,
                        )
                    else:
                        trial_wh = self._best_sortie_energy_wh(
                            launch_node, trial_deliveries, rendez_node
                        )
                        ceiling = self.model.drone_battery_capacity_wh + EPS
                        viable = (trial_wh <= ceiling and committed_wh + trial_wh <= ceiling)
                    if not viable:
                        break
                    deliveries.append(cand)
                if not deliveries:
                    continue

                # 新建 sortie 时使用能耗最优配送顺序
                ordered = self._best_sortie_order(launch_node, deliveries, rendez_node)
                new_sortie = [label, launch_node, list(ordered), rendez_node]
                drone_try = fast_copy_drone_routes(new_drone_routes)
                while v_idx >= len(drone_try):
                    drone_try.append([])

                insert_index = len(drone_try[v_idx])
                for idx, existing in enumerate(drone_try[v_idx]):
                    ex_launch = int(existing[1])
                    ex_pos = 0 if ex_launch == 0 else node_pos.get(ex_launch)
                    if ex_pos is not None and ex_pos > launch_pos:
                        insert_index = idx
                        break
                drone_try[v_idx].insert(insert_index, new_sortie)

                vehicle_try = [list(r) for r in new_vehicle_routes]
                delivery_set = set(deliveries)
                vehicle_try[v_idx] = [n for n in vehicle_try[v_idx] if int(n) not in delivery_set]
                # 完整解的 schedule + hover + SOC 检查
                if global_check(vehicle_try, drone_try, self.model):
                    return vehicle_try, drone_try
                # 该候选的悬停/电池不可行：保留作为最后的修复备选，继续尝试其它起降点
                fallback_attempt = (vehicle_try, drone_try)
        if fallback_attempt is not None:
            return self._finalize(
                vehicle_routes, drone_routes, [], fallback_attempt[0], fallback_attempt[1]
            )
        return vehicle_routes, drone_routes
        
    def _shaw_solution_features(self, vehicle_routes, drone_routes):
        """Build problem-specific features used by the improved Shaw removal.

        Returns
        -------
        completion_times : list[float]
            Completion time of every truck-drone team, including rendezvous waits.
        node_meta : dict[int, dict]
            Service-mode / sortie / synchronization metadata for each customer.
        max_wait : float
            Largest individual truck wait in the current solution.
        max_hover : float
            Largest non-depot drone early-arrival hover time.

        Notes
        -----
        The timing logic mirrors ``calculate_solution_time_cost``: truck travel time
        is accumulated along the vehicle route and, at every rendezvous position,
        the truck waits for the latest drone that has not yet returned.
        """
        n_vehicles = len(vehicle_routes)
        completion_times = [0.0] * n_vehicles
        node_meta = {}
        max_wait = 0.0
        max_hover = 0.0

        vehicle_speed = float(self.model.vehicle_speed)
        drone_speed = float(self.model.drone_speed)
        veh_dist = self.model.veh_dist_matrix
        dro_dist = self.model.dro_dist_matrix

        for v_idx, route_raw in enumerate(vehicle_routes):
            route = [int(x) for x in route_raw]
            if len(route) < 2:
                continue

            # Truck-served customers are first recorded as vehicle-mode nodes.
            pos = {}
            for idx, node in enumerate(route):
                if node != 0:
                    pos[node] = idx
                    if 0 < idx < len(route) - 1:
                        node_meta[node] = {
                            'service_type': 'vehicle',
                            'v_idx': v_idx,
                            'route_pos': idx,
                            'drone_id': None,
                            'sortie_idx': None,
                            'sortie_key': None,
                            'sync_wait': 0.0,
                            'drone_hover': 0.0,
                            'endpoint_sorties': set(),
                        }

            # Base truck arrival times without drone waiting.
            base_arrival = [0.0] * len(route)
            truck_time = 0.0
            for idx in range(1, len(route)):
                a, b = route[idx - 1], route[idx]
                truck_time += float(veh_dist[a, b]) / vehicle_speed
                base_arrival[idx] = truck_time

            # Build sortie timing records grouped by rendezvous position.
            sortie_records = []
            rendez_map = {}
            sorties = drone_routes[v_idx] if v_idx < len(drone_routes) else []
            for s_idx, sortie in enumerate(sorties):
                if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
                    continue
                drone_id = str(sortie[0])
                launch = int(sortie[1])
                deliveries = [int(d) for d in sortie[2]]
                rendez = int(sortie[3])

                launch_idx = 0 if launch == 0 else pos.get(launch)
                rendez_idx = len(route) - 1 if rendez == 0 else pos.get(rendez)
                if launch_idx is None or rendez_idx is None or launch_idx >= rendez_idx:
                    continue

                # Real flight time of the sortie (payload-aware physics), not a
                # plain distance/speed estimate.
                duration = evaluate_sortie_flight(
                    launch, deliveries, rendez, dro_dist,
                    self.model.customer_demands, order=deliveries,
                    speed_m_s=drone_speed,
                )['flight_time_s']
                record = {
                    's_idx': s_idx,
                    'sortie_key': (v_idx, s_idx),
                    'drone_id': drone_id,
                    'launch': launch,
                    'rendez': rendez,
                    'launch_idx': launch_idx,
                    'rendez_idx': rendez_idx,
                    'deliveries': deliveries,
                    'duration': duration,
                    'sync_wait': 0.0,
                    'drone_hover': 0.0,
                }
                sortie_records.append(record)
                rendez_map.setdefault(rendez_idx, []).append(record)

                # Mark structural coupling between truck endpoints and this sortie.
                for endpoint in (launch, rendez):
                    if endpoint != 0 and endpoint in node_meta:
                        node_meta[endpoint]['endpoint_sorties'].add((v_idx, s_idx))

            # Same synchronization propagation as the objective evaluator.
            effective_arrival = [0.0] * len(route)
            accumulated_wait = 0.0
            for idx, base_time in enumerate(base_arrival):
                truck_arrival = base_time + accumulated_wait
                effective_arrival[idx] = truck_arrival
                tasks = rendez_map.get(idx, [])
                if not tasks:
                    continue

                latest_drone = truck_arrival
                for rec in tasks:
                    drone_arrival = effective_arrival[rec['launch_idx']] + rec['duration']
                    rec_wait = max(0.0, drone_arrival - truck_arrival)
                    rec['sync_wait'] = rec_wait
                    # The evaluator never charges hover at the final depot.
                    rec['drone_hover'] = (
                        max(0.0, truck_arrival - drone_arrival)
                        if rec['rendez'] != 0 else 0.0
                    )
                    max_hover = max(max_hover, rec['drone_hover'])
                    max_wait = max(max_wait, rec_wait)
                    latest_drone = max(latest_drone, drone_arrival)

                if latest_drone > truck_arrival:
                    accumulated_wait += latest_drone - truck_arrival
                    effective_arrival[idx] = latest_drone

            completion_times[v_idx] = truck_time + accumulated_wait

            # Drone-served customers inherit their sortie's synchronization burden.
            for rec in sortie_records:
                for node in rec['deliveries']:
                    node_meta[node] = {
                        'service_type': 'drone',
                        'v_idx': v_idx,
                        'route_pos': None,
                        'drone_id': rec['drone_id'],
                        'sortie_idx': rec['s_idx'],
                        'sortie_key': rec['sortie_key'],
                        'sync_wait': rec['sync_wait'],
                        'drone_hover': rec['drone_hover'],
                        'endpoint_sorties': set(),
                    }

                # A truck endpoint also carries the synchronization burden of the
                # sortie that launches/rendezvouses there.  This makes directly
                # coupled truck-drone structures more likely to be removed together.
                for endpoint in (rec['launch'], rec['rendez']):
                    if endpoint != 0 and endpoint in node_meta:
                        node_meta[endpoint]['sync_wait'] = max(
                            float(node_meta[endpoint].get('sync_wait', 0.0)),
                            float(rec['sync_wait']),
                        )
                        node_meta[endpoint]['drone_hover'] = max(
                            float(node_meta[endpoint].get('drone_hover', 0.0)),
                            float(rec['drone_hover']),
                        )

        return completion_times, node_meta, max_wait, max_hover

    @staticmethod
    def _shaw_mode_relatedness(seed_meta, other_meta):
        """Smaller value means stronger truck-drone structural coupling."""
        if not seed_meta or not other_meta:
            return 1.0

        same_vehicle = seed_meta.get('v_idx') == other_meta.get('v_idx')
        if not same_vehicle:
            return 1.0

        seed_type = seed_meta.get('service_type')
        other_type = other_meta.get('service_type')
        seed_key = seed_meta.get('sortie_key')
        other_key = other_meta.get('sortie_key')

        # Same drone sortie: strongest possible relation.
        if seed_type == 'drone' and other_type == 'drone' and seed_key == other_key:
            return 0.0

        # A truck launch/rendezvous node and a drone customer from that exact sortie.
        seed_endpoints = seed_meta.get('endpoint_sorties', set())
        other_endpoints = other_meta.get('endpoint_sorties', set())
        if seed_key is not None and seed_key in other_endpoints:
            return 0.1
        if other_key is not None and other_key in seed_endpoints:
            return 0.1

        # Same physical drone, different sorties.
        if (
            seed_type == 'drone'
            and other_type == 'drone'
            and seed_meta.get('drone_id') == other_meta.get('drone_id')
        ):
            return 0.2

        # Same truck-drone team but different service modes.
        if seed_type != other_type:
            return 0.4

        # Same vehicle, same broad service mode, but no direct sortie coupling.
        return 0.6

    def destroy_operator_6(self, vehicle_routes, drone_routes, removal_ratio=None):
        """Bottleneck- and synchronization-aware Shaw related removal.

        Compared with the original Shaw operator, this version changes only the
        destruction logic.  The downstream Regret-2 repair is intentionally kept
        unchanged so the contribution of the new removal strategy can be isolated.

        Relatedness combines:
          1) service-mode-aware spatial distance;
          2) demand similarity;
          3) truck-drone structural/service-mode similarity;
          4) bidirectional truck-wait / drone-hover similarity;
          5) route criticality similarity under the makespan objective.

        The seed customer is no longer uniformly random.  Customers belonging to
        longer (more critical) truck-drone routes and customers associated with
        larger truck waits or drone hover times receive larger sampling probabilities.
        Related customers are drawn with rank-biased stochastic selection.
        """
        customer_pool = get_all_customers_with_details(
            self.model, vehicle_routes, drone_routes
        )

        if not customer_pool:
            return [list(r) for r in vehicle_routes], fast_copy_drone_routes(drone_routes), []

        # Needed by repair_operator_6 as its exact fallback solution.
        self.save_path(vehicle_routes, drone_routes)

        completion_times, node_meta, max_wait, max_hover = self._shaw_solution_features(
            vehicle_routes, drone_routes
        )
        cmax = max(completion_times) if completion_times else 0.0
        if not np.isfinite(cmax) or cmax <= 1e-12:
            cmax = 1.0

        # ------------------------------------------------------------------
        # 1) Bottleneck/synchronization-aware seed selection.
        #    eta > 1 concentrates probability on critical routes while keeping
        #    stochastic diversity.  High-wait customers receive an extra boost.
        # ------------------------------------------------------------------
        eta = 3.0
        seed_weights = []
        for info in customer_pool:
            node = int(info['node'])
            meta = node_meta.get(node, {})
            v_idx = int(info.get('v_idx', meta.get('v_idx', 0)))
            if 0 <= v_idx < len(completion_times):
                criticality = max(0.0, min(1.0, completion_times[v_idx] / cmax))
            else:
                criticality = 0.0
            wait_norm = (
                float(meta.get('sync_wait', 0.0)) / max_wait
                if max_wait > 1e-12 else 0.0
            )
            hover_norm = (
                float(meta.get('drone_hover', 0.0)) / max_hover
                if max_hover > 1e-12 else 0.0
            )
            # Include both directions of synchronization in the seed bias.
            weight = 0.05 + criticality ** eta * (1.0 + wait_norm + hover_norm)
            seed_weights.append(weight)

        try:
            seed_info = random.choices(customer_pool, weights=seed_weights, k=1)[0]
        except Exception:
            seed_info = random.choice(customer_pool)
        seed_node = int(seed_info['node'])
        seed_meta = node_meta.get(seed_node, {})

        # Normalizers for the heterogeneous distance metrics.
        veh_dist_mat = np.asarray(self.model.veh_dist_matrix, dtype=float)
        dro_dist_mat = np.asarray(self.model.dro_dist_matrix, dtype=float)
        max_veh_dist = self._cached_max_veh_dist
        max_dro_dist = max(
            1e-12,
            float(getattr(
                self.model,
                '_drone_distance_normalizer',
                np.max(dro_dist_mat) if dro_dist_mat.size else 1.0,
            )),
        )
        max_demand = max(1.0, float(getattr(self.model, '_max_customer_demand', 1.0)))

        # ------------------------------------------------------------------
        # 2) Problem-specific Shaw relatedness.
        #    Smaller score => more related => more likely to be jointly removed.
        #    These are explicit algorithm parameters and can later be calibrated
        #    through a small sensitivity analysis.
        # ------------------------------------------------------------------
        w_dist = 2.0
        w_demand = 1.0
        w_mode = 2.0
        w_sync = 3.0
        w_critical = 2.0

        seed_type = seed_meta.get('service_type', 'vehicle')
        seed_v_idx = int(seed_info.get('v_idx', seed_meta.get('v_idx', 0)))
        seed_criticality = (
            completion_times[seed_v_idx] / cmax
            if 0 <= seed_v_idx < len(completion_times) else 0.0
        )
        seed_wait = float(seed_meta.get('sync_wait', 0.0))
        seed_hover = float(seed_meta.get('drone_hover', 0.0))
        seed_demand = float(seed_info['demand'])

        infos = []
        scores = []
        for info in customer_pool:
            node = int(info['node'])
            if node == seed_node:
                continue

            meta = node_meta.get(node, {})
            other_type = meta.get('service_type', 'vehicle')

            # Service-mode-aware spatial relatedness.
            truck_rel = float(veh_dist_mat[seed_node, node]) / max_veh_dist
            drone_rel = float(dro_dist_mat[seed_node, node]) / max_dro_dist
            if seed_type == 'vehicle' and other_type == 'vehicle':
                dist_rel = truck_rel
            elif seed_type == 'drone' and other_type == 'drone':
                dist_rel = drone_rel
            else:
                # Cross-mode customers are evaluated in both geometries.
                dist_rel = 0.5 * (truck_rel + drone_rel)

            demand_rel = abs(seed_demand - float(info['demand'])) / max_demand
            mode_rel = self._shaw_mode_relatedness(seed_meta, meta)

            other_wait = float(meta.get('sync_wait', 0.0))
            other_hover = float(meta.get('drone_hover', 0.0))
            # Each direction is normalized independently so an early drone
            # is not masked by much larger truck waiting elsewhere.
            wait_rel = (
                abs(seed_wait - other_wait) / max_wait
                if max_wait > 1e-12 else 0.0
            )
            hover_rel = (
                abs(seed_hover - other_hover) / max_hover
                if max_hover > 1e-12 else 0.0
            )
            sync_rel = 0.5 * wait_rel + 0.5 * hover_rel

            other_v_idx = int(info.get('v_idx', meta.get('v_idx', 0)))
            other_criticality = (
                completion_times[other_v_idx] / cmax
                if 0 <= other_v_idx < len(completion_times) else 0.0
            )
            critical_rel = abs(seed_criticality - other_criticality)

            score = (
                w_dist * dist_rel
                + w_demand * demand_rel
                + w_mode * mode_rel
                + w_sync * sync_rel
                + w_critical * critical_rel
            )
            infos.append(info)
            scores.append(float(score))

        scores = np.asarray(scores, dtype=float)

        # Normal search keeps the original 5% destruction scale.  DUAL-LNS may
        # pass a larger ``removal_ratio`` after a long stagnation period to create
        # a deliberate diversification jump.  The default remains unchanged for
        # ordinary operator-6 calls and for the standalone LNS/ALNS baselines.
        num_customers = len(customer_pool)
        ratio = 0.05 if removal_ratio is None else float(removal_ratio)
        ratio = max(0.0, min(1.0, ratio))
        q = int(ratio * num_customers)
        q = min(num_customers, max(2 if num_customers >= 2 else 1, q))
        k = max(0, min(q - 1, len(scores)))

        # Rank-biased stochastic Shaw selection, without replacement.
        # u**p favours low-relatedness ranks while allowing exploration.
        # Re-rank after each draw to avoid selecting the same customer twice.
        rank_bias_p = 4.0
        ranked = sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(infos[i]['node'])))
        selected_infos = []
        for _ in range(k):
            m = len(ranked)
            rank = min(m - 1, int(random.random() ** rank_bias_p * m))
            selected_infos.append(infos[ranked.pop(rank)])

        nodes_to_remove_info = [seed_info] + selected_infos
        nodes_to_remove_set = {int(info['node']) for info in nodes_to_remove_info}

        # ======= The deletion / cascading-feasibility logic below is preserved =======
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)

        # 1) Remove selected truck-served customers.
        vehicle_deletes = {}
        for info in nodes_to_remove_info:
            if info['loc'][0] == 'vehicle':
                _, v_idx, node_idx = info['loc']
                vehicle_deletes.setdefault(v_idx, set()).add(int(node_idx))

        for v_idx, idx_set in vehicle_deletes.items():
            try:
                route = new_vehicle_routes[v_idx]
            except Exception:
                continue
            node_to_indices = {}
            for i, nd in enumerate(route):
                node_to_indices.setdefault(int(nd), []).append(i)
            actual_delete_indices = []
            for n in nodes_to_remove_set:
                if n in node_to_indices:
                    actual_delete_indices.extend(node_to_indices[n])
            for idx in sorted(set(actual_delete_indices), reverse=True):
                try:
                    del new_vehicle_routes[v_idx][idx]
                except Exception:
                    pass

        # 2) Remove selected drone-served customers; delete empty sorties.
        for v_idx in range(len(new_drone_routes)):
            valid_sorties = []
            for sortie in new_drone_routes[v_idx]:
                try:
                    delivery_nodes = sortie[2]
                except Exception:
                    delivery_nodes = []
                new_delivery = [
                    nd for nd in delivery_nodes
                    if int(nd) not in nodes_to_remove_set
                ]
                if len(new_delivery) == 0:
                    continue
                new_sortie = list(sortie)
                new_sortie[2] = new_delivery
                valid_sorties.append(
                    tuple(new_sortie) if isinstance(sortie, tuple) else new_sortie
                )
            new_drone_routes[v_idx] = valid_sorties

        # 3) Cascade: removing a truck launch/rendezvous node invalidates its sortie.
        all_vehicle_nodes = {
            int(node) for route in new_vehicle_routes for node in route
        }
        final_removed_nodes = set(nodes_to_remove_set)
        for v_idx in range(len(new_drone_routes)):
            valid_sorties = []
            for sortie in new_drone_routes[v_idx]:
                l, r = sortie[1], sortie[3]
                if int(l) not in all_vehicle_nodes or int(r) not in all_vehicle_nodes:
                    final_removed_nodes.update(int(n) for n in sortie[2])
                    continue
                valid_sorties.append(sortie)
            new_drone_routes[v_idx] = valid_sorties

        return new_vehicle_routes, new_drone_routes, list(final_removed_nodes)


    def repair_operator_6(self, destroyed_vehicle_routes, destroyed_drone_routes, removed_nodes):
        """Regret-2 repair for Shaw removal with incremental insertion costs.

        This preserves the original distance-based regret logic but avoids sorting
        every move list and avoids recomputing complete drone paths for each
        insertion position.
        """
        new_vehicle_routes = [list(r) for r in destroyed_vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(destroyed_drone_routes)
        unassigned_nodes = list(removed_nodes)

        for retry in range(21):
            while unassigned_nodes:
                selected = None  # (regret, order, node, loc)

                # Sortie base statistics are identical for every candidate node in
                # this regret round; compute them once instead of once per node.
                # Each sortie carries its real (energy-optimal) flight energy and
                # the cumulative energy already committed by its physical drone.
                drone_meta = []
                drone_energy_total = {}
                for v_idx, sorties in enumerate(new_drone_routes):
                    for s_idx, sortie in enumerate(sorties):
                        launch = int(sortie[1])
                        deliveries = [int(d) for d in sortie[2]]
                        rendez = int(sortie[3])
                        label = str(sortie[0])
                        base_payload = self._payload_kg(deliveries)
                        base_energy = self._best_sortie_energy_wh(launch, deliveries, rendez)
                        drone_meta.append(
                            (v_idx, s_idx, deliveries, label, launch, rendez,
                             base_payload, base_energy)
                        )
                        key = (v_idx, label)
                        drone_energy_total[key] = drone_energy_total.get(key, 0.0) + base_energy

                for order, node in enumerate(unassigned_nodes):
                    node = int(node)
                    best_cost = float('inf')
                    second_cost = float('inf')
                    best_loc = None

                    def consider(cost, loc):
                        nonlocal best_cost, second_cost, best_loc
                        cost = float(cost)
                        if cost < best_cost:
                            second_cost = best_cost
                            best_cost = cost
                            best_loc = loc
                        elif cost < second_cost:
                            second_cost = cost

                    # Truck insertions: Manhattan-distance delta, same criterion as
                    # the original implementation.
                    for v_idx, route in enumerate(new_vehicle_routes):
                        for pos in range(1, len(route)):
                            prev_node, next_node = int(route[pos - 1]), int(route[pos])
                            cost = (
                                self.model.veh_dist_matrix[prev_node, node]
                                + self.model.veh_dist_matrix[node, next_node]
                                - self.model.veh_dist_matrix[prev_node, next_node]
                            )
                            consider(cost, ('vehicle', v_idx, pos))

                    # Drone insertions are ranked by the *real energy delta*, not
                    # by the distance delta:
                    #   delta_energy = new_sortie_energy - old_sortie_energy
                    # The delta is converted into an equivalent flight distance
                    # only so that it stays comparable with the truck cost.
                    node_demand = self.model.customer_demands[node]
                    for (v_idx, s_idx, deliveries, label, launch, rendez,
                         base_payload, base_energy) in drone_meta:
                        # 第一步：载重
                        if base_payload + node_demand > self.model.drone_payload_capacity + 1e-12:
                            continue
                        other_energy = drone_energy_total.get((v_idx, label), 0.0) - base_energy
                        payload_after = base_payload + node_demand
                        for pos in range(len(deliveries) + 1):
                            candidate_deliveries = (
                                deliveries[:pos] + [node] + deliveries[pos:]
                            )
                            # 第二步/第三步：重新优化顺序并重算真实能耗
                            new_energy = self._best_sortie_energy_wh(
                                launch, candidate_deliveries, rendez
                            )
                            # 第四步：该物理无人机累计 SOC 必须仍然可行
                            if (other_energy + new_energy
                                    > self.model.drone_battery_capacity_wh + EPS):
                                continue
                            delta_energy = new_energy - base_energy
                            cost = self._energy_equivalent_distance(
                                delta_energy, payload_after
                            )
                            consider(cost, ('drone', v_idx, s_idx, pos))

                    if best_loc is None:
                        continue
                    regret = 0.0 if second_cost == float('inf') else second_cost - best_cost
                    candidate = (regret, -order, node, best_loc)
                    if selected is None or candidate[:2] > selected[:2]:
                        selected = candidate

                if selected is None:
                    break

                _, _, node, loc = selected
                if loc[0] == 'vehicle':
                    _, v_idx, pos = loc
                    new_vehicle_routes[v_idx].insert(pos, node)
                else:
                    _, v_idx, s_idx, pos = loc
                    sortie = new_drone_routes[v_idx][s_idx]
                    deliveries = [int(c) for c in sortie[2]]
                    deliveries.insert(pos, node)
                    # 插入后重新做 energy-aware 配送顺序优化
                    sortie[2] = self._best_sortie_order(sortie[1], deliveries, sortie[3])
                unassigned_nodes.remove(node)

            if not unassigned_nodes:
                return self._finalize(
                    destroyed_vehicle_routes, destroyed_drone_routes, removed_nodes,
                    new_vehicle_routes, new_drone_routes,
                )

            original_vehicle_routes, original_drone_routes = self.get_path()
            if retry >= 20:
                return original_vehicle_routes, original_drone_routes
            destroyed_vehicle_routes, destroyed_drone_routes, removed_nodes = self.destroy_operator_6(
                original_vehicle_routes, original_drone_routes
            )
            new_vehicle_routes = [list(r) for r in destroyed_vehicle_routes]
            new_drone_routes = fast_copy_drone_routes(destroyed_drone_routes)
            unassigned_nodes = list(removed_nodes)

        return self.get_path()

    def destroy_operator_7(self, vehicle_routes, drone_routes):
        """
        破坏算子8: 随机选择一个无人机任务，将其起飞点（或降落点）从车辆路径中移除，
        作为客户插入到该无人机任务中，并将起飞点（或降落点）移动到车辆路径上的前一个（或后一个）节点。
        """
        new_vehicle_routes = [list(route) for route in vehicle_routes]
        new_drone_routes = fast_copy_drone_routes(drone_routes)
        
        # 1. 收集所有可行的候选操作
        candidates = []
        # 遍历所有车辆的所有无人机任务
        for v_idx, sorties in enumerate(new_drone_routes):
            if v_idx >= len(new_vehicle_routes): continue
            vehicle_route = new_vehicle_routes[v_idx]
            
            for s_idx, sortie in enumerate(sorties):
                # 任务结构: [label, launch_node, delivery_nodes, rendez_node]
                launch_node = sortie[1]
                rendez_node = sortie[3]
                
                # 检查起飞点是否可以作为候选
                # 要求: Launch点必须是客户(非0)，且在车辆路径中存在
                if launch_node is not None and launch_node != 0:
                    try:
                        l_pos = vehicle_route.index(launch_node)
                        # Launch点移动规则: 新Launch = vehicle_route[l_pos - 1]
                        if l_pos > 0:
                            candidates.append({
                                'type': 'launch',
                                'v_idx': v_idx,
                                's_idx': s_idx,
                                'node': launch_node,
                                'pos_in_vehicle': l_pos
                            })
                    except ValueError:
                        pass 

                # 检查降落点是否可以作为候选
                if rendez_node is not None and rendez_node != 0:
                    try:
                        r_pos = vehicle_route.index(rendez_node)
                        # Rendez点移动规则: 新Rendez = vehicle_route[r_pos + 1]
                        if r_pos < len(vehicle_route) - 1:
                            candidates.append({
                                'type': 'rendez',
                                'v_idx': v_idx,
                                's_idx': s_idx,
                                'node': rendez_node,
                                'pos_in_vehicle': r_pos
                            })
                    except ValueError:
                        pass

        if not candidates:
            return new_vehicle_routes, new_drone_routes, []

        # 尝试多次，直到找到一个可行解
        random.shuffle(candidates)
        for cand in candidates:
            # 深拷贝用于尝试
            temp_vehicle_routes = [list(route) for route in new_vehicle_routes]
            temp_drone_routes = fast_copy_drone_routes(new_drone_routes)
            
            v_idx = cand['v_idx']
            s_idx = cand['s_idx']
            node_to_move = cand['node']
            
            route = temp_vehicle_routes[v_idx]
            pos_to_remove = cand['pos_in_vehicle']
            
            # 再次确认节点值匹配
            if pos_to_remove >= len(route) or route[pos_to_remove] != node_to_move:
                continue

            # 确定新的起降点节点值
            if cand['type'] == 'launch':
                new_endpoint_node = route[pos_to_remove - 1]
            else:
                new_endpoint_node = route[pos_to_remove + 1]
            
            # 从车辆路径执行移除
            del route[pos_to_remove]

            # --- 步骤2: 更新所有受影响的无人机任务 ---
            sorties = temp_drone_routes[v_idx]
            possible = True
            
            for i, s in enumerate(sorties):
                # 替换端点
                if s[1] == node_to_move:
                    s[1] = new_endpoint_node
                if s[3] == node_to_move:
                    s[3] = new_endpoint_node
                
                # 特别处理目标任务：插入 node_to_move 到配送列表
                if i == s_idx:
                    target_deliveries = None
                    min_energy = float('inf')
                    original_deliveries = [int(c) for c in s[2]]
                    curr_launch = s[1]
                    curr_rendez = s[3]
                    drone_label = s[0]

                    # 尝试所有插入位置：以真实能耗（能耗最优顺序下）作为评价依据
                    for ins_pos in range(len(original_deliveries) + 1):
                        temp_dels = (
                            original_deliveries[:ins_pos] + [node_to_move]
                            + original_deliveries[ins_pos:]
                        )

                        # 第一步：载重 <= 3 kg
                        if not self._payload_within_capacity(temp_dels):
                            continue
                        # 第二/三/四步：顺序重新优化 + 真实能耗 + 该无人机累计 SOC
                        if not self._drone_can_fly(
                            temp_drone_routes, v_idx, drone_label, temp_dels,
                            launch_node=curr_launch, rendezvous_node=curr_rendez,
                            skip_sortie_idx=s_idx,
                        ):
                            continue
                        energy = self._best_sortie_energy_wh(
                            curr_launch, temp_dels, curr_rendez
                        )
                        if energy < min_energy:
                            min_energy = energy
                            target_deliveries = temp_dels

                    if target_deliveries is not None:
                        s[2] = self._best_sortie_order(
                            curr_launch, target_deliveries, curr_rendez
                        )
                    else:
                        possible = False  # 无法插入
                        break

            if not possible:
                continue

            # --- 步骤3: 全局可行性检查（真实能耗 + 悬停能耗 + 累计 SOC） ---
            all_feasible = True
            for i, s in enumerate(sorties):
                deliveries = [int(c) for c in s[2]]
                if not deliveries or not self._payload_within_capacity(deliveries):
                    all_feasible = False
                    break
                if not self._drone_can_fly(
                    temp_drone_routes, v_idx, s[0], deliveries,
                    launch_node=s[1], rendezvous_node=s[3], skip_sortie_idx=i,
                ):
                    all_feasible = False
                    break
            if not all_feasible:
                continue

            self._normalize_drone_orders(temp_drone_routes)
            if global_check(temp_vehicle_routes, temp_drone_routes, self.model):
                return temp_vehicle_routes, temp_drone_routes, []

        return new_vehicle_routes, new_drone_routes, []

    def repair_operator_7(self, vehicle_routes, drone_routes, removed_nodes):
        """
        修复算子8: 这是一个空操作，因为 destroy_operator_7 已经完成了破坏和修复（变异）。
        """
        return vehicle_routes, drone_routes

    



    
    
    
    
