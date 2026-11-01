import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

import sweep_clustering

from drone_physics import (  # re-exported for callers that import from utils
    dp_energy_fixed_lr,
    energy_optimal_sortie,
    sortie_energy_wh,
    drone_power_w,
)
from evaluator import evaluate_solution

__all__ = [
    'nearest_neighbor_insertion', 'drone_clusters', 'dp_tsp_fixed_lr',
    'dp_energy_fixed_lr', 'energy_optimal_sortie', 'sortie_energy_wh',
    'drone_power_w', 'fast_copy_drone_routes', 'farthest_insertion',
    'global_check', 'is_sortie_feasible', 'insert_node_at_loc',
    'insert_two_nodes_safely', 'find_occurrence_between',
    'get_all_customers_with_details', 'calculate_vehicle_arrival_times',
    'evaluate_solution',
    'initial_customer_partition', 'kmeans_customer_partition',
    'partition_to_node_ids',
]


# --------------------------------------------------------------------------
# Truck customer partition (the initial clustering decision).
#
# This is the ONLY place where customers are grouped into trucks, and it is the
# only thing the angular sweep replaces.  ``drone_clusters`` below is a
# different decision (how one truck's customers are split among the drones it
# carries) and is deliberately left untouched.
# --------------------------------------------------------------------------
def partition_to_node_ids(partition):
    """0-based indices into ``customer_coords`` -> 1-based node ids.

    The sweep module works with plain list indices; the distance matrix and the
    TSP helpers index node ``i + 1`` for ``customer_coords[i]``.  This is the one
    conversion between the two conventions.
    """
    return [[int(index) + 1 for index in cluster] for cluster in partition]


def kmeans_customer_partition(customer_coords, num_vehicles, seed):
    """Legacy KMeans truck partition, kept for initialisation comparison runs."""
    kmeans_vehicles = KMeans(n_clusters=num_vehicles, random_state=seed, n_init=10).fit(customer_coords)
    # 1 索引的客户 ID，例如 [[1, 4, 5], [2, 3, 6]]
    return [(np.where(kmeans_vehicles.labels_ == i)[0] + 1).tolist()
            for i in range(num_vehicles)]


def initial_customer_partition(customer_coords, num_vehicles, depot_coord,
                               customer_demands=None, drone_payload_capacity=None,
                               seed=None, method=None, rotation=0.0):
    """Group customers into ``num_vehicles`` truck clusters.

    ``method='sweep'``  -- balanced angular sweep around the depot (default).
                           Needs ``customer_demands`` and
                           ``drone_payload_capacity``: the sweep balances on a
                           load score and must know which customers are
                           truck-only.
    ``method='kmeans'`` -- the legacy KMeans partition, for comparison runs.

    ``method=None`` follows ``sweep_clustering.INITIAL_CLUSTERING_METHOD``, read
    at call time so ``set_initial_clustering_method`` can switch strategies
    inside a running process.

    Returns ``num_vehicles`` lists of 1-based customer ids; their union is
    exactly ``1..len(customer_coords)``.
    """
    resolved = sweep_clustering.get_initial_clustering_method(method)
    if resolved == 'kmeans':
        return kmeans_customer_partition(customer_coords, num_vehicles, seed)
    if resolved != 'sweep':
        raise ValueError("unknown initial clustering method %r" % (resolved,))
    if customer_demands is None or drone_payload_capacity is None:
        raise ValueError(
            "the angular sweep needs customer_demands and drone_payload_capacity "
            "to balance the sectors; pass them, or select method='kmeans'"
        )
    partition = sweep_clustering.balanced_sweep_partition(
        customer_coords, depot_coord, num_vehicles, drone_payload_capacity,
        customer_demands, rotation=rotation,
    )
    return partition_to_node_ids(partition)


def farthest_insertion(customer_coords, num_vehicles, depot_coord, dist_matrix, seed,
                       partition=None, customer_demands=None,
                       drone_payload_capacity=None):
    """Farthest-insertion TSP per truck cluster.

    The truck cluster only fixes *which delivery system* owns a customer; the
    visiting order inside a cluster is produced by the normal farthest-insertion
    constructor, never by the sweep's angular order.

    ``partition`` (1-based customer ids, one list per truck) may be supplied by
    the caller -- the multi-start sweep driver uses this to evaluate several
    angular partitions with the same route constructor.  When it is ``None`` the
    configured clustering method builds it here.
    """
    if len(customer_coords) == 0:
        return [], 0.0
    if num_vehicles <= 0 or num_vehicles > len(customer_coords):
        raise ValueError('num_vehicles must be in [1, number of customers]')

    if partition is None:
        vehicle_clusters = initial_customer_partition(
            customer_coords, num_vehicles, depot_coord,
            customer_demands=customer_demands,
            drone_payload_capacity=drone_payload_capacity,
            seed=seed,
        )
    else:
        vehicle_clusters = [list(cluster) for cluster in partition]

    all_routes = []
    total_distance = 0.0

    for cluster_nodes in vehicle_clusters:
        if len(cluster_nodes) == 0:
            continue  # 跳过空簇

        # 调用辅助函数来为这个簇求解 TSP
        route, route_dist = _solve_tsp_farthest_insertion(cluster_nodes, dist_matrix)

        all_routes.append(route)
        total_distance += route_dist

    return all_routes, total_distance


def nearest_neighbor_insertion(customer_coords, num_vehicles, depot_coord, dist_matrix, seed,
                               partition=None, customer_demands=None,
                               drone_payload_capacity=None):
    # 保持与最远插入法一致的接口，便于在模型中直接替换
    if len(customer_coords) == 0:
        return [], 0.0
    if num_vehicles <= 0 or num_vehicles > len(customer_coords):
        raise ValueError('num_vehicles must be in [1, number of customers]')

    if partition is None:
        vehicle_clusters = initial_customer_partition(
            customer_coords, num_vehicles, depot_coord,
            customer_demands=customer_demands,
            drone_payload_capacity=drone_payload_capacity,
            seed=seed,
        )
    else:
        vehicle_clusters = [list(cluster) for cluster in partition]

    all_routes = []
    total_distance = 0.0

    for cluster_nodes in vehicle_clusters:
        if len(cluster_nodes) == 0:
            continue

        route, route_dist = _solve_tsp_nearest_neighbor(cluster_nodes, dist_matrix)
        all_routes.append(route)
        total_distance += route_dist

    return all_routes, total_distance


def _solve_tsp_farthest_insertion(customer_nodes, dist_matrix):
    # 转换为 set 以便高效地移除
    unvisited = set(customer_nodes)
    
    # --- 1. 初始化 ---
    # 找到离仓库 (0) 最远的客户 k
    k = max(unvisited, key=lambda c: dist_matrix[0][c])
    
    # 初始路径 (sub-tour): [0, k, 0]
    route = [0, k, 0]
    unvisited.remove(k)

    # --- 2. 迭代插入 ---
    while unvisited:
        
        # --- a. 选择客户 (S) ---
        # 找到一个未访问的客户 u，它与已在路径中的 *某个* 客户的距离最远
        farthest_u = None
        max_min_dist = -1 # 存储“最远的”那个“最小距离”
        
        # 遍历所有未访问的客户
        for u in unvisited:
            # 计算 u 到 *所有* 已在路径中节点的最小距离
            # (注意：route 包含仓库 0)
            min_dist_to_route = min(dist_matrix[u][j] for j in route)
            
            if min_dist_to_route > max_min_dist:
                max_min_dist = min_dist_to_route
                farthest_u = u
        
        u = farthest_u # 这就是我们本轮要插入的客户
        
        # --- b. 找到最佳插入位置 (I) ---
        # 遍历路径中的每条边 (i, j)，找到插入 u 的最小成本
        min_insert_cost = float('inf')
        best_insert_index = -1
        
        # 迭代 route 中的所有“边”
        for i in range(len(route) - 1):
            node_i = route[i]
            node_j = route[i+1]
            
            # 插入成本 = dist(i, u) + dist(u, j) - dist(i, j)
            cost = dist_matrix[node_i][u] + dist_matrix[u][node_j] - dist_matrix[node_i][node_j]
            
            if cost < min_insert_cost:
                min_insert_cost = cost
                # 我们将在索引 i+1 的位置插入 u
                best_insert_index = i + 1
                
        # --- c. 执行插入 ---
        route.insert(best_insert_index, u)
        unvisited.remove(u)

    # --- 3. 计算最终路径总距离 ---
    route_dist = 0.0
    for i in range(len(route) - 1):
        route_dist += dist_matrix[route[i]][route[i+1]]
        
    return route, route_dist

def _solve_tsp_nearest_neighbor(customer_nodes, dist_matrix):
    # 转换为 set 以便高效地移除（和原代码保持一致）
    unvisited = set(customer_nodes)
    
    # --- 1. 初始化 ---
    # 从仓库 0 开始
    route = [0]
    current_node = 0  # 当前所在位置
    
    # --- 2. 迭代选择最近邻 ---
    while unvisited:
        # --- a. 找到距离当前节点最近的未访问客户 u ---
        nearest_u = min(unvisited, key=lambda c: dist_matrix[current_node][c])
        
        # --- b. 将最近客户加入路径，并更新当前位置 ---
        route.append(nearest_u)
        unvisited.remove(nearest_u)
        current_node = nearest_u

    # --- 3. 最后返回仓库 ---
    route.append(0)
    
    # --- 4. 计算最终路径总距离（和原代码完全相同） ---
    route_dist = 0.0
    for i in range(len(route) - 1):
        route_dist += dist_matrix[route[i]][route[i+1]]
        
    return route, route_dist



def drone_clusters(route, num_drones_per_vehicle, coords, avg_drone_service, seed):
    customers = [int(i) for i in route if int(i) != 0]
    if not customers:
        return []
    if num_drones_per_vehicle <= 0:
        return []
    denom = max(1, num_drones_per_vehicle * (max(0, int(avg_drone_service)) + 2))
    k2 = max(1, len(customers) // denom)
    k2 = min(k2, len(customers))
    coords = np.asarray(coords)
    if len(coords) != len(customers):
        raise ValueError("coords/customer count mismatch in drone_clusters")
    # n_init is passed explicitly so the result does not depend on the sklearn
    # default (it changes in 1.4) and so sklearn stops emitting the FutureWarning.
    kmeans_drones = KMeans(n_clusters=k2, random_state=seed, n_init=10).fit(coords)
    drone_subclusters = [[] for _ in range(k2)]
    for j, label in enumerate(kmeans_drones.labels_):
        drone_subclusters[int(label)].append(customers[j])

    pos = {node: idx for idx, node in enumerate(route)}
    cluster_infos = []
    for orig_idx, cluster in enumerate(drone_subclusters):
        in_route_idxs = [pos[n] for n in cluster]
        cluster_infos.append((min(in_route_idxs), orig_idx, cluster))
    cluster_infos.sort(key=lambda x: (x[0], x[1]))
    return [info[2] for info in cluster_infos]


def calculate_vehicle_arrival_times(route,veh_dist_matrix,vehicle_speed):
    """
    计算车辆按给定路径的每个节点到达时间（基于曼哈顿距离）
    """
    # 初始化到达时间列表，起点时间为0
    arrival_times = [0.0]
    current_time = 0.0
    # 遍历路径中的每一段
    for i in range(len(route) - 1):
        # 获取起始节点和目标节点
        from_node = int(route[i])
        to_node = int(route[i+1])
        # 计算两点间的曼哈顿距离
        distance = veh_dist_matrix[from_node, to_node]
        # 根据车辆速度计算行驶时间并累加到当前时间
        current_time += distance / vehicle_speed
        # 将到达时间添加到列表中
        arrival_times.append(current_time)
    return arrival_times


def dp_tsp_fixed_lr(l_idx, r_idx, d_indices, dro_dist_matrix):
    """Exact TSP order for fixed launch/rendezvous and a small set of drone customers.

    Returns ``(launch, ordered_deliveries, rendezvous)`` for every input size,
    including the empty-delivery case.  The previous implementation returned a
    two-item tuple when ``d_indices`` was empty, which broke callers that unpack
    three values.
    """
    l_idx, r_idx = int(l_idx), int(r_idx)
    d_indices = [int(x) for x in d_indices]
    n = len(d_indices)
    if n == 0:
        return l_idx, [], r_idx
    if n == 1:
        return l_idx, d_indices.copy(), r_idx

    size_mask = 1 << n
    inf = float('inf')
    dp = [[inf] * n for _ in range(size_mask)]
    prev = [[-1] * n for _ in range(size_mask)]

    for u, g in enumerate(d_indices):
        dp[1 << u][u] = float(dro_dist_matrix[l_idx, g])

    for mask in range(1, size_mask):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            current = dp[mask][u]
            if current == inf:
                continue
            u_global = d_indices[u]
            remaining = ((1 << n) - 1) ^ mask
            v = 0
            while remaining:
                if remaining & 1:
                    new_mask = mask | (1 << v)
                    new_dist = current + float(dro_dist_matrix[u_global, d_indices[v]])
                    if new_dist < dp[new_mask][v]:
                        dp[new_mask][v] = new_dist
                        prev[new_mask][v] = u
                remaining >>= 1
                v += 1

    full = size_mask - 1
    best_last = min(
        range(n),
        key=lambda u: dp[full][u] + float(dro_dist_matrix[d_indices[u], r_idx]),
    )

    order_local = []
    mask = full
    u = best_last
    while u != -1:
        order_local.append(u)
        pu = prev[mask][u]
        mask &= ~(1 << u)
        u = pu
    order = [d_indices[i] for i in reversed(order_local)]
    return l_idx, order, r_idx

def insert_node_at_loc(vehicle_routes, drone_routes, node, loc):
    # 获取位置类型（车辆路径或无人机任务）
    loc_type = loc[0]
    
    # 根据位置类型执行不同的插入操作
    if loc_type == 'vehicle':
        # 车辆路径节点插入
        _, v_idx, node_idx_in_route = loc
        # 处理边界情况：如果插入位置是路径末尾，则调整为倒数第二个位置（在仓库节点之前）
        if node_idx_in_route == len(vehicle_routes[v_idx]):
            node_idx_in_route = len(vehicle_routes[v_idx]) - 1
        # 在指定位置插入节点
        vehicle_routes[v_idx].insert(node_idx_in_route, node)
        
    elif loc_type == 'drone':
        # 无人机任务节点插入
        _, v_idx, s_idx, delivery_idx = loc
        # 在无人机任务的配送列表指定位置插入节点
        drone_routes[v_idx][s_idx][2].insert(delivery_idx, node)

def insert_two_nodes_safely(v_routes, d_routes, node_a, loc_a, node_b, loc_b):
    # 判断是否插入到同一辆车的路径中
    if loc_a[0] == 'vehicle' and loc_b[0] == 'vehicle' and loc_a[1] == loc_b[1]:
        pos_a = loc_a[2]
        pos_b = loc_b[2]
        if pos_a <= pos_b:
            # 先插入a，再插入b
            insert_node_at_loc(v_routes, d_routes, node_a, loc_a)
            insert_node_at_loc(v_routes, d_routes, node_b, loc_b)
        else:
            # 先插入b，再插入a
            insert_node_at_loc(v_routes, d_routes, node_b, loc_b)
            insert_node_at_loc(v_routes, d_routes, node_a, loc_a)

    # 判断是否插入到同一次无人机任务中
    elif loc_a[0] == 'drone' and loc_b[0] == 'drone' and loc_a[1] == loc_b[1] and loc_a[2] == loc_b[2]:
        pos_a = loc_a[3]
        pos_b = loc_b[3]
        if pos_a <= pos_b:
            insert_node_at_loc(v_routes, d_routes, node_a, loc_a)
            insert_node_at_loc(v_routes, d_routes, node_b, loc_b)
        else:
            insert_node_at_loc(v_routes, d_routes, node_b, loc_b)
            insert_node_at_loc(v_routes, d_routes, node_a, loc_a)

    else:
        # 两个节点插入不同路径或任务，顺序无关，按给定顺序插入
        insert_node_at_loc(v_routes, d_routes, node_a, loc_a)
        insert_node_at_loc(v_routes, d_routes, node_b, loc_b)

def is_sortie_feasible(sortie, vehicle_routes, model, node_positions=None):
    """Feasibility of one sortie: capacity, endpoint order and flight energy.

    The old maximum-flight-distance test is gone.  Whether a sortie can be flown
    is decided by the payload and the real flight energy against the battery of
    the physical drone that flies it (the cumulative battery state of the whole
    solution is checked by :func:`global_check` /
    :func:`evaluator.evaluate_solution`).
    """
    if not isinstance(sortie, (list, tuple)) or len(sortie) != 4:
        return False
    try:
        launch_node = int(sortie[1])
        delivery_nodes = [int(x) for x in sortie[2]]
        rendezvous_node = int(sortie[3])
    except (TypeError, ValueError):
        return False

    if not delivery_nodes or 0 in delivery_nodes:
        return False
    if len(set(delivery_nodes)) != len(delivery_nodes):
        return False
    try:
        total_payload = sum(model.customer_demands[c] for c in delivery_nodes)
    except (IndexError, TypeError):
        return False
    if total_payload > model.drone_payload_capacity + 1e-12:
        return False

    route = [int(x) for x in vehicle_routes]
    if node_positions is None:
        node_positions = defaultdict(list)
        for idx, node in enumerate(route):
            node_positions[node].append(idx)

    launch_positions = [0] if launch_node == 0 else node_positions.get(launch_node, [])
    rendez_positions = [len(route) - 1] if rendezvous_node == 0 else node_positions.get(rendezvous_node, [])
    if not launch_positions or not rendez_positions:
        return False
    if not any(lp < rp for lp in launch_positions for rp in rendez_positions):
        return False

    # Necessary condition only: a single sortie can never exceed the battery on
    # its own, and hover energy (>= 0) is added later by the full evaluator.
    try:
        flight_energy = sortie_energy_wh(
            launch_node, delivery_nodes, rendezvous_node,
            model.dro_dist_matrix, model.customer_demands,
            speed_m_s=model.drone_speed,
        )
    except (IndexError, TypeError):
        return False
    battery = float(getattr(model, 'drone_battery_capacity_wh', float('inf')))
    return flight_energy <= battery + 1e-9


def global_check(vehicle_routes, drone_routes, model, check_customer_coverage=True):
    """Validate a complete truck-drone solution (single source of truth).

    Delegates to :func:`evaluator.evaluate_solution`, which checks route format,
    customer coverage, payload capacity, launch-recovery ordering, per-drone
    temporal ordering, real flight energy, hover energy and the cumulative
    battery/SOC of every physical drone.  Range is deliberately *not* checked:
    the maximum-flight-distance constraint has been removed.

    This function is intentionally strict because accepting a malformed
    candidate can make the objective value artificially small and contaminate
    the ALNS search.
    """
    evaluation = evaluate_solution(
        vehicle_routes, drone_routes, model,
        check_coverage=check_customer_coverage,
    )
    return evaluation.feasible

def find_occurrence_between(node_positions, node_val, min_exclusive=None, max_exclusive=None, prefer='first'):
    """
    在 node_positions 中查找某节点的出现位置，要求严格在 (min_exclusive, max_exclusive) 范围内
    """
    positions = node_positions.get(int(node_val), [])
    if not positions:
        return None
    filtered = []
    for p in positions:
        if min_exclusive is not None and not (p > min_exclusive):
            continue
        if max_exclusive is not None and not (p < max_exclusive):
            continue
        filtered.append(p)
    if not filtered:
        return None
    return filtered[0] if prefer == 'first' else filtered[-1]

def get_all_customers_with_details(model, vehicle_routes, drone_routes):
    """
    辅助函数: 获取所有客户的详细信息，用于Shaw算子
    该函数遍历所有车辆路径和无人机任务，收集所有已服务客户的信息，
    包括客户节点编号、位置信息、所属车辆索引和客户需求量。
    """
    customer_pool = []
    # 遍历所有车辆路径，收集车辆服务的客户信息
    for v_idx, route in enumerate(vehicle_routes):
        # 从索引1开始到倒数第二个元素（排除起始和结束的仓库节点0）
        for node_idx, node in enumerate(route[1:-1], start=1):
            customer_pool.append({
                'node': node,                           # 客户节点编号
                'loc': ('vehicle', v_idx, node_idx),    # 位置信息：(类型, 车辆索引, 节点索引)
                'v_idx': v_idx,                         # 所属车辆索引
                'demand': model.customer_demands[node]  # 客户需求量
            })
    
    # 遍历所有无人机任务，收集无人机服务的客户信息
    for v_idx, sorties in enumerate(drone_routes):
        for s_idx, sortie in enumerate(sorties):
            # 遍历当前任务的所有配送节点
            for delivery_idx, node in enumerate(sortie[2]):
                customer_pool.append({
                    'node': node,                                    # 客户节点编号
                    'loc': ('drone', v_idx, s_idx, delivery_idx),    # 位置信息：(类型, 车辆索引, 任务索引, 配送索引)
                    'v_idx': v_idx,                                  # 关联到其所属车辆
                    'demand': model.customer_demands[node]      # 客户需求量
                })
    return customer_pool

def fast_copy_drone_routes(drone_routes):
    """Fast structural copy for the standard sortie format [label, launch, deliveries, rendezvous]."""
    copied = []
    for group in drone_routes:
        new_group = []
        for sortie in group:
            if len(sortie) != 4:
                # Fail loudly here instead of producing a silently corrupted copy.
                raise ValueError(f"Invalid sortie format: {sortie!r}")
            eid, launch, deliveries, rendezvous = sortie
            new_group.append([eid, launch, list(deliveries), rendezvous])
        copied.append(new_group)
    return copied





        

