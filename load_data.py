"""
数据加载模块 (data loading).

The benchmark JSON files are physicalised **once** by ``physicalize_datasets.py``:

    * every coordinate is already in metres (original coordinate * 100);
    * every ``demand`` is already in kg (per-instance scaling; the maximum depends on the dataset);
    * the speeds written in the file are the fixed 30 / 60 km/h in m/s.

Runtime code must therefore never multiply coordinates by 100 and never
normalise demands again.  This loader validates the ``physicalized`` flag so a
non-physicalised file fails loudly instead of silently producing a wrong model.
"""
import json
import os

from drone_physics import (
    DRONE_BATTERY_CAPACITY_WH,
    DRONE_PAYLOAD_CAPACITY_KG,
    DRONE_SPEED_M_S,
    TRUCK_SPEED_M_S,
)


def load_data_from_json(json_file_path, verbose=False, require_physicalized=True):
    """
    从JSON文件加载数据集
    返回包含所有参数的字典

    单位: 距离 m, 时间 s, 速度 m/s, 质量 kg, 功率 W, 能量 Wh.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        physicalized = bool(data.get('physicalized', False))
        if require_physicalized and not physicalized:
            raise ValueError(
                f"{json_file_path} 尚未物理化：请先运行 python physicalize_datasets.py"
            )

        # 约定：节点 0 为 Depot，locations 仅包含客户 1..N。
        # JSON 的 number_of_nodes 包括 Depot，应等于 len(locations) + 1。
        customer_count = len(data['locations'])
        expected_total_nodes = customer_count + 1
        declared_total_nodes = int(data['number_of_nodes'])
        if declared_total_nodes != expected_total_nodes:
            raise ValueError(
                f"number_of_nodes={declared_total_nodes}，但存在 "
                f"{customer_count} 个客户 + 1 个仓库；"
                f"正确总节点数应为 {expected_total_nodes}"
            )

        # Speed / capacity come from the fixed physical constants, not from the
        # JSON, so a stale file can never inject a non-physical value.
        params = {
            'num_vehicles': data['the_number_of_trucks'],
            'num_drones_per_vehicle': data['the_number_of_dronescarried_by_each_truck'],
            'vehicle_speed': TRUCK_SPEED_M_S,          # 8.333333333 m/s
            'drone_speed': DRONE_SPEED_M_S,            # 16.666666667 m/s
            'drone_payload_capacity': DRONE_PAYLOAD_CAPACITY_KG,
            'drone_battery_capacity_wh': DRONE_BATTERY_CAPACITY_WH,
            'vehicle_max_distance': data.get('the_maximum_distance_of_the_truck'),
            'num_customers': len(data['locations']),  # 客户数，不包括 Depot
            'number_of_nodes': int(data['number_of_nodes']),  # 总节点数，包括 Depot
            'depot_coords': [data['depot']['x'], data['depot']['y']],
            'customer_coords': [],
            'customer_demands': [],
            'physicalized': physicalized,
            'coordinate_unit': data.get('coordinate_unit'),
            'demand_unit': data.get('demand_unit'),
            'source_file': json_file_path,
        }

        # Legacy pass-through so that gurobi_opt.py keeps receiving the same key.
        # The ALNS model never uses it: the maximum drone flight distance has
        # been removed from the problem definition.
        params['drone_max_distance'] = data.get('the_maximum_distance_of_the_drone')

        # 提取客户坐标和需求 (already m / kg -- no conversion here)
        for location in data['locations']:
            params['customer_coords'].append([location['x'], location['y']])
            params['customer_demands'].append(location['demand'])

        if verbose:
            print(f"成功从 {json_file_path} 加载数据:")
            print(f"  - 车辆数量: {params['num_vehicles']}")
            print(f"  - 每车无人机数量: {params['num_drones_per_vehicle']}")
            print(f"  - 客户数量: {params['num_customers']}")
            print(f"  - 仓库坐标 (m): {params['depot_coords']}")
            print(f"  - 无人机最大载量 (kg): {params['drone_payload_capacity']}")
            print(f"  - 无人机电池 (Wh): {params['drone_battery_capacity_wh']}")
            print(f"  - 卡车速度 (m/s): {params['vehicle_speed']}")
            print(f"  - 无人机速度 (m/s): {params['drone_speed']}")
            print(f"  - 最大客户需求 (kg): {max(params['customer_demands'])}")
            print("  - 无人机最大飞行距离: 已移除 (range 不再是约束)")

        return params

    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        return None
    except KeyError as e:
        print(f"错误: JSON文件缺少必要的字段 - {e}")
        return None
    except ValueError as e:
        # A known, actionable condition -- most often "not physicalised yet".
        # It used to fall through to the generic handler below and be reported
        # as an "unknown error", which hid the actual fix.
        print(f"错误: {json_file_path} 数据不合法 - {e}")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生未知错误 - {e}")
        return None


def list_available_data_files(data_dir="data"):
    """
    列出可用的数据文件
    """
    if not os.path.exists(data_dir):
        return []

    json_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(data_dir, file))

    return sorted(json_files)


def build_model_kwargs(data_params):
    """Map a loaded dataset onto ``vrp_with_drones`` keyword arguments.

    Keeps every runner (main.py, scripts, smoke tests) consistent and makes sure
    the removed range parameter is never forwarded to the model.
    """
    count = len(data_params['customer_coords'])
    if count != len(data_params['customer_demands']):
        raise ValueError('customer_coords 和 customer_demands 长度不一致')
    if int(data_params['num_customers']) != count:
        raise ValueError('num_customers 与客户坐标数量不一致')
    if 'number_of_nodes' in data_params and int(data_params['number_of_nodes']) != count + 1:
        raise ValueError('number_of_nodes 必须是客户数 + 1（包含仓库）')
    return {
        'num_customers': count,
        'num_vehicles': data_params['num_vehicles'],
        'num_drones_per_vehicle': data_params['num_drones_per_vehicle'],
        'vehicle_speed': data_params['vehicle_speed'],
        'drone_speed': data_params['drone_speed'],
        'drone_payload_capacity': data_params['drone_payload_capacity'],
        'depot_coords': data_params['depot_coords'],
        'customer_coords': data_params['customer_coords'],
        'customer_demands': data_params['customer_demands'],
    }
