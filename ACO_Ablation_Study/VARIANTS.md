# 消融版本说明（实现级）

本文件说明新六版本设计在代码中的确切差异，便于论文写作和审稿复核。

## 1. 配置差异

唯一定义在 `ablation_config.py::VARIANTS`。所有版本共享：

```python
service_policy_mode = "fixed"
adaptive_service_fraction = 1.0     # 100% strategy B
```

| 版本 | `service_proxy_strength` (κ) | `pheromone_edge` | `pheromone_assign` | `pheromone_service` | `pheromone_drone` |
|---|---:|---|---|---|---|
| `full` | **1.0** | True | True | True | True |
| `no_edge` | 1.0 | **False** | True | True | True |
| `no_assign` | 1.0 | True | **False** | True | True |
| `no_service` | 1.0 | True | True | **False** | True |
| `no_drone` | 1.0 | True | True | True | **False** |
| `no_workload_bias` | **0.0** | True | True | True | True |

单因素性：

- 四个 `no_*` 与 `full` 的唯一差异是表中对应布尔值；
- `no_workload_bias` 与 `full` 的唯一差异是 `service_proxy_strength`（1.0 → 0.0）；
- Tabu、缓存、ACS 参数、服务队列/能量启发式、物理评价器完全相同。

`validate_variant_isolation()` 在批量启动前强制验证以上全部条件。

## 2. κ=0 的实现

策略 B 的方向性项：

\[
\eta_{\mathrm{truck}}=e^{-\kappa z_c/2},\qquad
\eta_{\mathrm{drone}}=e^{+\kappa z_c/2}.
\]

代码：

```python
def _service_bias_terms(self, signed):
    strength = float(self.cfg.service_proxy_strength)
    if strength == 0.0:
        return 1.0, 1.0                    # kappa = 0
    log_ratio = strength * float(signed)
    return math.exp(-0.5 * log_ratio), math.exp(0.5 * log_ratio)
```

当 `no_workload_bias` 时：

```python
eta_truck, eta_drone = 1.0, 1.0
scores = [
    tau_service[c, 0] ** service_alpha * 1.0 ** service_beta,
    tau_service[c, 1] ** service_alpha * 1.0 ** service_beta,
]
mode = self._acs_pick([0, 1], scores)
```

因此服务方式仍由 `tau_service` 的信息素值 + ACS 概率规则选择，
局部/全局信息素更新照常进行；只是去掉了“预计改善完成时间”的双向偏好。

## 3. 四层信息素的全部读写位置

| 阶段 | tau_edge | tau_assign | tau_service | tau_drone |
|---|---|---|---|---|
| 初始化 `__init__` | 分配 | 分配 | 分配 | 分配 |
| 预热 `_warm_start_pheromone` | 条件写入 | 条件写入 | 条件写入 | 条件写入 |
| 主路径评分 `_construct_master_routes` | `_tau_edge_term` | `_tau_assign_term` | — | — |
| 服务评分 `_construct_service_balanced` | — | — | `_tau_service_term` | `_tau_drone_term` |
| 旧服务策略（当前默认不走） | — | — | `_tau_service_term` | `_tau_drone_term` |
| 局部更新 `_local_update_*` / 内联 | 条件跳过 | 条件跳过 | 条件跳过 | 条件跳过 |
| 重启 `_pheromone_restart` | 条件平滑 | 条件平滑 | 条件平滑 | 条件平滑 |
| 全局更新 `_global_update` | 蒸发/强化/clip 条件执行 | 同左 | 同左 | 同左 |

“条件”均指 `ACOConfig.pheromone_*` 为 True 时执行。关闭某层后：

- 该项在候选评分中严格等于 `1.0`（乘性中性），不是 `0.0`；
- 该层不再被任何代码路径读取或写入；
- 其他层、启发式、约束、随机数逻辑和决策变量不变。

## 4. 为什么关闭信息素不是删除决策

- `tau_edge` 只影响“下一个访问哪个客户”的评分权重；客户仍必须被访问。
- `tau_assign` 只影响“这笔货归哪辆卡车”的评分权重；客户仍必须归属一辆卡车。
- `tau_service` 只影响“卡车还是无人机配送”的评分权重；两种服务方式仍可选。
- `tau_drone` 只影响“由该卡车哪架物理无人机执行”的评分权重；无人机配送与
  物理无人机选择仍保留。

`no_service` 不会把无人机配送关掉；`no_drone` 也不会。

## 5. 已移除的旧版本

新设计不再包含：

- `fixed0`、`fixed50`、`fixed100`：服务策略概率消融；
- `dynamic_full`：旧 workload 动态控制器版本。

当前算法本身是 100% 策略 B，因此参照组命名为 `full`。
旧八版本的 30 秒结果已归档到 `results_archive/30s_8variant/`，不参与新分析。

## 6. 未加入的内容

没有协同完成时间代理、没有额外局部搜索、没有新的邻域或修复算子。
物理可行性仍只由 `evaluator.evaluate_solution` 和 `drone_physics` 决定。
