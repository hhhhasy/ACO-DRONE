# ACO+Tabu 消融实验系统（六组）

本目录是当前 ACO+Tabu 算法的独立消融实现。当前算法默认使用
**100% 策略 B**（双侧工作量感知服务构造），本实验研究：

1. 四层信息素分别关闭的单因素消融；
2. 取消策略 B 工作量偏向（令 κ=0）的消融。

> **新消融实验尚未正式运行。**
> 这里只有代码、测试、统计工具和一次每版本 2 秒的冒烟记录。
> 旧八版本设计的 30 秒结果已归档到 `results_archive/30s_8variant/`，
> 不属于本次新设计。

---

## 1. 六个版本

| 版本 | 服务策略 | κ | tau_edge | tau_assign | tau_service | tau_drone | 说明 |
|---|---|---:|---|---|---|---|---|
| `full` | 100% 策略 B | 1.0 | 开 | 开 | 开 | 开 | 当前 ACO 完整版，参照组 |
| `no_edge` | 100% 策略 B | 1.0 | **关** | 开 | 开 | 开 | 路径信息素消融 |
| `no_assign` | 100% 策略 B | 1.0 | 开 | **关** | 开 | 开 | 卡车归属信息素消融 |
| `no_service` | 100% 策略 B | 1.0 | 开 | 开 | **关** | 开 | 服务方式信息素消融 |
| `no_drone` | 100% 策略 B | 1.0 | 开 | 开 | 开 | **关** | 物理无人机归属信息素消融 |
| `no_workload_bias` | 100% 策略 B | **0.0** | 开 | 开 | 开 | 开 | 取消策略 B 的工作量双向偏好 |

所有版本：

- `service_policy_mode = "fixed"`，`adaptive_service_fraction = 1.0`；
- Tabu、解码缓存、ACS 参数、服务队列/能量启发式完全相同；
- 使用同一份精确物理评价器；
- 只有上表列出的一个因素不同。

旧的 `fixed0`、`fixed50`、`fixed100`、`dynamic_full` 四个策略/概率版本已从新设计中移除。

---

## 2. κ=0 消融的数学含义

策略 B 的方向性评分为：

\[
\eta_{\mathrm{truck}}=e^{-\kappa z_c/2},
\qquad
\eta_{\mathrm{drone}}=e^{+\kappa z_c/2},
\]

其中 \(z_c\) 是把客户交给无人机后，相对卡车服务对完成时间的预测改善量。
\(\kappa\) 对应代码中的 `service_proxy_strength`。

`no_workload_bias` 只把 \(\kappa\) 设为 0：

\[
\eta_{\mathrm{truck}}=\eta_{\mathrm{drone}}=1.
\]

此时服务方式仍由 ACS 在信息素上选择：

\[
S_{\mathrm{truck}}=(\tau_{c,0}^{\mathrm{service}})^{\alpha_s},
\qquad
S_{\mathrm{drone}}=(\tau_{c,1}^{\mathrm{service}})^{\alpha_s}.
\]

因此：

- 去掉了策略 B 最核心的负载感知双向偏好；
- 保留 `tau_service` 的信息素学习与引导；
- 服务方式决策、无人机配送、物理无人机选择全部保留；
- 不是把策略 B 删掉，也不是禁止无人机。

代码位置：

- `aco_solver.py::_service_bias_terms()`：κ=0 时直接返回 `(1.0, 1.0)`；
- `aco_solver.py::_construct_service_balanced()`：用返回的 `eta_truck/eta_drone` 与 `tau_service` 组成 ACS 分数。

---

## 3. 四层信息素如何关闭

统一由 `ACOConfig` 的四个布尔字段控制，默认全部为 `True`：

```python
pheromone_edge: bool = True
pheromone_assign: bool = True
pheromone_service: bool = True
pheromone_drone: bool = True
```

关闭某层后：

1. 候选评分中该层的乘性贡献由 `_tau_*_term()` 返回中性常数 `1.0`；
2. 跳过该层的局部更新、初始化强化、全局蒸发/强化/clip、重启平滑；
3. 决策变量全部保留：卡车路径、客户-卡车分配、服务方式、物理无人机选择；
4. 其他三层信息素、启发式、约束、随机搜索逻辑不变；
5. `pheromone_statistics()` 记录每层是否启用及均值/极值，不参与决策。

`no_service` 不会禁止无人机配送；`no_drone` 也不会。

---

## 4. 与当前正式 ACO 的一致性

`full` 对应当前正式 ACO 的默认设置：

```python
ACOConfig(
    service_policy_mode="fixed",
    adaptive_service_fraction=1.0,
    service_proxy_strength=1.0,
    ...  # 四层信息素全开、Tabu 默认开启
)
```

`tests/test_full_equivalence.py` 会把原项目 `aco_solver.py` 以别名加载，在同一个
合成小实例、同一种子/蚂蚁数/迭代数下与 `full` 对比，要求路径、服务方式、物理
无人机和 Makespan 逐位一致，并覆盖信息素重启分支。

---

## 5. 正式实验参数

| 参数 | 值 |
|---|---|
| 数据集 | `../大规模/` 下的 30 个物理化 JSON |
| 预算 | 10、20、30、50 秒（主论文 50 秒） |
| 蚂蚁数 | 30 |
| 种子 | 13, 21, 42, 50, 65, 77, 100, 123, 2024, 3407 |
| 版本 | 6 |
| 主实验规模 | 30 × 10 × 6 × **1 个预算(50s)** = **1800 次** |
| 全预算网格 | 30 × 10 × 6 × 4 = **7200 次** |

计时口径与 `aco_time_main.py` 一致：实例加载前开始计时，初始化时间计入预算，
剩余预算交给 `solve()`，每只蚂蚁前检查截止。

批量器只串行执行；不同并发度会改变 50 秒内完成的迭代数。

---

## 6. 环境

本机已验证：

```text
D:/Anaconda3/python.exe   # Python 3.9.12
numpy 1.22.3, scipy 1.8.0, matplotlib 3.6.2
```

建议固定 BLAS 线程：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

---

## 7. 单个任务

```bash
cd ACO_Ablation_Study

# 参照组 full
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
D:/Anaconda3/python.exe run_ablation.py ../大规模/A-n32-k5.json \
  --variant full --seed 13 --time-limit 50 --ants 30

# κ=0 消融
D:/Anaconda3/python.exe run_ablation.py ../大规模/A-n32-k5.json \
  --variant no_workload_bias --seed 13 --time-limit 50 --ants 30
```

输出：

```text
results/<variant>/<budget>s/<dataset>/seed<seed>/
├── run.json       # 完整配置、指标、方案、信息素状态、服务策略计数
├── solution.txt
└── history.csv
```

`run.json` 会记录 `service_proxy_strength`、`service_policy` 的实际调用次数
（`adaptive_ants` / `legacy_ants`）、概率均值和 `pheromone.enabled`。

同一组合已有完整结果时默认拒绝覆盖；只有显式 `--overwrite` 才会替换。

---

## 8. 正式批量运行

### 8.1 先看计划

```bash
D:/Anaconda3/python.exe run_all_ablations.py --dry-run
```

应显示 `total jobs: 1800`；不创建结果、不解析数据集、不求解。

### 8.2 主实验（50 秒，1800 次）

```bash
cd ACO_Ablation_Study
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
D:/Anaconda3/python.exe run_all_ablations.py --confirm RUN-FORMAL-ABLATION
```

默认版本就是 `full no_edge no_assign no_service no_drone no_workload_bias`，
预算 50 秒，10 个种子。纯求解时间约 25 小时（1800 × 50 秒），另加初始化/写盘
开销。

### 8.3 全预算网格（7200 次）

```bash
D:/Anaconda3/python.exe run_all_ablations.py \
  --budgets 10 20 30 50 --confirm RUN-FORMAL-ABLATION
```

### 8.4 断点续跑与一致性检查

重新执行同一命令即可续跑。完整结果会跳过；缺失/不完整会重跑；配置、代码或
数据集指纹不一致会停止并报告冲突。

```bash
# 只审计，不求解
D:/Anaconda3/python.exe run_all_ablations.py --check-only
```

可选：

- `--on-conflict skip|rerun`
- `--limit-runs N`
- `--limit-datasets N`

批量清单写入 `results/_batch_manifest.json`。

---

## 9. 统计分析

主实验完成后：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
D:/Anaconda3/python.exe analyze_ablation.py \
  --budget 50 --data-dir ../大规模
```

输出到 `results/_analysis/50s/`：

- 每个版本每个数据集：种子均值、最好/最差、样本标准差、CV；
- 等权平均 Makespan、冠军数、并列冠军数；
- 5 个相对 `full` 的配对比较（4 个信息素消融 + κ=0）；
- 配对双侧 Wilcoxon + Holm 校正（m=5）；
- 改善/退步/持平数据集数；
- 中文报告和图表。

统计口径：

- **配对单位 = 一个数据集在 10 个种子上的平均 Makespan**，不是单个种子；
- 只有全部版本、全部种子都完成且可行的数据集才进入配对检验；
- 缺失数据只报告、不插补、不编造；
- 代码指纹混用时标记 `MIXED_CODE` 并拒绝推断。

---

## 10. 测试

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
D:/Anaconda3/python.exe -m unittest discover -s tests -v
```

覆盖：

- 六版本注册与严格单因素性；
- `full` 与原项目当前 ACO 逐位等价（含强制重启分支）；
- 四层信息素关闭后评分恒为 1.0、更新被跳过、其他层正常；
- `no_workload_bias` 的 `eta_truck=eta_drone=1`；
- κ=0 时服务方式仍由 `tau_service` + ACS 决定；
- 客户不缺失/不重复；
- 输出路径隔离、断点续跑、`--dry-run` 不创建结果；
- 六版本分析器的配对检验与 Holm 校正。

本次交付已通过 **58 项单元测试**，并执行了一次每版本 2 秒的冒烟：

| 版本 | Makespan (s) | 可行 | 迭代 |
|---|---:|---|---:|
| full | 577.6335 | 是 | 32 |
| no_edge | 580.5618 | 是 | 30 |
| no_assign | 581.8418 | 是 | 36 |
| no_service | 581.8418 | 是 | 40 |
| no_drone | 560.1359 | 是 | 34 |
| no_workload_bias | 577.6335 | 是 | 34 |

冒烟只验证链路，不是性能证据。

---

## 11. 目录与归档

```text
ACO_Ablation_Study/
├── aco_solver.py, evaluator.py, drone_physics.py, fast_window_physics.py
├── ablation_config.py, run_state.py
├── run_ablation.py, run_all_ablations.py, analyze_ablation.py
├── tests/
├── results/                    # 新六版本结果（目前只有 _smoke/）
├── results_archive/
│   └── 30s_8variant/           # 旧八版本 30 秒结果（1200 次，已归档）
├── README.md
└── VARIANTS.md
```

旧归档包含 `fixed0/fixed50/fixed100/dynamic_full` 的 30 秒结果及其清单/压缩包，
仅作历史参考，**不参与新设计的批量续跑和统计分析**。新分析器只读取 `results/`
下当前六个版本的目录。

---

## 12. 声明

**新消融正式实验未运行。**

本次只完成了代码重构、58 项单元测试、1800 任务 dry-run 检查和一次六版本
2 秒冒烟。所有 10/20/30/50 秒正式实验由你本人启动。
