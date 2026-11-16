# 五算法收敛性实验

> 本机若 `python` 不是 Anaconda 解释器，请将命令中的 `python` 替换为
> `D:/Anaconda3/python.exe`。

本目录是独立实验项目，比较 ACO、GA、ALNS、PSO、VND 在真实墙钟时间下的
历史最优可行解收敛行为。

- 数据集：5 个固定算例
  - `P-n40-k5`（39 客户）
  - `B-n57-k7`（56 客户）
  - `A-n69-k9`（68 客户）
  - `E-n101-k8`（100 客户）
  - `M-n200-k16`（199 客户）
  - `A-n80-k10`（79 客户）
  - `B-n63-k10`（62 客户）
- 算法：ACO、GA、ALNS、PSO、VND
- 随机种子：`13, 21, 42, 50, 65, 77, 100, 123, 2024, 3407`
- 每次正式运行：**100 秒**（包含数据加载、初始化和搜索；350 次串行约 9.72 小时）
- 物理参数：**200 Wh / 3 kg**
- 正式任务：7 × 5 × 10 = **350 次**（原 5 个数据集 250 次已完成；新增 2 个数据集需补 100 次）
- 正式结果：`results/`；收敛历史：`history/`；
  分析：`analysis/`；图片：`figures/`

当前 ACO 固定使用：

```json
{
  "service_policy_mode": "fixed",
  "adaptive_service_fraction": 1.0,
  "service_proxy_strength": 1.0
}
```

即 100% 策略 B、完整工作量感知偏向（κ=1）。这三个字段同时写在
`config/convergence_config.json` 和适配器里，避免后续默认值变化悄悄改变实验口径。

旧 ACO 仍为自适应策略时产生的 2026-10-27 结果已归档到
`archive_old_adaptive_20261027/`；含 VNS 的最近一批结果已归档到
`archive_vns_20261115/`。当前 `results/`、`history/`、`analysis/`、`figures/`
为空，等待使用 PSO 新算法集重新运行；正式任务仍为 7 × 5 × 10 = 350 次。

---

## 1. 安全机制

`run_convergence.py` 默认只预览 350 个任务，不创建求解器：

```bash
python run_convergence.py --dry-run
```

只有显式传入 `--execute` 才会启动正式求解；重新执行会跳过已完成组合。

单独跑一个 ACO 组合：

```bash
python run_convergence.py --algorithms ACO \
  --datasets P-n40-k5 --seeds 13 --execute
```

---

## 2. 收敛历史记录

每个可行解事件都会写入独立的 `history/<run_id>.csv`：

```text
algorithm
dataset
seed
time_budget_seconds
elapsed_time_seconds
best_makespan_s
iteration_or_generation
evaluations
event_type
```

`event_type` 为 `initial`、`improvement` 或 `final`。

实现原则：

- 使用 `time.perf_counter()` 记录真实墙钟时间；时间起点在数据加载前，
  因此曲线包含初始化和搜索时间。
- 只有经过统一评价器确认可行的有限 Makespan 才会进入历史。
- 历史最优值单调不增；非改善/不可行事件被记录器拒绝。
- 时间戳单调不减；不按迭代次数换算时间。
- 不使用不同预算的终值拼接收敛曲线；不进行线性插值。
- 首次可行解之前保持缺失（曲线为 NaN），分析时前向保持。
- 每次运行的 history 文件独立命名，不互相覆盖。

### 五个算法的接入方式

- ACO：`AntColonySolver` 可选 `history_callback`，在首次/改善时回调；
  适配器记录初始解并在最终评价后写入 final。
- GA：`ClassicGA` 可选 `history_callback`；初始种群最佳与每次严格改善回调。
- ALNS：`vrp_with_drones.local_search` 新增可选 `history_callback`；
  初始解由适配器记录，搜索中每次最佳改善回调。
- PSO：`ParticleSwarmSolver` 可选 `history_callback`；记录 PSO 自身的初始解、每次全局最优改善和最终解。
- VND：可选 `history_callback`；初始解与每次最佳改善回调。

### 对原有代码的最小改动

没有修改任何优化算子、目标函数或超参数。仅新增参数传递/日志接口：

1. `model.py`
   - `vrp_with_drones.__init__` 新增可选 `drone_battery_capacity_wh`。
   - `local_search` 新增可选 `history_callback=None`，并记录候选评价数。
2. `aco_solver.py`
   - 可选 `history_callback`；`constructed_ants` 计数器（仅记录）。
3. `ga_solver.py`
   - 可选 `history_callback`；`evaluation_count`、`initial_best`（仅记录）。
4. `vnd_solver.py`
   - 可选 `history_callback` 和首次/改善回调（仅记录）。
5. `pso_solver.py`
   - 可选 `history_callback`（仅记录首次/改善/最终事件）。

本机适配器在导入 `model`/sklearn 前会 neutralize `threadpoolctl` 的库扫描
（该机器的 `openblas_get_config()` 返回 NULL 会导致 sklearn 崩溃）。这只是运行
环境兼容处理，不改变算法逻辑；BLAS 线程固定为 1。

`drone_physics.py`、`evaluator.py`、`fast_window_physics.py`、`operators.py`
均未修改。ACO 使用当前根版本（fixed100/κ=1）。

---

## 3. 目录结构

```text
convergence_experiment/
├── config/convergence_config.json
├── adapters/
│   ├── __init__.py
│   ├── algorithms.py          # 五算法薄适配器 + 统一评价摘要
│   └── history.py             # 真实时间/单调最优历史记录器
├── run_convergence.py         # 默认仅预览；--execute 才运行
├── analyze_convergence.py     # 对齐/统计/报告/LaTeX（不调用求解器）
├── plot_convergence.py        # 只读分析 CSV 重新绘图
├── tests/test_convergence.py
├── results/                   # 新结果（当前为空）
├── history/                   # 新收敛历史（当前为空）
├── analysis/                  # 新分析（当前为空）
├── figures/                   # 新图片（当前为空）
├── archive_vns_20261115/       # 最近一批含 VNS 的结果（归档，不参与新分析）
├── archive_old_adaptive_20261027/
│   ├── results/               # 旧自适应 ACO 的 250 次结果
│   ├── history/
│   ├── analysis/
│   └── figures/
├── README.md
└── requirements.txt
```

旧归档只作历史参考，新分析器默认只读取当前的 `results/` 与 `history/`。

---

## 4. 分析

```bash
python analyze_convergence.py \
  --results-dir results \
  --history-dir history \
  --output-dir analysis \
  --figures-dir figures
```

分析只读取结果 CSV/JSON 和 history，不调用求解器。

---

## 5. 测试

```bash
python -m unittest discover -s tests -v
```

测试包括：

- 350 任务矩阵与配置检查；
- 当前 ACO 固定为 `fixed/1.0/κ=1`；
- history 单调性、首次可行、前向保持；
- 合成数据上的完整分析与绘图；
- 1 秒级五算法适配器冒烟（仅验证接口，不是正式实验）。

旧 VNS 结果已归档；新 PSO 算法集的 350 次正式运行尚未执行。测试不会重跑正式实验。
