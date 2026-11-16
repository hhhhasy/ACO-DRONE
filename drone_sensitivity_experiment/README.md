# 无人机资源灵敏度实验

> 本机若 `python` 不是 Anaconda 解释器，请将命令中的 `python` 替换为
> `D:/Anaconda3/python.exe`。

本目录是独立实验项目，研究无人机电池容量与最大载重对五种算法
（ACO、GA、ALNS、PSO、VND）的影响。

- 数据集：`../大规模/M-n151-k12.json`（150 客户）
- 电池容量：`100, 125, 150, 175, 200, 225, 250 Wh`（固定载重 3 kg）
- 最大载重：`1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 kg`（固定电池 200 Wh）
- 两维共享基准：`200 Wh + 3 kg`
- 共 **15 个唯一资源组合**（7 个电池水平 + 9 个载重水平 − 1 个共享基准）
- 5 算法 × 15 组合 × 10 种子 = **750 个正式任务**
- 每次正式运行 **50 秒**（包含数据加载与初始化；750 次串行约 10.4 小时）
- 结果：`results/`；分析：`analysis_M-n151-k12/`；
  图片：`figures_M-n151-k12/`（由配置决定）

当前 ACO 固定使用：

```json
{
  "service_policy_mode": "fixed",
  "adaptive_service_fraction": 1.0,
  "service_proxy_strength": 1.0
}
```

即 100% 策略 B、完整工作量感知偏向（κ=1）。这三个字段同时写在
`config/sensitivity_config.json` 和适配器里，避免后续默认值变化悄悄改变实验口径。

旧的含 VNS 的结果已归档到 `archive_vns_20261115/`；当前 `results/` 为空，首次使用新算法集运行才会生成新结果。

---

## 1. 安全机制

`run_sensitivity.py` 默认只预览任务矩阵，不创建求解器：

```bash
python run_sensitivity.py --dry-run
```

只有显式传入 `--execute` 才会启动正式求解。`--dry-run` 与 `--execute` 互斥。
750 个正式任务必须由用户自行启动。

单独跑一个 ACO 组合：

```bash
python run_sensitivity.py --algorithms ACO --seeds 13 \
  --battery-levels 200 --payload-levels 3 --execute
```

---

## 2. 物理参数是否真正生效

五种算法和统一评价器使用同一组对象属性/构造参数：

| 算法 | 参数传递位置 |
|---|---|
| ACO | `ProblemData.drone_battery_capacity_wh` / `drone_payload_capacity` |
| GA | `ProblemData.drone_battery_capacity_wh` / `drone_payload_capacity` |
| ALNS | `vrp_with_drones(..., drone_battery_capacity_wh=...)` / payload 参数 |
| PSO | `pso_decoder.load_problem` 的 `ProblemData`；覆盖 `drone_battery_capacity_wh` / `drone_payload_capacity` |
| VND | 复用 ALNS `vrp_with_drones` 模型 |

- 任务构造、无人机预检查、解码器、`drone_physics` 能耗计算和
  `evaluator.evaluate_solution` 都读取这些参数。
- 电池容量改变时，电池质量仍使用原模型的固定物理质量，没有改成与 Wh 成比例。
- 无人机仍然不可充电；累计电池约束仍由评价器按每架物理无人机检查。
- 不同参数组合每次都重新加载数据/构建模型，不复用旧参数的对象或缓存。
- 所有结果最终都经过 `evaluator.evaluate_solution` 复验。

### 对原有代码的最小改动

为了让灵敏度参数真正进入物理模型，只增加了参数传递/日志接口；
没有修改任何优化机制、目标函数或超参数：

1. `model.py`
   - `vrp_with_drones.__init__` 新增可选参数 `drone_battery_capacity_wh`。
   - `local_search` 新增可选 `history_callback=None` 与候选评价计数器。
2. `aco_solver.py`
   - 增加可选 `history_callback` 属性和 `constructed_ants` 计数器（仅记录）。
3. `ga_solver.py`
   - 增加可选 `history_callback`、`evaluation_count`、`initial_best`（仅记录）。
4. `vnd_solver.py`
   - 增加可选 `history_callback` 和首次/改善事件回调（仅记录）。
5. `pso_solver.py`
   - 增加可选 `history_callback`（仅记录首次/改善/最终事件）。

本机适配器在导入 `model`/sklearn 前会 neutralize `threadpoolctl` 的库扫描
（该机器的 `openblas_get_config()` 返回 NULL 会导致 sklearn 崩溃）。这只是运行
环境兼容处理，不改变算法逻辑；BLAS 线程固定为 1。

`drone_physics.py`、`evaluator.py`、`fast_window_physics.py`、`operators.py`
以及所有算法超参数均未修改。ACO 使用当前根版本（fixed100/κ=1）；PSO 使用独立的 `pso_decoder.py` 和平衡插入初始解，不依赖 ACO 解码器。

---

## 3. 目录结构

```text
drone_sensitivity_experiment/
├── config/sensitivity_config.json
├── adapters/
│   ├── __init__.py
│   └── algorithms.py          # 五算法薄适配器 + 统一评价摘要
├── run_sensitivity.py         # 默认仅预览；--execute 才运行
├── analyze_sensitivity.py     # 统计 + 报告 + LaTeX（不调用求解器）
├── plot_sensitivity.py        # 只读分析 CSV 重新绘图
├── tests/test_sensitivity.py  # 配置/任务矩阵测试（不求解）
├── results/                   # 新正式结果（当前为空，运行后生成）
├── analysis_M-n151-k12/       # 分析输出（运行后生成）
├── figures_M-n151-k12/        # 图片（运行后生成）
├── archive_vns_20261115/      # 旧 VNS 时代结果（归档，不参与新分析）
├── README.md
└── requirements.txt
```

每个正式运行写出：

```text
results/runs/<run_id>.json
results/runs.csv
results/experiment_used.json
```

`run_id` 形如 `ACO__M-n151-k12__b200__p3__s13`。

---

## 4. 分析

```bash
python analyze_sensitivity.py \
  --input-dir results \
  --output-dir analysis_M-n151-k12 \
  --figures-dir figures_M-n151-k12
```

分析只读取结果 CSV/JSON，不调用求解器。

---

## 5. 测试

```bash
python -m unittest discover -s tests -v
```

测试只验证配置和任务矩阵：

- 15 个唯一资源组合；
- 750 个任务；
- 当前 ACO 固定为 `fixed/1.0/κ=1`；
- ACO 任务参数与配置一致。

不运行 50 秒正式实验。
