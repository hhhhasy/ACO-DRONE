# ACO-DRONE: Multi-truck / Multi-drone Delivery Comparison

Comparison-experiment code for cooperative truck–drone delivery under an
identical wall-clock budget. The repository contains five independent solvers
and the shared physical evaluation layer:

- **ACO+Tabu** — Ant Colony System with tabu memory, exact-decoding cache and
  the dual-sided workload-aware service construction (strategy B).
- **GA** — literature-informed genetic algorithm.
- **ALNS** — adaptive large neighbourhood search.
- **VND** — variable neighbourhood descent using the ALNS neighbourhoods.
- **PSO** — independent random-key particle swarm with its own balanced-insertion decoder.

All algorithms use the same truck-drone physics, payload/battery limits,
synchronisation rules and exact `evaluator.evaluate_solution()` objective.

## Layout

| File | Role |
|---|---|
| `aco_solver.py` | ACO+Tabu solver and exact decoder |
| `aco_time_main.py` | ACO equal-wall-clock runner |
| `aco_iter_main.py` | ACO iteration-limited runner |
| `ga_solver.py`, `ga_time_main.py` | GA solver / runner |
| `alns_time_main.py` | ALNS runner (uses `model.py` / `operators.py`) |
| `vnd_solver.py`, `vnd_time_main.py` | VND solver / runner |
| `pso_solver.py`, `pso_main.py`, `pso_time_main.py`, `pso_decoder.py` | PSO solver / runner / decoder |
| `model.py`, `operators.py`, `utils.py` | ALNS model, neighbourhoods, shared utilities |
| `load_data.py` | benchmark JSON loader / model kwargs |
| `evaluator.py` | exact truck-drone feasibility and makespan evaluator |
| `drone_physics.py`, `fast_window_physics.py` | drone energy / sortie physics |
| `run_five_time_budgets.py` | batch runner for equal-time comparisons |
| `sweep_clustering.py` | angular sweep initial customer partition |
| `大规模/` | 30 physicalized benchmark instances |

## Requirements

Python 3.9+ and:

```bash
pip install -r requirements.txt
```

On Windows, pinning BLAS threads is recommended for reproducible floating-point
behaviour:

```bat
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
```

## Data format

`大规模/*.json` are already **physicalized**:

- coordinates in metres;
- demands in kilograms;
- depot is node 0; `locations` is the authoritative customer list;
- truck speed 30 km/h, drone speed 60 km/h;
- drone payload capacity 3 kg, battery 200 Wh.

The ACO/GA loaders accept the two `number_of_nodes` conventions found in the
benchmark set (customer count or total node count). `load_data.py` is stricter
and requires `number_of_nodes == len(locations) + 1`.

## Quick start

ACO+Tabu:

```bash
python aco_time_main.py 大规模/A-n32-k5.json --time-limit 50 --ants 30 --seed 13
```

GA:

```bash
python ga_time_main.py 大规模/A-n32-k5.json --time-limit 50 --population 30 --seed 13
```

ALNS:

```bash
python alns_time_main.py 大规模/A-n32-k5.json --time-limit 50 --seed 13
```

VND:

```bash
python vnd_time_main.py 大规模/A-n32-k5.json --time-limit 50 --seed 13
```

PSO:

```bash
python pso_time_main.py 大规模/A-n32-k5.json --time-limit 50 --swarm 30 --seed 13
```

Batch comparison:

```bash
python run_five_time_budgets.py \
  --data-dir 大规模 \
  --times 10 20 30 50 \
  --seeds 13 21 42 50 65 77 100 123 2024 3407 \
  --output-dir results
```

`run_five_time_budgets.py` supports `ACO, GA, ALNS, PSO, VND`; its default
selection is `VND, PSO`. Pass `--algorithms ACO GA ALNS PSO VND` to run all five.

## ACO service-policy default

The ACO solver defaults to **100% strategy B** (the dual-sided workload-aware
service construction), matching the `fixed100` ablation setting:

```python
ACOConfig(service_policy_mode="fixed", adaptive_service_fraction=1.0)
```

Other settings are available through the public `ACOConfig` without changing
any runner:

```python
# original dynamic workload controller (base probability 0.50)
ACOConfig(service_policy_mode="workload", adaptive_service_fraction=0.5)

# 100% legacy service heuristic
ACOConfig(service_policy_mode="fixed", adaptive_service_fraction=0.0)
```

Historical adaptive-mode results are therefore not directly comparable with the
current default; set the mode explicitly when reproducing older runs.

## Experiment projects

Two self-contained experiment projects are included as code only:

- `convergence_experiment/` — seven-instance, five-algorithm convergence
  histories and analysis;
- `drone_sensitivity_experiment/` — battery/payload sensitivity experiment.

The ACO pheromone ablation code is in `ACO_Ablation_Study/`. Its results and
archives, as well as all result/analysis/figure directories of the experiment
projects, are intentionally excluded from this repository.

## Outputs

Runners write solution text, JSON summaries and convergence history under the
chosen `--output-dir` (defaults are runner-specific). Generated results,
archives and caches are not part of this repository.
