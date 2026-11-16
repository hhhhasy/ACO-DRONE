"""CLI-compatible PSO runner. Output: PSO/<budget>s/seed<seed>/summary_*.json."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import pso_decoder
import drone_physics
import evaluator
import fast_window_physics
import pso_solver
from pso_solver import PSOConfig, ParticleSwarmSolver, load_problem
from pso_decoder import save_solution_txt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Random-key PSO, total wall-clock budget')
    ap.add_argument('data')
    ap.add_argument('--time-limit',type=float,default=0.0,help='positive: wall-clock mode; 0: iteration mode')
    ap.add_argument('--iterations',type=int,default=500,help='used only when time-limit is 0')
    ap.add_argument('--seed',type=int,default=13)
    ap.add_argument('--swarm',type=int,default=30)
    ap.add_argument('--cache-size',type=int,default=1024)
    ap.add_argument('--output-dir',default='pso_time_results')
    ap.add_argument('--quiet',action='store_true')
    args=ap.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit<0:
        ap.error('time-limit must be finite and nonnegative')
    if args.time_limit == 0 and args.iterations <= 0:
        ap.error('iterations must be positive in iteration mode')
    if args.swarm < 2 or args.cache_size < 0:
        ap.error('swarm must be >= 2 and cache-size must be >= 0')
    started=time.perf_counter()
    p=load_problem(args.data)
    cfg=PSOConfig(swarm_size=args.swarm,iterations=0 if args.time_limit > 0 else args.iterations,seed=args.seed,cache_size=args.cache_size,
                  time_limit_seconds=(max(1e-9,args.time_limit-(time.perf_counter()-started)) if args.time_limit > 0 else 0.0))
    solver=ParticleSwarmSolver(p,cfg)
    # Include construction/configuration overhead in the remaining solver budget.
    cfg.time_limit_seconds=(max(1e-9,args.time_limit-(time.perf_counter()-started)) if args.time_limit > 0 else 0.0)
    best=solver.solve()
    actual=time.perf_counter()-started
    run_label = (('%g'%args.time_limit).replace('.','p')+'s') if args.time_limit > 0 else ('%diter'%args.iterations)
    folder=Path(args.output_dir)/'PSO'/run_label/('seed%d'%args.seed)
    folder.mkdir(parents=True,exist_ok=True)
    base=Path(args.data).stem
    save_solution_txt(str(folder/('solution_'+base+'.txt')),p,best)
    initial={'vehicle_routes':solver.initial.vehicle_routes,'drone_routes':solver.initial.drone_routes,
             'makespan_s':solver.initial.objective}
    initial_sig=hashlib.sha256(json.dumps([initial['vehicle_routes'],initial['drone_routes']],
                                separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    initial['signature_sha256']=initial_sig
    (folder/('initial_'+base+'.json')).write_text(json.dumps(initial,indent=2),encoding='utf-8')
    payload=dict(algorithm='PSO',algorithm_variant='independent-random-key-greedy-insertion-tabu20-v3',
        tabu_size=pso_solver._TABU_SIZE,tabu_stats=solver.tabu_stats,
        stopping_mode='time' if args.time_limit > 0 else 'iterations',
        dataset=base,seed=args.seed,time_budget_seconds=args.time_limit,
        actual_algorithm_runtime_seconds=actual,solver_runtime_seconds=solver.runtime_seconds,
        initialization_runtime_seconds=solver.initialization_runtime_seconds,
        initialization_method='pso_balanced_insertion_drone_hints',initial_signature_sha256=initial_sig,
        initial_makespan_s=solver.initial.objective,initial_population_best_s=solver.initial_population_best,
        best_makespan_s=best.objective,feasible=best.feasible,
        iterations_used=solver.iterations_used,iterations_completed=solver.iterations_completed,
        evaluations=solver.evaluations,proposals=solver.proposals,cache_hits=solver.cache_hits,
        evaluations_scope='PSO candidate decodes only; each decode may perform multiple physical evaluations; excludes initial seed and final validation',
        swarm_size_used=solver.swarm_size_used,terminated_by_time_limit=solver.time_limit_hit,
        config=asdict(cfg),data_sha256=digest(args.data),
        source_sha256={Path(m.__file__).name:digest(m.__file__) for m in
          (pso_solver,pso_decoder,drone_physics,evaluator,fast_window_physics)},history=solver.history)
    payload['source_sha256'][Path(__file__).name] = digest(__file__)
    out=folder/('summary_'+base+'.json')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PSO budget=%.3fs actual=%.3fs initial=%.4f best=%.4f iterations=%d'%
          (args.time_limit,actual,solver.initial.objective,best.objective,solver.iterations_completed))
    print(out)

if __name__=='__main__':main()
