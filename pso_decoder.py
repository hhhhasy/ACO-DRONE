"""Independent route-first/greedy-sortie-insertion decoder for random-key PSO.

This is a project-specific heuristic, not a reproduction of a paper's solver.
Only the common physical model/evaluator are shared with other algorithms.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import numpy as np
from drone_physics import (TRUCK_SPEED_M_S, DRONE_SPEED_M_S,
    DRONE_PAYLOAD_CAPACITY_KG, DRONE_BATTERY_CAPACITY_WH)
from fast_window_physics import prepare_fixed_order_flight
from evaluator import evaluate_solution

WINDOW_CANDIDATES = 8

@dataclass
class ProblemData:
    num_customers: int
    num_vehicles: int
    num_drones_per_vehicle: int
    vehicle_speed: float
    drone_speed: float
    drone_payload_capacity: float
    drone_battery_capacity_wh: float
    depot_coords: tuple
    customer_coords: list
    customer_demands: list
    veh_dist_matrix: np.ndarray
    dro_dist_matrix: np.ndarray
    source_file: str = ''

@dataclass
class ParticleCode:
    master_routes: list
    service: list
    drone: list

@dataclass
class PSOSolution:
    code: ParticleCode
    vehicle_routes: list
    drone_routes: list
    evaluation: object

    @property
    def feasible(self): return self.evaluation.feasible
    @property
    def objective(self): return self.evaluation.objective
    def clone(self): return copy.deepcopy(self)


def load_problem(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    if not raw.get('physicalized',False):
        raise ValueError('dataset must already be physicalized (m/kg)')
    locs=raw['locations']; n=len(locs)
    if int(raw['number_of_nodes']) not in (n,n+1):
        raise ValueError('number_of_nodes does not match locations')
    depot=(float(raw['depot']['x']),float(raw['depot']['y']))
    coords=[(float(c['x']),float(c['y'])) for c in locs]
    a=np.asarray([depot]+coords); delta=a[:,None,:]-a[None,:,:]
    demands=[0.]+[float(c['demand']) for c in locs]
    if not np.isfinite(a).all() or not all(math.isfinite(q) and q>=0 for q in demands):
        raise ValueError('coordinates/demands must be finite; demands nonnegative')
    return ProblemData(n,int(raw['the_number_of_trucks']),
        int(raw['the_number_of_dronescarried_by_each_truck']),
        TRUCK_SPEED_M_S,DRONE_SPEED_M_S,DRONE_PAYLOAD_CAPACITY_KG,
        DRONE_BATTERY_CAPACITY_WH,depot,coords,demands,
        np.abs(delta).sum(axis=2),np.sqrt(np.square(delta).sum(axis=2)),str(path))


def build_initial_code(p):
    """Farthest-first balanced cheapest insertion, then detour-saving mode hints."""
    routes=[[] for _ in range(p.num_vehicles)]
    lengths=[0.]*p.num_vehicles
    dist=p.veh_dist_matrix
    order=sorted(range(1,p.num_customers+1),key=lambda c:(-float(dist[0,c]),c))
    for c in order:
        winner=None
        for v,g in enumerate(routes):
            closed=[0]+g+[0]
            for pos in range(len(g)+1):
                extra=float(dist[closed[pos],c]+dist[c,closed[pos+1]]-dist[closed[pos],closed[pos+1]])
                longest=max(lengths[v]+extra,max((lengths[w] for w in range(p.num_vehicles) if w!=v),default=0.))
                key=(longest,extra,len(g),v,pos)
                if winner is None or key<winner[0]:winner=(key,v,pos,extra)
        _,v,pos,extra=winner
        routes[v].insert(pos,c);lengths[v]+=extra
    service=[0]*(p.num_customers+1);drone=[0]*(p.num_customers+1)
    if p.num_drones_per_vehicle:
        for g in routes:
            loads=[0.]*p.num_drones_per_vehicle
            closed=[0]+g+[0]
            for i,c in enumerate(g,1):
                if p.customer_demands[c]>p.drone_payload_capacity+1e-9:continue
                a,b=closed[i-1],closed[i+1]
                old=float(dist[a,c]+dist[c,b])/p.vehicle_speed
                direct=float(dist[a,b])/p.vehicle_speed
                flight=float(p.dro_dist_matrix[a,c]+p.dro_dist_matrix[c,b])/p.drone_speed
                if old>max(direct,flight)+1e-9:
                    service[c]=1
                    di=min(range(len(loads)),key=lambda d:(loads[d],d))
                    drone[c]=di;loads[di]+=p.customer_demands[c]
    return ParticleCode(routes,service,drone)


def _validate_code(p,code):
    if len(code.master_routes)!=p.num_vehicles:
        raise ValueError('incorrect vehicle group count')
    flat=[c for g in code.master_routes for c in g]
    if sorted(flat)!=list(range(1,p.num_customers+1)):
        raise ValueError('customer partition is invalid')
    if len(code.service)!=p.num_customers+1 or len(code.drone)!=p.num_customers+1:
        raise ValueError('incorrect gene length')
    for c in flat:
        if code.service[c] not in (0,1):raise ValueError('invalid service gene')
        if p.num_drones_per_vehicle and not 0<=code.drone[c]<p.num_drones_per_vehicle:
            raise ValueError('invalid drone index')


def decode(p,code):
    """Start with all-truck service; insert non-worsening feasible drone tasks.

    Proposed groups retain particle order, not energy-optimal reordering.
    Existing launch/recovery anchors are protected. Failed groups are retried as
    singletons; failed singles remain on the truck. Full evaluation decides
    acceptance; window ranking is only an inexpensive shortlist heuristic.
    """
    _validate_code(p,code)
    routes=[[0]+list(g)+[0] for g in code.master_routes]
    drones=[[] for _ in routes]
    ev=evaluate_solution(routes,drones,p,True)
    if not ev.feasible:raise RuntimeError('all-truck baseline infeasible: '+ev.reason)
    baseline=ev.objective
    for v,master in enumerate(code.master_routes):
        for di in range(p.num_drones_per_vehicle):
            seq=[c for c in master if code.service[c] and code.drone[c]==di
                 and p.customer_demands[c]<=p.drone_payload_capacity+1e-9]
            groups=[];group=[];payload=0.
            for c in seq:
                if group and payload+p.customer_demands[c]>p.drone_payload_capacity+1e-9:
                    groups.append(group);group=[];payload=0.
                group.append(c);payload+=p.customer_demands[c]
            if group:groups.append(group)
            pending=list(groups)
            while pending:
                delivery=pending.pop(0)
                anchors={node for s in drones[v] for node in (s[1],s[3])}
                if any(c in anchors for c in delivery):
                    if len(delivery)>1:pending[:0]=[[c] for c in delivery]
                    continue
                members=set(delivery)
                reduced=[c for c in routes[v] if c not in members]
                prefix=[0.]
                for a,b in zip(reduced,reduced[1:]):
                    prefix.append(prefix[-1]+float(p.veh_dist_matrix[a,b])/p.vehicle_speed)
                flight=prepare_fixed_order_flight(p,delivery)
                label=f'{v+1}-D{di+1}'
                candidates=[]
                for li in range(len(reduced)-1):
                    for ri in range(li+1,len(reduced)):
                        # Check same-drone ordering before shortlist selection.
                        overlap=False
                        for s in drones[v]:
                            if s[0]!=label:continue
                            l=0 if s[1]==0 else reduced.index(s[1])
                            r=len(reduced)-1 if s[3]==0 else reduced.index(s[3])
                            if not (ri<=l or li>=r):overlap=True;break
                        if overlap:continue
                        ft,fe=flight(reduced[li],reduced[ri])
                        if fe>p.drone_battery_capacity_wh+1e-9:continue
                        wait=max(0.,ft-(prefix[ri]-prefix[li]))
                        candidates.append(((prefix[-1]+wait,fe,li,ri),li,ri))
                candidates.sort()
                shortlist=candidates[:WINDOW_CANDIDATES]
                # Include whole-route depot rendezvous if structurally available.
                widest=next((x for x in candidates if x[1]==0 and x[2]==len(reduced)-1),None)
                if widest is not None and widest not in shortlist:shortlist.append(widest)
                best=None
                for _,li,ri in shortlist:
                    trial_routes=[list(r) for r in routes];trial_routes[v]=reduced
                    trial_drones=copy.deepcopy(drones)
                    trial_drones[v].append([label,reduced[li],list(delivery),reduced[ri]])
                    def order(s):
                        return (0 if s[1]==0 else reduced.index(s[1]),
                                len(reduced)-1 if s[3]==0 else reduced.index(s[3]),s[0])
                    trial_drones[v].sort(key=order)
                    candidate=evaluate_solution(trial_routes,trial_drones,p,True)
                    if not candidate.feasible:continue
                    # Other trucks are unchanged, so improving this team's time
                    # cannot worsen global makespan. Equal time may save mileage.
                    if candidate.vehicle_completion_times[v]>ev.vehicle_completion_times[v]+1e-9:continue
                    key=(candidate.vehicle_completion_times[v],candidate.objective,
                         candidate.total_flight_energy_wh+candidate.total_hover_energy_wh,li,ri)
                    if best is None or key<best[0]:
                        best=(key,trial_routes,trial_drones,candidate)
                if best is not None:
                    _,routes,drones,ev=best
                elif len(delivery)>1:
                    pending[:0]=[[c] for c in delivery]
    if ev.objective>baseline+1e-7:raise RuntimeError('non-worsening decoder invariant failed')
    return PSOSolution(copy.deepcopy(code),routes,drones,ev)


def save_solution_txt(path,p,solution):
    text='truck_route=\n'+repr(solution.vehicle_routes)+'\n'
    for v,sorties in enumerate(solution.drone_routes,1):
        text+=f'drone_routes_vehicle{v}=\n'+repr(sorties)+'\n'
    text+=f'总时间成本: {solution.objective:.6f}\n'
    text+='无人机任务格式：[物理无人机编号, 起飞节点, 配送节点列表, 回收节点]\n'
    Path(path).write_text(text,encoding='utf-8')
