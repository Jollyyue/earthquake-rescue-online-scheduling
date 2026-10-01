#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible event-driven rescue scheduling experiment.

This is a clean-room reconstruction of ``simulation.py``.  It preserves the
original network, demand model, team parameters, five public algorithm names
and the cumulative weighted rescue-time objective, but replaces the mutable,
algorithm-specific simulation loops with one shared transition engine.

Important modelling choices
---------------------------
* ``P_block_raw = 1000 * P01`` is retained from the paper/code.  The probability
  used for Bernoulli edge states is ``clip(P_block_raw, 0, 1)``.  The raw and
  clipped values and whether clipping occurred are exported.
* A blocked edge remains closed until an explicit ``repair_complete`` event.
* Fatigue is integrated continuously.  For a team with initial efficiency e,
  fatigue lambda and accumulated service time s, work over dt is
  e/lambda*(exp(-lambda*s)-exp(-lambda*(s+dt))).
* GA and ACO search the same feasible atomic action set used by the other
  methods and receive the same number of objective evaluations
  (``search_budget``).
* Rollout candidates are explicit ``(team,node,victim_type)`` actions.  Every
  candidate is evaluated from a deep-copied state using stochastic future
  road-state trajectories and common-random-number seeds.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import heapq
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

EPS = 1e-9
INF = 1e12
ALGORITHMS = ("Greedy", "Rollout-Time", "Rollout-Benefit", "GA", "ACO")
VICTIM_TYPES = ("Q", "H")
ROAD_SHOCK_TIMES = (30.0, 60.0, 90.0, 120.0)
_ROLLOUT_POOL: Optional[ProcessPoolExecutor] = None


@dataclass
class Team:
    id: int
    kind: str
    location: int
    speed: float
    initial_efficiency: float
    fatigue_rate: float
    service_clock: float = 0.0
    status: str = "idle"  # idle, travelling, serving, repairing
    target: Optional[Tuple[int, str]] = None
    destination: Optional[int] = None
    event_token: int = 0


@dataclass
class Scenario:
    scenario_id: int
    seed: int
    graph: nx.Graph
    demand: Dict[Tuple[int, str], float]
    weight: Dict[Tuple[int, str], float]
    rescue_teams: List[Team]
    repair_teams: List[Team]
    repair_work: Dict[Tuple[int, int], float]
    p01: float
    p_block_raw: float
    p_block: float
    p_block_clipped: bool
    road_shock_draws: Dict[float, Dict[Tuple[int, int], float]]


@dataclass
class Event:
    time: float
    order: int
    kind: str
    team_id: int
    payload: Dict[str, Any] = field(default_factory=dict)

    def heap_item(self) -> Tuple[float, int, "Event"]:
        return (self.time, self.order, self)


@dataclass
class State:
    t: float
    graph: nx.Graph
    remaining: Dict[Tuple[int, str], float]
    initial_demand: Dict[Tuple[int, str], float]
    weight: Dict[Tuple[int, str], float]
    rescue: Dict[int, Team]
    repair: Dict[int, Team]
    repair_work: Dict[Tuple[int,int], float]
    p_block: float
    events: List[Tuple[float, int, Event]] = field(default_factory=list)
    active: Dict[Tuple[int, str], set] = field(default_factory=dict)
    objective: float = 0.0
    first_response: Optional[float] = None
    completion_time: Optional[float] = None
    event_counter: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    served_tasks: set = field(default_factory=set)
    early_open_violations: int = 0
    duplicate_service_violations: int = 0


def daoluduse(ac: float, pga: float, m: float, a01: float, b01: float) -> float:
    x = ac / pga
    if not 0.0 < x < 1.0:
        raise ValueError("ac/PGA must be in (0,1)")
    dn = 0.215 + math.log10(((1 - x) ** 2.341) * (x ** -1.436))
    return m * (1.0 - math.exp(-a01 * dn**b01))


def base_graph() -> nx.Graph:
    edges = [
        (1,2,21),(1,3,21),(1,4,51),(2,3,21),(3,4,55),(2,5,43),(2,7,31),
        (5,7,26),(7,8,70),(6,8,68),(7,9,68),(8,9,58),(7,11,49),(7,10,54),
        (10,11,50),(9,11,23),(5,12,50),
    ]
    g = nx.Graph()
    for u, v, w in edges:
        g.add_edge(u, v, weight=float(w), blocked=False, repair_complete=None)
    return g


def fuzzy_renshu(magnitude: int, building: int, population: int) -> Tuple[np.ndarray, np.ndarray]:
    maps = {
        "earthquake": {1:[10,20,30], 2:[30,40,50], 3:[50,60,70]},
        "building": {1:[10,20,30], 2:[30,40,50], 3:[50,60,70]},
        "population": {1:[20,30,40], 2:[40,50,60], 3:[60,70,80]},
    }
    e = np.asarray(maps["earthquake"][magnitude], float)
    b = np.asarray(maps["building"][building], float)
    p = np.asarray(maps["population"][population], float)
    return 0.5*e + 0.3*b + 0.2*p, 0.3*e + 0.5*b + 0.2*p


def generate_scenario(scenario_id: int, seed: int) -> Scenario:
    """Create all exogenous randomness exactly once for paired comparisons."""
    rng = np.random.default_rng(seed)
    g = base_graph()
    p01 = daoluduse(0.1, 0.8, 0.2698, 0.0005, 3.3776)
    raw = 1000.0 * p01
    p_block = float(np.clip(raw, 0.0, 1.0))
    clipped = not math.isclose(raw, p_block)
    for u, v in g.edges:
        g[u][v]["blocked"] = bool(rng.random() < p_block)

    magnitude = [2]*11
    building = [2,2,2,2,2,2,3,2,1,2,1]
    population = [3,1,2,1,1,1,3,2,1,2,1]
    severity = [5,5,5,3,3,1,3,1,1,1,1]
    q_people, h_people, omega = [], [], []
    for i in range(11):
        fq, fh = fuzzy_renshu(magnitude[i], building[i], population[i])
        # Main experiment: centroid-defuzzified nominal affected population.
        # A fuzzy membership function is not treated as a probability density.
        q_people.append(int(round(float(np.mean(fq)))))
        h_people.append(int(round(float(np.mean(fh)))))
        s = severity[i]
        omega.append(float(s))
    total_q, total_h = max(1, sum(q_people)), max(1, sum(h_people))
    demand: Dict[Tuple[int,str], float] = {}
    weight: Dict[Tuple[int,str], float] = {}
    for i in range(11):
        # Original mutual/self-aid equation, with the required non-negative truncation.
        aid = q_people[i] * max(0.0, math.exp(-0.2*omega[i]) - 0.3)
        demand[(i+1,"Q")] = float(max(0.0, q_people[i] - aid))
        demand[(i+1,"H")] = float(max(0, 3*h_people[i]))
        weight[(i+1,"Q")] = omega[i]*q_people[i]/total_q
        weight[(i+1,"H")] = omega[i]*h_people[i]/total_h

    rescue = [
        Team(i+1,"rescue",12,50,30,0.05) for i in range(5)
    ] + [Team(i+6,"rescue",7,40,20,0.10) for i in range(2)]
    repair = [Team(101+i,"repair",7,50,20,0.15) for i in range(2)]
    repair_work = {tuple(sorted((u,v))): float(rng.integers(5,16)) for u,v in g.edges}
    shock_draws = {
        at: {tuple(sorted((u, v))): float(rng.random()) for u, v in g.edges}
        for at in ROAD_SHOCK_TIMES
    }
    return Scenario(scenario_id, seed, g, demand, weight, rescue, repair,
                    repair_work, p01, raw, p_block, clipped, shock_draws)


def make_scenarios(n: int, master_seed: int) -> List[Scenario]:
    seq = np.random.SeedSequence(master_seed)
    return [generate_scenario(i, int(s.generate_state(1)[0]))
            for i, s in enumerate(seq.spawn(n), start=1)]


def state_from_scenario(s: Scenario) -> State:
    st = State(0.0, copy.deepcopy(s.graph), copy.deepcopy(s.demand),
                 copy.deepcopy(s.demand), copy.deepcopy(s.weight),
                 {t.id:copy.deepcopy(t) for t in s.rescue_teams},
                 {t.id:copy.deepcopy(t) for t in s.repair_teams},
                 copy.deepcopy(s.repair_work), s.p_block,
                 active={k:set() for k in s.demand})
    for at, draws in s.road_shock_draws.items():
        push_event(st, at, "road_shock", 0, draws=copy.deepcopy(draws),
                   p_block=s.p_block)
    return st


def open_graph(g: nx.Graph) -> nx.Graph:
    return nx.edge_subgraph(g, [(u,v) for u,v,d in g.edges(data=True)
                                if not d.get("blocked", False)]).copy()


def route(g: nx.Graph, source: int, target: int) -> Tuple[Optional[List[int]], float]:
    if source == target:
        return [source], 0.0
    def edge_weight(u: int, v: int, data: Dict[str, Any]) -> float:
        return INF if data.get("blocked", False) else float(data.get("weight", 1.0))
    try:
        dist, p = nx.single_source_dijkstra(g, source, target, weight=edge_weight)
        if dist >= INF/2:
            return None, INF
        return p, float(dist)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, INF


def fatigue_work(team: Team, dt: float, start_clock: Optional[float] = None,
                 multiplier: float = 1.0) -> float:
    """The single service-duration/work function used by execution and look-ahead."""
    if dt <= 0:
        return 0.0
    s = team.service_clock if start_clock is None else start_clock
    e, lam = team.initial_efficiency*multiplier, team.fatigue_rate
    if lam <= EPS:
        return e*dt
    return e/lam * (math.exp(-lam*s) - math.exp(-lam*(s+dt)))


def time_for_work(team: Team, work: float, start_clock: Optional[float] = None,
                  multiplier: float = 1.0) -> float:
    if work <= EPS:
        return 0.0
    s = team.service_clock if start_clock is None else start_clock
    e, lam = team.initial_efficiency*multiplier, team.fatigue_rate
    if lam <= EPS:
        return work/max(EPS,e)
    inside = math.exp(-lam*s) - work*lam/max(EPS,e)
    if inside <= EPS:
        return INF
    return max(0.0, -math.log(inside)/lam - s)


def remaining_service_capacity(team: Team) -> float:
    """Maximum future workload the team can still deliver under Eq. (13)."""
    if team.fatigue_rate <= EPS:
        return INF
    return (team.initial_efficiency / team.fatigue_rate *
            math.exp(-team.fatigue_rate * team.service_clock))


def committed_capacity(st: State, task: Tuple[int,str]) -> float:
    """Capacity already committed by travelling/serving teams to this task."""
    return sum(remaining_service_capacity(tm) for tm in st.rescue.values()
               if tm.status in {"travelling","serving"} and tm.target==task)


def task_finish_dt(st: State, task: Tuple[int,str]) -> float:
    ids = st.active.get(task, set())
    if not ids or st.remaining[task] <= EPS:
        return INF
    teams = [st.rescue[i] for i in ids]
    target = st.remaining[task]
    hi = 1.0
    while hi < 1e6 and sum(fatigue_work(t,hi) for t in teams) < target:
        hi *= 2
    if hi >= 1e6:
        return INF
    lo = 0.0
    for _ in range(60):
        mid = (lo+hi)/2
        if sum(fatigue_work(t,mid) for t in teams) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def advance(st: State, new_t: float) -> None:
    dt = new_t-st.t
    if dt < -EPS:
        raise RuntimeError("time moved backwards")
    if dt <= EPS:
        st.t = new_t
        return
    for task, ids in st.active.items():
        if st.remaining[task] <= EPS or not ids:
            continue
        done = sum(fatigue_work(st.rescue[i],dt) for i in ids)
        st.remaining[task] = max(0.0, st.remaining[task]-done)
    for team in st.rescue.values():
        if team.status == "serving":
            team.service_clock += dt
    st.t = new_t


def push_event(st: State, at: float, kind: str, team_id: int, **payload: Any) -> None:
    st.event_counter += 1
    ev = Event(at, st.event_counter, kind, team_id, payload)
    heapq.heappush(st.events, ev.heap_item())


def log(st: State, event: str, **details: Any) -> None:
    st.trace.append({"time":round(st.t,8), "event":event, **details})


def schedule_repairs(st: State) -> None:
    """Assign reachable blocked edges to idle repair teams; edges stay closed."""
    assigned = {tuple(sorted(ev.payload["edge"])) for _,_,ev in st.events
                if ev.kind == "repair_complete"}
    candidates = [tuple(sorted((u,v))) for u,v,d in st.graph.edges(data=True)
                  if d.get("blocked",False) and tuple(sorted((u,v))) not in assigned]
    for tm in sorted(st.repair.values(), key=lambda x:x.id):
        if tm.status != "idle" or not candidates:
            continue
        best = None
        for edge in candidates:
            for endpoint in edge:
                _, dist = route(st.graph, tm.location, endpoint)
                if dist < INF and (best is None or dist < best[0]):
                    best = (dist, edge, endpoint)
        if best is None:
            # If the repair depot is isolated by its incident blocked edges, allow
            # access to an incident edge from the depot side; the edge itself is
            # still never traversed before repair completion.
            incident = [(e, tm.location) for e in candidates if tm.location in e]
            if not incident:
                continue
            edge, endpoint = incident[0]
            best = (0.0, edge, endpoint)
        dist, edge, endpoint = best
        travel = dist/max(EPS,tm.speed)
        repair_dt = time_for_work(tm, st.repair_work[edge])
        finish = st.t+travel+repair_dt
        tm.status="repairing"; tm.destination=endpoint; tm.event_token += 1
        push_event(st,finish,"repair_complete",tm.id,edge=edge,endpoint=endpoint,
                   service_dt=repair_dt,token=tm.event_token)
        st.graph[edge[0]][edge[1]]["repair_complete"] = finish
        candidates.remove(edge)
        log(st,"repair_start",team=tm.id,edge=str(edge),finish=finish)


def candidate_actions(st: State) -> List[Tuple[int,int,str]]:
    actions=[]
    idle=sorted((tm for tm in st.rescue.values() if tm.status=="idle"),
                key=lambda tm:tm.id)
    # One atomic decision is made for the next available team.  The same epoch
    # may contain further atomic decisions for the remaining idle teams.
    for tm in idle[:1]:
        for (node,kind),rem in st.remaining.items():
            if rem <= EPS:
                continue
            task=(node,kind)
            # Preserve necessary parallel rescue, but prohibit redundant
            # dispatch once already committed teams can finish the workload.
            if committed_capacity(st,task) >= rem-EPS:
                continue
            _,dist=route(st.graph,tm.location,node)
            if dist < INF:
                actions.append((tm.id,node,kind))
    return actions


def action_operational_time(st: State, action: Tuple[int,int,str]) -> float:
    tid,node,kind=action; tm=st.rescue[tid]; task=(node,kind)
    _,dist=route(st.graph,tm.location,node)
    if dist >= INF:
        return INF
    # Eq. (14) uses the service-time estimate P=h/[H*R(tau)] at the current
    # decision epoch.  Actual execution below still integrates fatigue
    # continuously as service time advances.
    current_capacity=(tm.initial_efficiency *
                      math.exp(-tm.fatigue_rate*tm.service_clock))
    service=st.remaining[task]/max(EPS,current_capacity)
    return dist/max(EPS,tm.speed) + service


def stage_cost(st: State, action: Tuple[int,int,str]) -> float:
    """Eq. (14): severity/weight times current travel plus service time."""
    return st.weight[(action[1], action[2])] * action_operational_time(st, action)


def dispatch(st: State, action: Tuple[int,int,str], accumulate_cost: bool=True) -> bool:
    tid,node,kind=action; tm=st.rescue[tid]; task=(node,kind)
    if tm.status!="idle" or st.remaining.get(task,0)<=EPS:
        return False
    path,dist=route(st.graph,tm.location,node)
    if path is None:
        return False
    if accumulate_cost:
        st.objective += stage_cost(st, action)
    tm.status="travelling"; tm.destination=node; tm.target=task; tm.event_token+=1
    arrive=st.t+dist/max(EPS,tm.speed)
    push_event(st,arrive,"arrival",tid,task=task,path=path,token=tm.event_token)
    log(st,"dispatch",team=tid,node=node,victim_type=kind,arrival=arrive,path=str(path))
    return True


def handle_event(st: State, ev: Event) -> None:
    if ev.kind=="arrival":
        tm=st.rescue[ev.team_id]
        if ev.payload["token"]!=tm.event_token: return
        task=tuple(ev.payload["task"]); tm.location=task[0]
        if st.remaining[task]<=EPS:
            tm.status="idle"; tm.target=None; return
        tm.status="serving"; tm.target=task; st.active[task].add(tm.id)
        if st.first_response is None: st.first_response=st.t
        log(st,"service_join",team=tm.id,node=task[0],victim_type=task[1],remaining=st.remaining[task])
    elif ev.kind=="repair_complete":
        tm=st.repair[ev.team_id]
        if ev.payload["token"]!=tm.event_token: return
        edge=tuple(ev.payload["edge"]); expected=st.graph[edge[0]][edge[1]].get("repair_complete")
        if expected is None or st.t+EPS < expected:
            st.early_open_violations += 1; return
        st.graph[edge[0]][edge[1]]["blocked"]=False
        tm.service_clock += ev.payload["service_dt"]
        tm.location=ev.payload["endpoint"]; tm.status="idle"; tm.destination=None
        log(st,"repair_complete",team=tm.id,edge=str(edge))
    elif ev.kind=="road_shock":
        newly_blocked=[]
        for edge,draw in ev.payload["draws"].items():
            u,v=tuple(edge)
            if not st.graph[u][v].get("blocked",False) and draw < ev.payload["p_block"]:
                st.graph[u][v]["blocked"]=True
                st.graph[u][v]["repair_complete"]=None
                newly_blocked.append(tuple(sorted((u,v))))
        # A travelling team's latest confirmed location remains its departure
        # node.  If its committed path is interrupted, invalidate the arrival
        # token and return it to the feasible action set for re-optimization.
        for tm in st.rescue.values():
            if tm.status != "travelling":
                continue
            arrival = next((x for _,_,x in st.events
                            if x.kind=="arrival" and x.team_id==tm.id
                            and x.payload.get("token")==tm.event_token), None)
            path = arrival.payload.get("path",[]) if arrival else []
            path_edges={tuple(sorted((a,b))) for a,b in zip(path,path[1:])}
            if path_edges.intersection(newly_blocked):
                tm.event_token += 1
                tm.status="idle"; tm.destination=None; tm.target=None
                log(st,"travel_interrupted",team=tm.id)
        log(st,"road_shock",blocked=str(newly_blocked))


def complete_tasks(st: State) -> None:
    for task,rem in list(st.remaining.items()):
        if rem > 1e-7 or not st.active[task]:
            continue
        if task in st.served_tasks:
            st.duplicate_service_violations += 1
        st.served_tasks.add(task); st.remaining[task]=0.0
        for tid in list(st.active[task]):
            tm=st.rescue[tid]; tm.status="idle"; tm.target=None; tm.destination=None
        st.active[task].clear()
        log(st,"task_complete",node=task[0],victim_type=task[1])


def immediate_score(st: State, action: Tuple[int,int,str], benefit: bool=False) -> float:
    task=(action[1],action[2]); finish=action_operational_time(st,action)
    if benefit:
        # Eq. (25): remaining workload per unit operational time; no I/weight.
        return -st.remaining[task]/max(EPS,finish)
    # Eq. (24): time-minimizing base policy; no I/weight.
    return finish


def replace_future_road_scenarios(st: State, seed: int, p_block: float) -> None:
    """Resample only future road transitions; all other state is held fixed."""
    rng=random.Random(seed)
    for _,_,ev in st.events:
        if ev.kind=="road_shock" and ev.time>st.t+EPS:
            ev.payload["draws"]={tuple(sorted((u,v))):rng.random()
                                 for u,v in st.graph.edges}
            ev.payload["p_block"]=p_block


def advance_to_next_event(st: State, t_max: float) -> bool:
    active_dts=[task_finish_dt(st,k) for k in st.active]
    next_completion=st.t+min(active_dts,default=INF)
    next_event=st.events[0][0] if st.events else INF
    nxt=min(next_completion,next_event,t_max)
    if nxt>=INF/2 or nxt<=st.t+EPS and next_event>st.t+EPS:
        return False
    advance(st,nxt);complete_tasks(st)
    while st.events and st.events[0][0]<=st.t+EPS:
        _,_,ev=heapq.heappop(st.events);handle_event(st,ev)
    return True


def simulate_base_policy(st: State, first: Tuple[int,int,str], benefit: bool,
                         t_max: float=300.0) -> float:
    """Eq. (22): event-driven forward simulation through the terminal epoch."""
    initial_objective=st.objective
    if not dispatch(st,first):
        return INF
    while st.t<t_max-EPS and any(v>EPS for v in st.remaining.values()):
        complete_tasks(st);schedule_repairs(st)
        # Repairs and rescue decisions use the same state-transition engine as
        # the actual experiment.  The scenario object is not needed here:
        # existing/future repair events are already contained in the copied state.
        while True:
            acts=candidate_actions(st)
            if not acts: break
            a=min(acts,key=lambda x:immediate_score(st,x,benefit))
            if not dispatch(st,a): break
        if not advance_to_next_event(st,t_max): break
    unfinished=sum(1 for v in st.remaining.values() if v>EPS)
    penalty=1e6*unfinished
    return st.objective-initial_objective+penalty


def _rollout_seed_job(args: Tuple[State, Sequence[Tuple[int,int,str]], int,
                                  bool, float]) -> Tuple[List[float],List[float]]:
    """Evaluate all candidates for one CRN road scenario in a worker process."""
    st,acts,seed,benefit,p_block=args
    values=[];terminal_times=[]
    for action in acts:
        shadow=copy.deepcopy(st)
        replace_future_road_scenarios(shadow,seed,p_block)
        values.append(simulate_base_policy(shadow,action,benefit))
        terminal_times.append(shadow.t)
    return values,terminal_times


def _shutdown_rollout_pool() -> None:
    global _ROLLOUT_POOL
    if _ROLLOUT_POOL is not None:
        _ROLLOUT_POOL.shutdown(wait=True)
        _ROLLOUT_POOL=None


atexit.register(_shutdown_rollout_pool)


def rollout_action(st: State, rng: random.Random, n_sim: int, benefit: bool,
                   p_block: float) -> Optional[Tuple[int,int,str]]:
    global _ROLLOUT_POOL
    acts=candidate_actions(st)
    if not acts:return None
    # Common random numbers: candidate actions use the same future road seeds.
    seeds=[rng.randrange(2**63) for _ in range(n_sim)]
    # Limit parallelism to reduce sustained CPU temperature and fan noise.
    workers=max(1,min(n_sim,(os.cpu_count() or 2)-1,4))
    jobs=[(st,acts,seed,benefit,p_block) for seed in seeds]
    # Exact shortcut: before the next scheduled road shock, all Monte Carlo
    # trajectories are identical.  If every candidate terminates before that
    # shock, one forward simulation is mathematically equivalent to N_sim runs.
    first_values,first_times=_rollout_seed_job(jobs[0])
    next_shock=min((ev.time for _,_,ev in st.events
                    if ev.kind=="road_shock" and ev.time>st.t+EPS),default=INF)
    if max(first_times,default=st.t)<next_shock-EPS:
        scenario_values=[first_values]
    elif workers==1:
        scenario_values=[first_values]+[_rollout_seed_job(job)[0] for job in jobs[1:]]
    else:
        if _ROLLOUT_POOL is None:
            _ROLLOUT_POOL=ProcessPoolExecutor(max_workers=workers)
        # chunksize keeps all candidates sharing one road seed in the same worker,
        # minimizing serialization while preserving exact CRN comparisons.
        scenario_values=[first_values]+[x[0] for x in _ROLLOUT_POOL.map(
            _rollout_seed_job,jobs[1:],chunksize=max(1,n_sim//workers))]
    scores={a:sum(row[i] for row in scenario_values)/len(scenario_values)
            for i,a in enumerate(acts)}
    return min(acts,key=lambda a:scores[a])


def decode_joint_assignment(st: State, genome: Sequence[Tuple[int,str]]) \
        -> Tuple[List[Tuple[int,int,str]],float]:
    """Map a task-order chromosome to one feasible joint assignment."""
    shadow=copy.deepcopy(st);initial=shadow.objective;joint=[]
    for task in genome:
        acts=candidate_actions(shadow)
        action=next((a for a in acts if (a[1],a[2])==task),None)
        if action is None:
            continue
        if dispatch(shadow,action):
            joint.append(action)
        if not any(tm.status=="idle" for tm in shadow.rescue.values()):
            break
    return joint,shadow.objective-initial


def ga_joint_actions(st: State, rng: random.Random, budget: int) \
        -> List[Tuple[int,int,str]]:
    """GA search over the joint assignment of all currently idle teams."""
    tasks=[task for task,rem in st.remaining.items() if rem>EPS]
    idle=sorted((tm for tm in st.rescue.values() if tm.status=="idle"),
                key=lambda tm:tm.id)
    if not tasks or not idle:return []
    pop_size=max(4,min(12,max(4,budget//5)))
    first_team=idle[0]
    guided=sorted(tasks,key=lambda task:stage_cost(
        st,(first_team.id,task[0],task[1])) if route(
            st.graph,first_team.location,task[0])[0] is not None else INF)
    population=[guided]
    while len(population)<pop_size:
        genome=tasks.copy();rng.shuffle(genome);population.append(genome)
    used=0;best_actions=[];best_cost=INF
    while used<budget:
        scored=[]
        for genome in population:
            if used>=budget:break
            actions,cost=decode_joint_assignment(st,genome);used+=1
            scored.append((cost,genome,actions))
            if actions and cost<best_cost:
                best_cost,best_actions=cost,actions
        if not scored:break
        scored.sort(key=lambda x:x[0]);parents=[x[1] for x in scored[:max(2,len(scored)//2)]]
        next_population=[parents[0].copy()]
        while len(next_population)<pop_size:
            p1,p2=rng.sample(parents,2) if len(parents)>1 else (parents[0],parents[0])
            cut=rng.randrange(len(tasks)+1)
            child=p1[:cut]+[task for task in p2 if task not in p1[:cut]]
            if len(child)>1 and rng.random()<.35:
                i,j=rng.sample(range(len(child)),2);child[i],child[j]=child[j],child[i]
            next_population.append(child)
        population=next_population
    return best_actions


def aco_action(st: State, rng: random.Random, budget: int) -> Optional[Tuple[int,int,str]]:
    """Pheromone search over current atomic actions; no offline route sequence."""
    acts=candidate_actions(st)
    if not acts:return None
    pher={a:1.0 for a in acts};best=None;best_cost=INF
    for _ in range(max(1,budget)):
        weights=[pher[a]/max(EPS,stage_cost(st,a)) for a in acts]
        a=rng.choices(acts,weights=weights,k=1)[0];c=stage_cost(st,a)
        if c<best_cost:best,best_cost=a,c
        for x in acts:pher[x]*=.8
        pher[a]+=1.0/max(EPS,c)
    return best


def select_action(st: State, algorithm: str, rng: random.Random,
                  n_sim: int, search_budget: int) -> Optional[Tuple[int,int,str]]:
    acts=candidate_actions(st)
    if not acts:return None
    if algorithm=="Greedy": return min(acts,key=lambda a:stage_cost(st,a))
    if algorithm=="Rollout-Time": return rollout_action(st,rng,n_sim,False,st.p_block)
    if algorithm=="Rollout-Benefit": return rollout_action(st,rng,n_sim,True,st.p_block)
    if algorithm=="ACO": return aco_action(st,rng,search_budget)
    raise ValueError(algorithm)


def run_algorithm(scenario: Scenario, algorithm: str, algorithm_seed: int,
                  t_max: float=300.0, n_sim: int=50, search_budget: int=60,
                  debug: bool=False) -> Tuple[Dict[str,Any],List[Dict[str,Any]]]:
    st=state_from_scenario(scenario);rng=random.Random(algorithm_seed)
    started=time.perf_counter();schedule_repairs(st)
    while st.t<t_max-EPS and any(v>EPS for v in st.remaining.values()):
        complete_tasks(st);schedule_repairs(st)
        if algorithm=="GA":
            # The chromosome represents the complete joint assignment of all
            # currently idle teams at this decision epoch.
            joint=ga_joint_actions(st,rng,search_budget)
            for action in joint:
                if not dispatch(st,action):
                    raise RuntimeError("decoded GA joint action became infeasible")
        else:
            while True:
                a=select_action(st,algorithm,rng,n_sim,search_budget)
                if a is None or not dispatch(st,a):break
        if not advance_to_next_event(st,t_max):break
    complete_tasks(st)
    unfinished=sum(1 for v in st.remaining.values() if v>1e-7)
    if unfinished==0:st.completion_time=st.t
    runtime=time.perf_counter()-started
    min_remaining=min(st.remaining.values())
    result={
        "scenario_id":scenario.scenario_id,"seed":scenario.seed,
        "algorithm":algorithm,"algorithm_seed":algorithm_seed,
        "objective":st.objective,"runtime":runtime,
        "first_response":st.first_response,"completion_time":st.completion_time,
        "unfinished_tasks":unfinished,"termination":"completed" if unfinished==0 else "T_max_or_unreachable",
        "min_remaining_work":min_remaining,"duplicate_service_violations":st.duplicate_service_violations,
        "early_open_violations":st.early_open_violations,"events":len(st.trace),
        "dispatches":sum(x.get("event")=="dispatch" for x in st.trace),
        "P01":scenario.p01,"P_block_raw_1000x":scenario.p_block_raw,
        "P_block":scenario.p_block,"P_block_clipped":scenario.p_block_clipped,
        "N_sim":n_sim,"search_budget":search_budget,
    }
    return result,st.trace if debug else []


def write_csv(path: Path, rows: List[Dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for alg in ALGORITHMS:
        vals=np.asarray([r["objective"] for r in rows if r["algorithm"]==alg],float)
        q1,med,q3=np.quantile(vals,[.25,.5,.75])
        out.append({"algorithm":alg,"n":len(vals),"mean":vals.mean(),"median":med,
                    "sd":vals.std(ddof=1) if len(vals)>1 else 0.0,"q1":q1,"q3":q3,"iqr":q3-q1})
    write_csv(path,out);return out


def make_plots(outdir: Path, rows: List[Dict[str,Any]]) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib is not installed; CSV results were written and plots were skipped.",
              flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns=None
    data=[[r["objective"] for r in rows if r["algorithm"]==a] for a in ALGORITHMS]
    plt.figure(figsize=(10,6));plt.boxplot(data,tick_labels=ALGORITHMS,showmeans=True)
    plt.ylabel("Cumulative weighted rescue time");plt.xticks(rotation=15);plt.tight_layout()
    plt.savefig(outdir/"five_algorithms_boxplot.png",dpi=220);plt.close()
    plt.figure(figsize=(10,6))
    for a,vals in zip(ALGORITHMS,data):
        if sns and len(set(vals))>1:sns.kdeplot(vals,label=a,fill=False)
        else:plt.hist(vals,bins=12,density=True,histtype="step",label=a)
    plt.xlabel("Cumulative weighted rescue time");plt.ylabel("Density");plt.legend();plt.tight_layout()
    plt.savefig(outdir/"five_algorithms_kde.png",dpi=220);plt.close()


def experiment(n_scenarios: int, master_seed: int, n_sim: int, search_budget: int,
               outdir: Path, smoke: bool=False) -> List[Dict[str,Any]]:
    scenarios=make_scenarios(n_scenarios,master_seed);rows=[];debug=[]
    seed_seq=np.random.SeedSequence(master_seed+99173).spawn(n_scenarios*len(ALGORITHMS))
    k=0
    for s in scenarios:
        for alg in ALGORITHMS:
            aseed=int(seed_seq[k].generate_state(1)[0]);k+=1
            print(f"scenario={s.scenario_id:03d} algorithm={alg:16s} starting...",
                  flush=True)
            result,trace=run_algorithm(s,alg,aseed,n_sim=n_sim,search_budget=search_budget,debug=smoke)
            rows.append(result)
            if smoke:debug.extend({"scenario_id":s.scenario_id,"algorithm":alg,**x} for x in trace)
            print(f"scenario={s.scenario_id:03d} algorithm={alg:16s} objective={result['objective']:.4f} unfinished={result['unfinished_tasks']}")
    outdir.mkdir(parents=True,exist_ok=True)
    write_csv(outdir/"simulation2_results.csv",rows)
    write_summary(outdir/"simulation2_summary.csv",rows)
    with (outdir/"scenario_manifest.json").open("w",encoding="utf-8") as f:
        json.dump([{"scenario_id":s.scenario_id,"seed":s.seed} for s in scenarios],f,indent=2)
    if smoke:write_csv(outdir/"simulation2_debug_events.csv",debug)
    make_plots(outdir,rows)
    return rows


def smoke_assertions(rows: List[Dict[str,Any]]) -> None:
    assert all(r["unfinished_tasks"]==0 for r in rows),"unfinished tasks"
    assert all(r["min_remaining_work"]>=-1e-7 for r in rows),"negative work"
    assert all(r["duplicate_service_violations"]==0 for r in rows),"duplicate completion"
    assert all(r["early_open_violations"]==0 for r in rows),"road opened early"
    assert all(r["termination"] in {"completed","T_max_or_unreachable"} for r in rows)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--scenarios",type=int,default=100)
    p.add_argument("--master-seed",type=int,default=20260914)
    p.add_argument("--n-sim",type=int,default=50,help="rollout futures per candidate")
    p.add_argument("--search-budget",type=int,default=60)
    p.add_argument("--output-dir",type=Path,default=Path("simulation2_outputs"))
    p.add_argument("--smoke",action="store_true")
    return p.parse_args()


if __name__=="__main__":
    args=parse_args()
    rows=experiment(args.scenarios,args.master_seed,args.n_sim,args.search_budget,args.output_dir,args.smoke)
    if args.smoke:smoke_assertions(rows)
    print(f"Wrote {len(rows)} paired algorithm results to {args.output_dir.resolve()}")
