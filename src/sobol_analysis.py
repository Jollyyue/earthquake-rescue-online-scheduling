#!/usr/bin/env python3
"""Section 6.2: Jansen analysis of the unchanged Chapter 5 engine.
Default is a cost/design preview. --smoke runs N=8,R=1,N_sim=50;
--execute runs the requested design. See README.txt before a production run.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np

NAMES = ('N_team', 'm_lambda', 'm_v', 'gamma_h', 'P01')
DEFAULT_ENGINE = Path('/Users/yuejingyao/Desktop/code/7788/simulation2_outputs/simulation3.py')
ENGINE_HASH = '0fa281ce7649d837dc30b7e47182ba3ce80a9ea622b94ced99a35577c3aadd2f'
DEFAULT_RANGES = dict(zip(NAMES, ((5,15),(.9,1.1),(.8,1.2),(.8,1.2),(.2,.6))))
MASTER_SEED = 20260914
N_SIM = 50


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_engine(path):
    if digest(path) != ENGINE_HASH:
        raise ValueError('Engine hash differs from audited Chapter 5/OFAT engine; re-audit before use.')
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location('simulation3', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['simulation3'] = module
    spec.loader.exec_module(module)
    return module


def team_pool(base, n):
    """Nested heterogeneous pool; original seven IDs/configurations recovered exactly.
    Prefix order balances the 5:2 types. Engine sees teams sorted by unique ID,
    preserving its original tie-breaking at N=7. Extra teams retain type/depot.
    """
    if int(n) != n or not 5 <= n <= 100:
        raise ValueError('N_team must be an integer in [5,100]')
    n = int(n)
    original = {t.id:t for t in base.rescue_teams}
    pool = [copy.deepcopy(original[i]) for i in (1,6,2,3,4,7,5)]
    n_a = 5
    for count in range(8, n+1):
        target_a = int(math.floor(5*count/7 + .5))
        is_a = target_a > n_a
        team = copy.deepcopy(original[1 if is_a else 6])
        team.id = count
        pool.append(team)
        n_a += int(is_a)
    return sorted(pool[:n], key=lambda t:t.id)


def configure(engine, base, x):
    n, ml, mv, gh, p = map(float,x)
    if not np.isfinite(x).all() or min(ml,mv,gh) <= 0 or not 0 <= p <= 1:
        raise ValueError(f'Illegal parameter vector: {x}')
    s = copy.deepcopy(base)
    s.rescue_teams = team_pool(base,n)
    for t in s.rescue_teams:
        t.fatigue_rate *= ml
        t.speed *= mv
    # Engine's centroid-derived Q workload after linear mutual aid and H workload
    # (3*h) are both scaled. Equivalent to scaling nominal h before fixed alpha
    # aid/conversion. Fuzzy definition and normalized objective weights unchanged.
    s.demand = {task: gh*work for task,work in base.demand.items()}
    # Replay ONLY initial edge uniforms; retain repair work and road-shock draws.
    rng = np.random.default_rng(s.seed)
    for u,v in s.graph.edges:
        s.graph[u][v]['blocked'] = bool(rng.random() < p)
        s.graph[u][v]['repair_complete'] = None
    s.p01, s.p_block_raw, s.p_block, s.p_block_clipped = p/1000, p, p, False
    assert len({t.id for t in s.rescue_teams}) == int(n)
    assert asdict(s.repair_teams[0]) == asdict(base.repair_teams[0])
    assert s.repair_work == base.repair_work and s.weight == base.weight
    assert s.road_shock_draws == base.road_shock_draws
    capacity = sum(t.initial_efficiency/t.fatigue_rate for t in s.rescue_teams)
    if capacity <= sum(s.demand.values()):
        raise ValueError('Total asymptotic rescue capacity cannot cover workload; invalid domain.')
    return s


def seeds(r):
    # Exact Chapter 5/OFAT seed layout, selecting RB (index 2 of five algorithms).
    env = np.random.SeedSequence(MASTER_SEED).spawn(r)
    alg = np.random.SeedSequence(MASTER_SEED+99173).spawn(5*r)
    return [(int(env[i].generate_state(1)[0]),
             int(alg[5*i+2].generate_state(1)[0])) for i in range(r)]


def validate_ranges(ranges):
    if set(ranges) != set(NAMES):
        raise ValueError(f'Ranges must contain exactly {NAMES}')
    for name in NAMES:
        bounds = ranges[name]
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            raise ValueError(f'Invalid bounds: {name}')
    lo,hi=ranges['N_team']
    if lo!=int(lo) or hi!=int(hi) or lo<5 or hi>100:
        raise ValueError('N_team bounds must be integers in [5,100]')
    if any(ranges[k][0]<=0 for k in NAMES[1:4]):
        raise ValueError('Multipliers must be positive')
    if ranges['P01'][0]<0 or ranges['P01'][1]>1:
        raise ValueError('P01 bounds must be in [0,1]')


def design(n, ranges, seed, sampler):
    if n<2 or n & (n-1):
        raise ValueError('N must be a power of two >=2 for nested convergence designs')
    metadata = {'numpy':np.__version__, 'sampler_seed':seed}
    if sampler != 'random':
        try:
            import scipy
            from scipy.stats import qmc
        except ImportError:
            if sampler == 'sobol':
                raise RuntimeError('SciPy unavailable; install SciPy or explicitly use --sampler random')
            sampler='random'
        else:
            u=qmc.Sobol(d=10,scramble=True,seed=seed).random_base2(int(math.log2(n)))
            metadata.update(sampler='scipy.stats.qmc.Sobol', scipy=scipy.__version__, scramble=True)
    if sampler == 'random':
        u=np.random.default_rng(seed).random((n,10))
        metadata.update(sampler='numpy IID uniform; NOT quasi-random')
    def transform(z):
        out=np.empty_like(z)
        for i,name in enumerate(NAMES):
            lo,hi=ranges[name]
            # Equal probability for every integer, unlike rounding endpoints.
            out[:,i] = lo+np.floor(z[:,i]*(hi-lo+1)) if i==0 else lo+z[:,i]*(hi-lo)
        return out
    a,b=transform(u[:,:5]),transform(u[:,5:])
    mats=[a,b]
    for i in range(5):
        ab=a.copy();ab[:,i]=b[:,i];mats.append(ab)
    return np.stack(mats,axis=1), metadata


def jansen(y):
    """y[row, matrix], ordered A,B,AB_1,...,AB_5; AB_i=A with column i of B.
    Eq25: S_i=1-mean((Y_B-Y_ABi)^2)/(2*V).
    Eq26: ST_i=mean((Y_A-Y_ABi)^2)/(2*V).
    V: sample variance (ddof=1) of pooled A/B outputs, not hybrid outputs.
    No clipping, reordering, or enforcement of S<=ST.
    """
    if y.ndim!=2 or y.shape[1]!=7 or not np.isfinite(y).all():
        raise ValueError('Incomplete/nonfinite Sobol outputs')
    v=float(np.var(y[:,:2].ravel(),ddof=1))
    if v<=1e-12*max(1.,float(np.mean(y[:,:2]**2))):
        raise ValueError('Output variance is zero/near zero; indices undefined')
    s=1-np.mean((y[:,1,None]-y[:,2:])**2,axis=0)/(2*v)
    st=np.mean((y[:,0,None]-y[:,2:])**2,axis=0)/(2*v)
    return s,st,v


def indices(y, bootstrap, seed):
    s,st,v=jansen(y)
    ci=None;bad=0
    if bootstrap:
        rng=np.random.default_rng(seed);draws=[]
        for _ in range(bootstrap):
            # Paired ROW resampling preserves A/B/all hybrid blocks together.
            try:
                bs,bt,_=jansen(y[rng.integers(0,len(y),len(y))])
                draws.append(np.stack([bs,bt]))
            except ValueError:
                bad+=1
        if draws:
            ci=np.quantile(draws,[.025,.975],axis=0)
    rows=[]
    for i,name in enumerate(NAMES):
        row={'parameter':name,'S_i':float(s[i]),'ST_i':float(st[i])}
        if ci is not None:
            row.update(S_low=float(ci[0,0,i]),S_high=float(ci[1,0,i]),
                       ST_low=float(ci[0,1,i]),ST_high=float(ci[1,1,i]))
        rows.append(row)
    warnings=[]
    if np.any(s<0) or np.any(st<0) or np.any(s>1) or np.any(st>1) or np.any(s>st):
        warnings.append('Indices outside [0,1] or S>ST: inspect sampling error/convergence; values unmodified.')
    if sum(s)>1:
        warnings.append('Sum of first-order indices exceeds 1; inspect sampling error/convergence.')
    return rows,dict(variance=v,warnings=warnings,bootstrap_failed_draws=bad,
                     ci_scope='Approximate paired-row bootstrap conditional on fixed seed panel; QMC rows are dependent; not a rigorous randomized-QMC CI.')


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def injection_audit(engine, base):
    baseline=[7,1,1,1,base.p_block]
    s=configure(engine,base,baseline)
    assert [asdict(t) for t in s.rescue_teams] == [asdict(t) for t in base.rescue_teams]
    assert s.demand==base.demand and s.weight==base.weight
    assert list(s.graph.edges(data=True))==list(base.graph.edges(data=True))
    for i,value in enumerate([5,1.1,1.2,1.2,.6]):
        x=baseline.copy();x[i]=value;t=configure(engine,base,x);st=engine.state_from_scenario(t)
        if i==0: assert len(st.rescue)==5 and len({a.fatigue_rate for a in st.rescue.values()})==2
        if i==1: assert all(math.isclose(st.rescue[k.id].fatigue_rate,k.fatigue_rate*value) for k in base.rescue_teams)
        if i==2: assert all(math.isclose(st.rescue[k.id].speed,k.speed*value) for k in base.rescue_teams)
        if i==3: assert all(math.isclose(st.remaining[k],v*value) for k,v in base.demand.items())
        if i==4:
            assert st.p_block==value
            assert all(ev.payload['p_block']==value for _,_,ev in st.events if ev.kind=='road_shock')
    return {'baseline_identity':True,'five_state_injections':True,
            'baseline_vector':baseline,'baseline_teams':[asdict(t) for t in base.rescue_teams],
            'baseline_total_workload':sum(base.demand.values()),
            'pool_composition':{str(n):[sum(t.location==12 for t in team_pool(base,n)),sum(t.location==7 for t in team_pool(base,n))] for n in range(5,16)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation3',type=Path,default=DEFAULT_ENGINE)
    p.add_argument('--n',type=int,default=512)
    p.add_argument('--replicates',type=int,default=3)
    p.add_argument('--sampler',choices=['auto','sobol','random'],default='auto')
    p.add_argument('--scramble-seed',type=int,default=20260923)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--ranges',type=Path,help='JSON object mapping five input names to [low,high]')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--convergence',action='store_true',help='Analyze prefixes 128/256/512 of ONE largest design')
    args=p.parse_args()
    if args.smoke:args.n,args.replicates=8,1
    if args.replicates<1 or args.bootstrap<0:p.error('replicates>=1 and bootstrap>=0 required')
    ranges=json.loads(args.ranges.read_text()) if args.ranges else DEFAULT_RANGES
    validate_ranges(ranges)
    engine=load_engine(args.simulation3)
    pairs=seeds(args.replicates)
    bases=[engine.generate_scenario(i+1,e) for i,(e,a) in enumerate(pairs)]
    audit=injection_audit(engine,bases[0])
    # Necessary capacity check at the hardest box corner; NOT a completion guarantee.
    configure(engine,bases[0],[ranges['N_team'][0],ranges['m_lambda'][1],ranges['m_v'][0],ranges['gamma_h'][1],ranges['P01'][1]])
    x,metadata=design(args.n,ranges,args.scramble_seed,args.sampler)
    manifest=dict(status='SMOKE ONLY' if args.smoke else 'PLANNED', n=args.n,
        D=5,replicates=args.replicates,N_sim=N_SIM,parameter_vectors=7*args.n,
        engine_runs=7*args.n*args.replicates,seed_pairs=pairs,ranges=ranges,
        range_status='N_team/m_v/gamma_h and pool expansion provisional; confirm before publication',
        engine=str(args.simulation3.resolve()),engine_sha256=digest(args.simulation3),
        script_sha256=digest(__file__),python=platform.python_version(),sampling=metadata,
        fixed_parameters={'alpha':.2,'beta':.3,'t_max':300,'strategy':'Rollout-Benefit'},
        duplicate_vectors=int(len(x.reshape(-1,5))-len(np.unique(x.reshape(-1,5),axis=0))),
        equal_team_hybrids=int(np.sum(x[:,0,0]==x[:,1,0])),audit=audit)
    print(json.dumps({k:manifest[k] for k in ('status','n','replicates','parameter_vectors','engine_runs','sampling')},indent=2),flush=True)
    if not (args.execute or args.smoke):
        print('Preview only. --smoke validates; --execute starts actual computation.');return
    out=args.output_dir or Path(__file__).resolve().parent/('smoke_results' if args.smoke else f'results_N{args.n}_R{args.replicates}')
    out.mkdir(parents=True,exist_ok=False) # Never overwrite previous experiments.
    write_json(out/'manifest.json',manifest)
    np.savez(out/'design.npz',x=x)
    y=np.full((args.n,7,args.replicates),np.nan)
    count=0;started=time.perf_counter();rows=[]
    try:
        with (out/'raw_evaluations.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=None
            for j in range(args.n):
                for k,label in enumerate(('A','B',*['AB_'+n for n in NAMES])):
                    for r,(base,(_,algseed)) in enumerate(zip(bases,pairs)):
                        scenario=configure(engine,base,x[j,k])
                        result,_=engine.run_algorithm(scenario,'Rollout-Benefit',algseed,n_sim=N_SIM,search_budget=60)
                        count+=1
                        row=dict(row=j,matrix=label,replicate=r,**{'input_'+name:float(value) for name,value in zip(NAMES,x[j,k])},**result)
                        if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);f.flush()
                        if result['termination']!='completed' or result['unfinished_tasks']!=0 or not math.isfinite(result['objective']):
                            raise RuntimeError(f'Noncompleted/nonfinite engine run at row {j}, {label}, replicate {r}; no Sobol indices emitted.')
                        if result['early_open_violations'] or result['duplicate_service_violations'] or result['min_remaining_work'] < -1e-7:
                            raise RuntimeError('Engine integrity violation')
                        y[j,k,r]=result['objective']
                        print(f'{count}/{manifest["engine_runs"]} row={j} {label} rep={r} objective={result["objective"]:.8f} completed',flush=True)
                np.save(out/'outputs_by_replicate.npy',y)
        mean_y=y.mean(axis=2)
        sizes=sorted({args.n,*([n for n in (128,256,512) if n<=args.n] if args.convergence else [])})
        for size in sizes:
            table,diag=indices(mean_y[:size],args.bootstrap,args.scramble_seed+size)
            write_csv(out/f'indices_N{size}.csv',table)
            write_json(out/f'diagnostics_N{size}.json',diag)
        # Replicate-prefix comparison exposes finite-panel sensitivity. R=1 is
        # conditional scenario sensitivity, NOT demonstrated E[Y|X] convergence.
        panel=[]
        for r in range(1,args.replicates+1):
            s,st,v=jansen(y[:,:,:r].mean(axis=2))
            panel.append(dict(replicates=r,S=s.tolist(),ST=st.tolist(),variance=v))
        write_json(out/'replicate_convergence.json',panel)
        elapsed=time.perf_counter()-started
        write_json(out/'integrity.json',dict(passed=True,completed=count,expected=manifest['engine_runs'],
            elapsed_seconds=elapsed,mean_seconds_per_run=elapsed/count,
            projected_N512_R3_hours=(elapsed/count)*3584*3/3600,
            engine_unchanged=digest(args.simulation3)==ENGINE_HASH,
            warning='Smoke indices are diagnostics only; no factor ranking inference. Fixed-panel convergence remains unproven.'))
    except Exception as exc:
        np.save(out/'outputs_by_replicate.npy',y)
        write_json(out/'integrity.json',dict(passed=False,attempted=count,valid=int(np.isfinite(y).sum()),error=str(exc)))
        raise
    finally:
        engine._shutdown_rollout_pool()
    print(f'Finished: {out}',flush=True)


if __name__=='__main__':
    main()
