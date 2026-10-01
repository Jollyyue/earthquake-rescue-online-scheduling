import sys,json,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sobol_analysis as s
import numpy as np
def main():
    engine=s.load_engine(s.DEFAULT_ENGINE)
    pairs=s.seeds(1);base=engine.generate_scenario(1,pairs[0][0])
    x,meta=s.design(4096,s.DEFAULT_RANGES,20260923,'sobol')
    # Known additive function has analytically known S=ST, includes discrete N_team.
    variances=np.array([10.,.2**2/12,.4**2/12,.4**2/12,.4**2/12])
    w=np.array([.1,2,3,4,5]);expected=w*w*variances/sum(w*w*variances)
    y=np.einsum('nkd,d->nk',x,w)
    si,sti,v=s.jansen(y)
    assert max(abs(si-expected))<.01 and max(abs(sti-expected))<.01
    small,_=s.design(128,s.DEFAULT_RANGES,20260923,'sobol')
    assert np.array_equal(x[:128],small)
    for n in range(5,16):
     t=s.team_pool(base,n)
     assert len(t)==n and len({a.id for a in t})==n
     assert len({a.fatigue_rate for a in t})==2
    baseline=[7,1,1,1,base.p_block]
    result,_=engine.run_algorithm(s.configure(engine,base,baseline),'Rollout-Benefit',pairs[0][1],n_sim=50)
    again,_=engine.run_algorithm(s.configure(engine,base,baseline),'Rollout-Benefit',pairs[0][1],n_sim=50)
    with open('/Users/yuejingyao/Desktop/code/7788/simulation2_outputs/run_1/simulation2_results.csv',encoding='utf-8-sig') as f:
     historic=next(r for r in csv.DictReader(f) if r['scenario_id']=='1' and r['algorithm']=='Rollout-Benefit')
    assert result['objective']==float(historic['objective'])
    assert result['objective']==again['objective']
    assert result['termination']==again['termination']=='completed'
    report={'analytic_additive_test':{'N':4096,'expected':expected.tolist(),'S':si.tolist(),'ST':sti.tolist()},'nested_design':True,'heterogeneous_pool':True,'baseline_replay_model_evaluations':2,'baseline_replay_completed':2,'chapter5_objective':float(historic['objective']),'adapter_objective':result['objective'],'repeat_objective':again['objective'],'N_sim':50,'engine_sha256':s.digest(s.DEFAULT_ENGINE)}
    s.write_json(Path(__file__).resolve().parent/'validation_rerun.json',report)
    print(json.dumps(report,indent=2))
    engine._shutdown_rollout_pool()

if __name__=='__main__':
    main()
