"""Read-only validation diagnosis; never consumes the new holdout."""
import json
from pathlib import Path
import numpy as np
from runtime import ART
from run import write

OUT=ART/'results/optimized'


def main():
    records={}
    for folder in sorted(OUT.glob('pendulum_*_s*')):
        for mode in ['value','no_value']:
            p=folder/('eval_'+mode)/'summary.json'
            if not p.exists():continue
            summary=json.loads(p.read_text());eps=summary['episodes']
            traces=[json.loads((p.parent/('trace_%02d.json'%i)).read_text()) for i in range(len(eps))]
            h=np.array([r['horizon'] for tr in traces for r in tr])
            key=folder.name+'/'+mode
            records[key]={'mean_cost':summary['mean_total_cost'],
                'performance':float(np.mean([e['performance_cost'] for e in eps])),
                'compute':float(np.mean([e['computation_cost'] for e in eps])),
                'constraint_cost':float(np.mean([e['constraint_cost'] for e in eps])),
                'failed_cases':[i for i,e in enumerate(eps) if e['termination']=='constraint'],
                'h_quantiles':np.quantile(h,[0,.1,.5,.9,1]).tolist(),
                'fraction_h_below20':float(np.mean(h<20)),
                'case_rows':[{**{k:e[k] for k in ['episode','total_cost','performance_cost','computation_cost','constraint_cost','steps','termination','mean_horizon','solver_failure_steps']},
                    'h_min':min(r['horizon'] for r in tr),'h_max':max(r['horizon'] for r in tr),
                    'first_solver_failure':next((t for t,r in enumerate(tr) if not r['solver_success']),None),
                    'end_state':tr[-1]['state'],'final_residual':tr[-1]['max_constraint_residual']}
                    for e,tr in zip(eps,traces)]}
    fixed=records['pendulum_fixed_h30_s0/value']
    comparison={}
    for seed in range(3):
        key='pendulum_rl_s%d/value'%seed;r=records[key]
        comparison[key]={'mean_difference':r['mean_cost']-fixed['mean_cost'],
            'performance_difference':r['performance']-fixed['performance'],
            'compute_difference':r['compute']-fixed['compute'],
            'constraint_difference':r['constraint_cost']-fixed['constraint_cost'],
            'case_differences':[a['total_cost']-b['total_cost'] for a,b in zip(r['case_rows'],fixed['case_rows'])],
            'value_minus_zero':r['mean_cost']-records[key.replace('/value','/no_value')]['mean_cost']}
    stable_grid=[records['pendulum_fixed_h%d_s0/value'%h] for h in range(20,51,5)]
    grid_costs=np.array([[e['total_cost'] for e in r['case_rows']] for r in stable_grid])
    oracle={'scope':'H20..50 per-scene hindsight fixed choice; not implementable and not an upper bound on dynamic H.',
            'mean_cost':float(grid_costs.min(axis=0).mean()),
            'selected_h':(20+5*grid_costs.argmin(axis=0)).tolist()}
    result={'scope':'Original 10 validation scenes only; descriptive, fixed comparator seed0 pending replication.',
          'records':records,'comparison':comparison,'hindsight_fixed_oracle':oracle}
    write(OUT/'pendulum_validation_diagnosis.json',result)
    print(json.dumps({'comparison':comparison,'hindsight_fixed_oracle':oracle},indent=2))


if __name__=='__main__':main()
