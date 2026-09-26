"""Independently recompute refined holdout costs and physical constraints."""
import hashlib
import argparse
import json
import numpy as np
from runtime import ART,ROOT
from run import write

ap=argparse.ArgumentParser();ap.add_argument('--group',default='refined',choices=['refined','paper_defaults']);args=ap.parse_args()
out=ART/'results'/args.group
assert json.loads((out/'holdout_completed.json').read_text())['complete']
rows=[]
folders=sorted(p.parent for p in out.glob('*/completed.json'))
assert len(folders)==12
for folder in folders:
    spec=json.loads((folder/'manifest.json').read_text())
    completed=json.loads((folder/'completed.json').read_text())
    batch=spec['adaptations'].get('batch_size',64)
    assert completed['steps']==15000 and completed['updates']==15000-max(100,batch)+1
    assert completed['weights_changed'] and completed['evaluation_frozen']
    assert spec['adaptations']['aligned'] and spec['adaptations']['scaled_obs']
    task=spec['task']
    assert spec['config_sha256']==hashlib.sha256((ART/'configs'/(task+'.json')).read_bytes()).hexdigest()
    assert spec['test_bank_sha256']==hashlib.sha256((ART/'configs'/(task+'_validation_bank.json')).read_bytes()).hexdigest()
    cases=json.loads((ART/'configs'/(task+'_holdout_bank.json')).read_text())['cases']
    for mode in ['holdout_value','holdout_no_value']:
        ev=folder/mode
        summary=json.loads((ev/'summary.json').read_text())
        assert len(summary['episodes'])==len(cases)==20
        values=[]
        for j,(episode,case) in enumerate(zip(summary['episodes'],cases)):
            trace=json.loads((ev/('trace_%02d.json'%j)).read_text())
            assert len(trace)==episode['steps']
            assert episode['solver_failure_steps']==sum(not r['solver_success'] for r in trace)
            costs=[];violations=[]
            for t,row in enumerate(trace):
                s=row['state'];u={k:float(np.asarray(v).ravel()[0]) for k,v in row['input'].items()}
                assert row['horizon'] in range(1,51)
                if spec['fixed_horizon'] is not None:assert row['horizon']==spec['fixed_horizon']
                def ref(name):return case['tvp'][name][t+2]['true'][0]
                if task=='vehicle':
                    perf=(s['x']-ref('trajectory_x'))**2+(s['y']-ref('trajectory_y'))**2
                    assert -1e-5<=u['u_s']<=5+1e-5 and abs(u['u_omega'])<=4+1e-5
                    violation=any(np.hypot(s['x']-ref('obj_%d_x'%k),s['y']-ref('obj_%d_y'%k))<=ref('obj_%d_r'%k) for k in range(3))
                else:
                    perf=.5*.8*s['v']**2+.2*.25*s['v']*s['omega']*np.cos(s['theta'])+(2/3)*.2*.25**2*s['omega']**2-.2*9.81*.25*np.cos(s['theta'])+10*(s['pos']-ref('pos_r'))**2+.1*u['u1']**2
                    assert abs(u['u1'])<=5+1e-5
                    violation=abs(s['pos'])>1.5 or abs(s['theta'])>np.pi/2
                assert np.isclose(perf,row['performance'],atol=1e-8)
                assert np.isclose(row['compute'],row['horizon']*(.001 if task=='vehicle' else .003))
                total=perf+row['compute']+row['constraint']
                assert np.isclose(-row['reward'],total)
                costs.append(total);violations.append(bool(violation))
            assert not any(violations[:-1])
            if violations[-1]:assert episode['termination']=='constraint'
            assert np.isclose(sum(costs),episode['total_cost'])
            values.append(sum(costs))
        assert np.isclose(np.mean(values),summary['mean_total_cost'])
        assert summary['goal_episodes']==sum(r['termination']=='goal' for r in summary['episodes'])
        assert summary['constraint_episodes']==sum(r['termination']=='constraint' for r in summary['episodes'])
        rows.append({'run':folder.name,'evaluation':mode,'episodes':20,'passed':True})
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'experiments/bohn2021_reproduction').glob('*.py')}
write(out/'audit.json',{'passed':True,'training_runs':len(rows)//2,'holdout_episodes':len(rows)*20,
    'checks':['15000 transitions and expected batch-dependent update count','frozen evaluation','physical-time reference alignment',
              'independent physical stage cost','horizon and total cost','input bounds','physical constraint termination'],
    'rows':rows,'script_sha256':hashes})
print('Verified',len(rows)*20,'holdout episodes')
