"""Independently reconcile every logged fixed-H training/evaluation transition.

Uses analytic unicycle integration and SciPy DOP853 for pendulum (rtol/atol1e-12).
Cross-integrator comparison tolerance is rtol/atol1e-7; this is a numerical
consistency check, not a change to the control-performance acceptance rules.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from conservative_iteration import OUT
from paper_h_soft_probe import read, digest
from paper_grid_audit_report import close, scalar
from branch_calibration_audit import observation
from run import write


def integrate(task,s,u):
    if task=='vehicle':
        v,w=scalar(u['u_s']),scalar(u['u_omega']);angle=w*.1
        distance=v*.1*np.sinc(angle/(2*np.pi))
        return dict(x=s['x']+distance*np.cos(s['theta']+angle/2),
                    y=s['y']+distance*np.sin(s['theta']+angle/2),theta=s['theta']+angle)
    force=scalar(u['u1'])
    def rhs(t,y):
        pos,v,theta,omega=y;c=np.cos(theta);sn=np.sin(theta)
        acceleration=(.2*9.81*sn*c-(4/3)*(force+.2*.25*omega**2*sn))/(.2*c*c-(4/3)*.8)
        angular=(.8*9.81*sn-c*(force+.2*.25*omega**2*sn))/((4/3)*.8*.25-.2*.25*c*c)
        return [v,acceleration,omega,angular]
    keys=['pos','v','theta','omega']
    sol=solve_ivp(rhs,(0,.04),[s[k] for k in keys],method='DOP853',rtol=1e-12,atol=1e-12)
    assert sol.success
    return dict(zip(keys,sol.y[:,-1]))


def audit(folder):
    spec=read(folder/'manifest.json');done=read(folder/'completed.json');instrument=read(folder/'instrumentation_completed.json')
    assert instrument['passed'] and done['status']=='complete'
    for p,h in instrument['hashes'].items():assert digest(Path(p))==h
    files=list(folder.glob('transitions_*.jsonl'));resfiles=list(folder.glob('resets_*.jsonl'))
    assert len(files)==len(resfiles)==1,'Interrupted attempts require a separate audit'
    rows=[json.loads(line) for line in files[0].open()]
    resets=[json.loads(line) for line in resfiles[0].open()]
    counts=instrument['counts'];task=spec['task'];h=spec['fixed_horizon'];steps=spec['steps']
    assert len(rows)==counts['step_attempts']==counts['step_completed']
    assert len(resets)==counts['reset_attempts']==counts['reset_completed']
    assert [r['index'] for r in rows]==list(range(1,len(rows)+1))
    assert [r['reset'] for r in resets]==list(range(1,len(resets)+1))
    assert done['steps']==steps==instrument['training_steps']
    assert len(rows)-steps==instrument['extra_evaluation_steps']
    assert done['updates']==steps-max(100,spec['adaptations']['batch_size'])+1
    assert done['weights_changed'] and done['evaluation_frozen']
    by_reset={r['reset']:r for r in resets};previous=None;max_error=0.;refs_checked=0;input_excess_steps=0;failures=0
    episodes=[];current=dict(steps=0,total_cost=0.,solver_failures=0)
    for r in rows:
        assert r['executed_horizon']==h
        reset=by_reset[r['reset']];t=r['t'];s=r['state'];o=np.array(r['next_observation']);u=r['input']
        if previous is None or previous['reset']!=r['reset']:
            assert t==0 and r['previous_state']==reset['state_after_warmup']
            np.testing.assert_array_equal(r['observation'],reset['observation'])
            if previous is not None:assert previous['done'] or r['index']==steps+1
        else:
            assert not previous['done'] and t==previous['t']+1 and r['previous_state']==previous['state']
            np.testing.assert_array_equal(r['observation'],previous['next_observation'])
        expected=integrate(task,r['previous_state'],u)
        for key,value in expected.items():
            np.testing.assert_allclose(s[key],value,rtol=1e-7,atol=1e-7)
            max_error=max(max_error,abs(s[key]-value))
        if task=='vehicle':
            np.testing.assert_allclose(o[:3],[s['x']/30,s['y']/30,s['theta']/np.pi],rtol=0,atol=1e-12)
            cost=float(np.sum((o[3:5]*5)**2))
            violated=any(np.hypot(o[5+3*j]*10,o[6+3*j]*10)<=o[7+3*j] for j in range(3))
            excess=max(0.,-scalar(u['u_s']),scalar(u['u_s'])-5,abs(scalar(u['u_omega']))-4)
            limit=150;lam=.001;penalty=2
        else:
            np.testing.assert_allclose(o[:4],[s['pos']/1.5,s['v']/5,s['theta']/(np.pi/2),s['omega']/10],rtol=0,atol=1e-12)
            cost=.4*s['v']**2+.05*s['v']*s['omega']*math.cos(s['theta'])+(2/3)*.2*.25**2*s['omega']**2-.2*9.81*.25*math.cos(s['theta'])+10*(s['pos']-o[4])**2+.1*scalar(u['u1'])**2
            violated=abs(s['pos'])>1.5 or abs(s['theta'])>np.pi/2
            excess=max(0.,abs(scalar(u['u1']))-5);limit=100;lam=.003;penalty=10
        close(cost,r['performance'],'physical cost');close(lam*h,r['compute'],'H penalty')
        close(penalty*(limit-t-1) if violated else 0.,r['constraint'],'constraint cost')
        close(-r['reward'],r['performance']+r['compute']+r['constraint'],'reward decomposition')
        assert r['done']==(r['termination'] is not None) and t<limit
        assert violated==(r['termination']=='constraint')
        if r['termination']=='steps':assert t+1==limit
        if r['termination']=='goal':assert task=='vehicle'
        if not r['done']:assert t+1<limit
        assert excess<=1e-5
        input_excess_steps+=int(excess>0);failures+=int(not r['solver_success'])
        tvps=reset['available_tvps'];names=['pos_r'] if task=='pendulum' else ['trajectory_x','trajectory_y']+['obj_%d_%s'%(j,k) for j in range(3) for k in ('x','y','r')]
        if all(len(tvps[k])>t+2 for k in names):
            np.testing.assert_allclose(o,observation(task,s,dict(tvp=tvps),t+1),rtol=0,atol=1e-12);refs_checked+=1
        if r['index']<=steps:
            current['steps']+=1;current['total_cost']-=r['reward'];current['solver_failures']+=int(not r['solver_success'])
            if r['done']:
                episodes.append(dict(current,termination=r['termination']));current=dict(steps=0,total_cost=0.,solver_failures=0)
        previous=r
    saved=read(folder/'training_episodes.json');assert len(saved)==len(episodes)==done['train_episodes']
    for a,b in zip(saved,episodes):
        assert a['steps']==b['steps'] and a['termination']==b['termination'] and a['solver_failures']==b['solver_failures']
        close(a['total_cost'],b['total_cost'],'training episode sum')
    assert sum(e['steps'] for e in episodes)+current['steps']==steps
    result=dict(passed=True,task=task,seed=spec['seed'],fixed_h=h,training_steps=steps,logged_steps=len(rows),
        partial_final_training_episode=current,updates=done['updates'],logged_resets=len(resets),
        max_cross_integrator_state_error=max_error,reference_rows_checked_from_reset=refs_checked,
        solver_failure_steps=failures,input_roundoff_excess_steps=input_excess_steps,
        limitations='Later dynamically extended references unavailable in reset snapshot; for those rows physical cost uses recorded observation reference. Goal termination distance and solver success are not independently recomputed here. No new simulation using the study controller, no policy/test selection.',
        hashes={str(p):digest(p) for p in files+resfiles+[folder/'manifest.json',folder/'completed.json',folder/'instrumentation_completed.json',folder/'training_episodes.json']},
        source_hash=digest(Path(__file__)))
    write(folder/'independent_log_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('hashes','limitations')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);a=ap.parse_args();audit(a.folder)
