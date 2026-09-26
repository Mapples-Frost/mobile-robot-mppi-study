"""Independent bookkeeping, replay, dynamics and reporting of recovery diagnosis."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
from conservative_recovery_diagnosis import OUT,PREVIOUS,register,TASKS,BASE,BLOCK
from conservative_iteration_evaluate import arm_name
from conservative_policy_model import choose
from conservative_iteration_audit import audit_context
from branch_calibration_audit import audit_trace
from conservative_fixed_log_audit import integrate
from fixed_policy_branches import metrics
from paper_h_soft_probe import read,digest
from run import write


def main(smoke=False):
    register();phase='smoke' if smoke else 'formal'
    jobs=[('vehicle',2,f) for f in ('primary','round0')]+[('pendulum',0,f) for f in ('primary','round0')] if smoke else [(t,s,f) for t in TASKS for s in range(3) for f in ('primary','round0','round1')]
    rows=[];groups=[];hashes={};steps=0;max_error=0.
    for task,seed,family in jobs:
        h=BASE[task] if family=='primary' else 0;arm=arm_name(family,seed,h);folder=OUT/phase/task/arm
        completed=read(folder/'completed.json');assert completed['passed']
        for p,value in completed['hashes'].items():assert digest(Path(p))==value
        bank=read(PREVIOUS/'banks'/(task+'_validation.json'))['cases']
        policy=read(PREVIOUS/('%s_s%d_r%d'%(task,seed,int(family[-1])))/'policy.json') if family!='primary' else None
        summary=read(folder/'summary.json');calls=[json.loads(l) for l in (folder/'solver_calls.jsonl').read_text().splitlines()]
        counts=read(folder/'solver_attempts.json');assert counts==summary['solver_counts']
        assert len(calls)==counts['solve_completed']==counts['solve_attempts']
        assert sum(c['kind']=='warmup' for c in calls)==counts['warmup_attempts']
        assert sum(c['kind'].startswith('retry') for c in calls)==counts['retry_attempts']
        ids=[2 if task=='vehicle' else 0] if smoke else list(range(24))
        assert [(e['repeat'],e['case']) for e in summary['episodes']]==[(r,c) for r in range(2 if smoke else 1) for c in ids]
        cursor=0;cache={}
        for e in summary['episodes']:
            cid=e['case'];repeat=e['repeat'];trace=read(folder/('repeat%d_trace_%02d.json'%(repeat,cid)))
            old=read(PREVIOUS/'evaluations/validation'/task/arm/('trace_%02d.json'%cid))
            audit_trace(task,bank[cid],trace);audit_context(task,bank[cid],trace,0)
            assert all(e[k]==v for k,v in metrics(task,trace).items())
            assert e['original_metrics']==metrics(task,old)
            assert calls[cursor]['kind']=='warmup' and calls[cursor]['step']==-1 and calls[cursor]['case']==cid;cursor+=1
            for t,row in enumerate(trace):
                if t%BLOCK==0:selected=choose(policy,row['policy_context']['features']) if policy else BASE[task]
                assert selected==row['horizon']
                event=row['recovery'];attempts=event['attempts']
                assert 1<=len(attempts)<=3 and [a['kind'] for a in attempts]==['initial','retry_zero','retry_previous'][:len(attempts)]
                assert calls[cursor:cursor+len(attempts)]==attempts;cursor+=len(attempts)
                for index,a in enumerate(attempts):
                    assert a['case']==cid and a['step']==t
                    accepted=a['finite'] and a['success'] and max(a['constraint_residual'],a['bound_residual'])<=1e-5
                    assert bool(a['accepted'])==bool(accepted)
                    if index<len(attempts)-1:assert not accepted
                assert len(attempts)==3 or attempts[-1]['accepted']
                assert event['restored_original']==(not attempts[-1]['accepted'])
                assert event['recovered']==(not attempts[0]['accepted'] and attempts[-1]['accepted'])
                assert row['solver_success']==event['final_success']==(attempts[0]['success'] if event['restored_original'] else attempts[-1]['success'])
                key=json.dumps([row['previous_state'],row['input']],sort_keys=True)
                if key not in cache:cache[key]=integrate(task,row['previous_state'],row['input'])
                for name,value in cache[key].items():
                    actual=row['state'][name];np.testing.assert_allclose(actual,value,rtol=1e-7,atol=1e-7);max_error=max(max_error,abs(actual-value))
            if not any(r['recovery']['recovered'] for r in trace):
                assert [{k:v for k,v in r.items() if k!='recovery'} for r in trace]==old,'Unsuccessful retries changed trajectory'
            computed=dict(initial_solver_failed_steps=sum(not r['recovery']['attempts'][0]['success'] for r in trace),
                initial_unusable_steps=sum(not r['recovery']['attempts'][0]['accepted'] for r in trace),
                recovered_steps=sum(r['recovery']['recovered'] for r in trace),retry_attempts=sum(len(r['recovery']['attempts'])-1 for r in trace))
            assert all(e[k]==v for k,v in computed.items());steps+=len(trace)
            rows.append(dict(task=task,seed=seed,family=family,case=cid,repeat=repeat,total_cost=e['total_cost'],
                original_cost=e['original_metrics']['total_cost'],cost_difference=e['total_cost']-e['original_metrics']['total_cost'],
                success=e['success'],original_success=e['original_metrics']['success'],constraint=e['constraint'],
                original_constraint=e['original_metrics']['constraint'],final_solver_failures=e['solver_failure_steps'],
                original_solver_failures=e['original_metrics']['solver_failure_steps'],steps=e['steps'],**computed))
        assert cursor==len(calls)
        attempts=[read(p) for p in folder.glob('attempt_*.json')]
        assert sum(a['step_calls'] for a in attempts)==sum(e['steps'] for e in summary['episodes'])
        assert sum(a['reset_calls'] for a in attempts)==len(summary['episodes'])
        groups.append(dict(task=task,seed=seed,family=family,episodes=len(summary['episodes']),counts=counts,numerical_integrations=len(cache)))
        hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    dest=OUT/('audit_'+phase);dest.mkdir(exist_ok=True)
    with (dest/'all_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(passed=True,phase=phase,conditions=len(jobs),episodes=len(rows),recorded_steps=steps,groups=groups,
        maximum_independent_dynamics_error=max_error,hashes=hashes,source_hash=digest(Path(__file__)),
        limits='Same exposed scenarios; neither new validation nor a test. Original failure and added retry counts retained. Cached independent single-step integration does not re-solve NLP or recheck its residual function. No acceleration claim.')
    write(dest/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('hashes','groups')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
