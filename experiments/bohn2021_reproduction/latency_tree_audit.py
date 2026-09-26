"""Independent saved-trace dynamics, measured-time, routing and counter audit."""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
from latency_tree_protocol import OUT, REG, ROOT, TASKS, read, write, sha, verify
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from conservative_fixed_log_audit import integrate
from latency_tree_policy import features


def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)


def clean(trace):
    result=copy.deepcopy(trace)
    for row in result:
        row.pop('timing',None)
        for attempt in row['recovery']['attempts']:attempt.pop('solver_s',None)
    return result


def expected_h(policy,values,state):
    kind=policy['kind']
    if kind=='constant':return policy['h']
    if kind in ('certificate','combined'):
        theta,omega=state['theta'],state['omega']
        on=math.atan((5.+1e-5)/(.8*9.81))+1e-10<abs(theta)<math.pi/2-1e-10 and theta*omega>=0
        if on:return 5
        if kind=='certificate':return 30
        return expected_h(policy['learned'],values,state)
    assert kind=='tree'
    node=0
    for depth in range(2):
        split=policy['nodes'][node]
        node=node*2+1+int(values[split['feature']]>split['threshold'])
    return policy['leaves'][node-3]


def condition(folder):
    done=read(folder/'completed.json');assert done['passed']
    for p,h in done['hashes'].items():assert sha(Path(p))==h,p
    s=read(folder/'summary.json');task=s['task'];policy=s['policy'];cases=read(Path(s['bank']))['cases']
    assert len(s['episodes'])==len(cases)*s['repeats']
    assert [(e['repeat'],e['case']) for e in s['episodes']]==[(r,c) for r in range(s['repeats']) for c in range(len(cases))]
    calls=[json.loads(line) for line in (folder/'solver_calls.jsonl').read_text().splitlines()]
    counts=read(folder/'solver_attempts.json');assert counts==s['solver_counts']
    assert len(calls)==counts['solve_attempts']==counts['solve_completed']
    cursor=0;total_steps=0;cache={};max_error=0.;initial_states=[]
    for ep in s['episodes']:
        repeat,cid=ep['repeat'],ep['case'];path=folder/('r%d_trace_%02d.json'%(repeat,cid));trace=read(path)
        assert trace==[json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        audit_trace(task,cases[cid],trace);audit_context(task,cases[cid],trace,0)
        assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid and calls[cursor]['step']==-1;cursor+=1
        reset=read(folder/('r%d_reset_%02d.json'%(repeat,cid)))
        assert reset['reset_gross_s']>=reset['controller_gross_s']>reset['logging_s']>0
        close(reset['controller_s'],reset['controller_gross_s']-reset['logging_s'])
        initial_states.append(dict(repeat=repeat,case=cid,state=trace[0]['previous_state'],input=trace[0]['policy_context']['previous_input']))
        for step,row in enumerate(trace):
            ctx=row['policy_context']
            assert ctx['previous_initial_failure']==(not trace[step-1]['recovery']['attempts'][0]['success'] if step else False)
            assert ctx['previous_final_failure']==(not trace[step-1]['solver_success'] if step else False)
            # Feature formula also has independent synthetic known-answer checks;
            # here audit causal source context then verify persisted transforms.
            np.testing.assert_array_equal(row['tree_features'],features(task,ctx))
            assert row['horizon']==expected_h(policy,row['tree_features'],row['previous_state'])
            attempts=row['recovery']['attempts'];assert calls[cursor:cursor+len(attempts)]==attempts;cursor+=len(attempts)
            assert 1<=len(attempts)<=3 and [a['kind'] for a in attempts]==['initial','retry_zero','retry_previous'][:len(attempts)]
            for i,a in enumerate(attempts):
                assert a['case']==cid and a['step']==step
                accepted=a['finite'] and a['success'] and max(a['constraint_residual'],a['bound_residual'])<=1e-5
                assert a['accepted']==accepted
                if i<len(attempts)-1:assert not accepted
            assert len(attempts)==3 or attempts[-1]['accepted']
            event=row['recovery'];assert event['restored_original']==(not attempts[-1]['accepted'])
            assert event['recovered']==(not attempts[0]['accepted'] and attempts[-1]['accepted'])
            assert event['final_success']==row['solver_success']==(attempts[0]['success'] if event['restored_original'] else attempts[-1]['success'])
            tm=row['timing'];assert all(math.isfinite(v) and v>0 for v in tm.values())
            close(tm['controller_s'],tm['controller_gross_s']-tm['logging_s'])
            close(tm['decision_s'],tm['controller_s']+tm['selection_s'])
            close(tm['decision_gross_s'],tm['decision_s']+tm['logging_s'])
            assert tm['controller_s']>=sum(a['solver_s'] for a in attempts)-1e-10
            key=json.dumps([row['previous_state'],row['input']],sort_keys=True)
            if key not in cache:cache[key]=integrate(task,row['previous_state'],row['input'])
            for name,value in cache[key].items():
                np.testing.assert_allclose(row['state'][name],value,rtol=1e-7,atol=1e-7)
                max_error=max(max_error,abs(row['state'][name]-value))
        expected=dict(steps=len(trace),success=trace[-1]['termination']==('goal' if task=='vehicle' else 'steps'),
            constraint=trace[-1]['termination']=='constraint',termination=trace[-1]['termination'],
            total_cost=sum(-r['reward'] for r in trace),performance_cost=sum(r['performance'] for r in trace),
            h_penalty=sum(r['compute'] for r in trace),constraint_cost=sum(r['constraint'] for r in trace),
            physical_constraint_cost=sum(r['performance'] for r in trace)+sum(r['constraint'] for r in trace),
            solver_failure_steps=sum(not r['solver_success'] for r in trace),initial_failed_steps=sum(not r['recovery']['attempts'][0]['success'] for r in trace),
            retries=sum(len(r['recovery']['attempts'])-1 for r in trace),
            recovered_steps=sum(r['recovery']['recovered'] for r in trace),switches=sum(a['horizon']!=b['horizon'] for a,b in zip(trace,trace[1:])),
            mean_horizon=sum(r['horizon'] for r in trace)/len(trace),decision_total_s=sum(r['timing']['decision_s'] for r in trace),
            decision_mean_s=sum(r['timing']['decision_s'] for r in trace)/len(trace),gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),
            logging_total_s=sum(r['timing']['logging_s'] for r in trace),decision_p95_s=float(np.percentile([r['timing']['decision_s'] for r in trace],95)),
            deadline_exceed_steps=sum(r['timing']['decision_s']>(.1 if task=='vehicle' else .04) for r in trace),case=cid,repeat=repeat)
        assert set(expected)==set(ep)
        for key,value in expected.items():
            if isinstance(value,float):close(ep[key],value)
            else:assert ep[key]==value,(key,ep[key],value)
        if repeat:assert clean(trace)==clean(read(folder/('r0_trace_%02d.json'%cid)))
        if s['split']=='select' and folder.name.startswith('r1_'):
            first=folder.with_name('r0_'+folder.name[3:])/path.name
            assert clean(trace)==clean(read(first))
        total_steps+=len(trace)
    meters=[read(p) for p in folder.glob('attempt_*.json')]
    assert sum(m['step_calls'] for m in meters)==total_steps==s['steps']
    assert sum(m['reset_calls'] for m in meters)==len(s['episodes'])==s['resets']==counts['warmup_attempts']
    assert cursor==len(calls) and counts['retry_attempts']==sum(e['retries'] for e in s['episodes'])
    assert len(calls)==total_steps+len(s['episodes'])+counts['retry_attempts']
    assert s['logging_operations']==1+3*len(calls)
    return dict(task=task,seed=s['seed'],folder=str(folder),episodes=len(s['episodes']),steps=total_steps,
                solves=len(calls),retries=counts['retry_attempts'],independent_integrations=len(cache),max_dynamics_error=max_error,
                starts=initial_states)


def main(phase):
    verify();checks=read(OUT/'synthetic_checks.json');assert checks['passed']
    for p,h in checks['source_hashes'].items():assert sha(Path(p))==h
    output=OUT/('audit_'+phase+'.json')
    if output.exists():
        result=read(output)
        for p,h in result['hashes'].items():assert sha(Path(p))==h
        assert result['passed']; print('Existing '+phase+' audit verified');return
    if phase=='smoke':
        done=read(OUT/'smoke/completed.json');assert done['passed']
        folders=sorted(p.parent for p in (OUT/'smoke').glob('*/completed.json'));assert len(folders)==12
    else:
        status=read(OUT/'status.json');assert status['complete'] and not status['active']
        assert not (Path('/proc')/str(status['pid'])/'cmdline').exists()
        folders=sorted(p.parent for p in (OUT/'train').rglob('summary.json'))
        assert len(folders)>=318
    rows=[];starts={};hashes={str(REG):sha(REG),str(OUT/'synthetic_checks.json'):sha(OUT/'synthetic_checks.json'),str(Path(__file__).resolve()):sha(Path(__file__))}
    for folder in folders:
        result=condition(folder);key=(result['task'],result['seed'],read(folder/'summary.json')['split'])
        normalized=[s for s in result['starts'] if s['repeat']==0]
        if key in starts:assert normalized==starts[key]
        else:starts[key]=normalized
        result.pop('starts');rows.append(result);hashes[str(folder/'completed.json')]=sha(folder/'completed.json')
        print(json.dumps(result),flush=True)
    result=dict(passed=True,phase=phase,conditions=len(rows),episodes=sum(r['episodes'] for r in rows),
                steps=sum(r['steps'] for r in rows),solves=sum(r['solves'] for r in rows),retries=sum(r['retries'] for r in rows),
                independent_integrations=sum(r['independent_integrations'] for r in rows),max_dynamics_error=max(r['max_dynamics_error'] for r in rows),
                groups=rows,hashes=hashes,test_accessed=False,scope='Saved trace/cost/causal-context/formula/routing/dynamics/timing/counter audit; training efficacy is not independent evidence.')
    write(output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('hashes','groups')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','train'),required=True)
    main(parser.parse_args().phase)
