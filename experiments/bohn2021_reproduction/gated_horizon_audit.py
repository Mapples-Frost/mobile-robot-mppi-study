"""Independent dynamics, bookkeeping and exhaustive selection audit of gate search."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import candidates,decide
from conservative_iteration_audit import audit_context
from branch_calibration_audit import audit_trace
from conservative_fixed_log_audit import integrate
from fixed_policy_branches import metrics
from paper_h_soft_probe import read,digest
from run import write


def audit_candidate(folder,task,cases,baseline=None,smoke=False,fixed_h=None):
    completed=read(folder/'completed.json');r=completed['result'];policy=r['policy']
    for p,h in completed['hashes'].items():assert digest(Path(p))==h
    calls=[json.loads(line) for line in (folder/'solver_calls.jsonl').read_text().splitlines()]
    counts=read(folder/'solver_attempts.json')
    assert len(calls)==counts['solve_attempts']==counts['solve_completed']
    assert sum(c['kind']=='warmup' for c in calls)==counts['warmup_attempts']
    assert sum(c['kind'].startswith('retry') for c in calls)==counts['retry_attempts']
    episodes=r['episodes'];assert episodes
    expected=[(rep,cid) for rep in range(2 if smoke else 1) for cid in range(len(cases))]
    assert [(e['repeat'],e['case']) for e in episodes]==expected[:len(episodes)]
    assert r['fully_evaluated']==(len(episodes)==len(expected))
    assert r['unrun_cases']==([] if smoke else list(range(len(episodes),len(cases))))
    cursor=0;steps=0;cache={};max_error=0.;rejections=[]
    for e in episodes:
        cid=e['case'];rep=e['repeat'];trace=read(folder/('r%d_trace_%02d.json'%(rep,cid)))
        audit_trace(task,cases[cid],trace);audit_context(task,cases[cid],trace,0)
        m=metrics(task,trace)
        assert all(e[k]==v for k,v in m.items())
        assert e['physical_constraint_cost']==m['performance_cost']+m['constraint_cost']
        assert e['initial_failed_steps']==sum(not x['recovery']['attempts'][0]['success'] for x in trace)
        assert e['retries']==sum(len(x['recovery']['attempts'])-1 for x in trace)
        assert e['recovered_steps']==sum(x['recovery']['recovered'] for x in trace)
        assert e['switches']==sum(a['horizon']!=b['horizon'] for a,b in zip(trace,trace[1:]))
        assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid and calls[cursor]['step']==-1;cursor+=1
        for t,row in enumerate(trace):
            h,gate=decide(policy,row['policy_context']) if fixed_h is None else (fixed_h,dict(use_short=False,reason='fixed',fixed_h=fixed_h))
            assert h==row['horizon']
            assert set(gate)==set(row['gate'])
            for name,value in gate.items():
                if name=='values':
                    assert set(value)==set(row['gate'][name])
                    for key,v in value.items():np.testing.assert_allclose(v,row['gate'][name][key],rtol=1e-13,atol=1e-13)
                else:assert value==row['gate'][name]
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
        if rep:
            def clean(data):
                d=copy.deepcopy(data)
                for row in d:
                    for a in row['recovery']['attempts']:a.pop('solver_s')
                return d
            assert clean(trace)==clean(read(folder/('r0_trace_%02d.json'%cid)))
        if baseline is not None and not smoke:
            b=baseline[cid];reasons=[]
            if e['success']<b['success']:reasons.append('success')
            if e['constraint']>b['constraint']:reasons.append('constraint')
            if e['initial_failed_steps']>b['initial_failed_steps']:reasons.append('initial_solver_failures')
            if e['solver_failure_steps']>b['solver_failure_steps']:reasons.append('final_solver_failures')
            if e['physical_constraint_cost']>b['physical_constraint_cost']+.05*abs(b['physical_constraint_cost'])+1e-8:reasons.append('per_case_physical_cost')
            if reasons:
                rejections.append(dict(case=cid,reasons=reasons));assert e==episodes[-1]
        steps+=len(trace)
    assert r['rejected']==rejections
    assert cursor==len(calls)
    assert r['mean_raw_cost']==float(np.mean([e['total_cost'] for e in episodes]))
    assert r['mean_physical_cost']==float(np.mean([e['physical_constraint_cost'] for e in episodes]))
    attempts=[read(p) for p in folder.glob('attempt_*.json')]
    assert sum(a['step_calls'] for a in attempts)==steps
    assert sum(a['reset_calls'] for a in attempts)==len(episodes)
    return dict(policy=policy,episodes=len(episodes),steps=steps,counts=counts,numerical_integrations=len(cache),maximum_dynamics_error=max_error,hash=digest(folder/'completed.json'))


def main(smoke=False):
    freeze();groups=[];phase='smoke' if smoke else 'train'
    for task in TASKS:
        for seed in ([0] if smoke else range(3)):
            root=OUT/phase/('%s_s%d'%(task,seed))
            bank=OUT/'banks'/('%s_%s_bank.json'%(task,'smoke' if smoke else 'train_s%d'%seed))
            assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
            cases=read(bank)['cases'];results=[];base=None
            policies=[dict(id='fixed',task=task)]+([c for c in candidates(task) if c['id']=='h5_p2_g5'] if smoke else candidates(task))
            for policy in policies:
                folder=root/policy['id'];r=read(folder/'completed.json')['result'];assert r['policy']==policy
                audit=audit_candidate(folder,task,cases,base,smoke);audit.update(task=task,seed=seed);groups.append(audit);results.append(r)
                if policy['id']=='fixed':base=r['episodes']
            if smoke:
                done=read(root/'smoke_completed.json');assert done['passed'] and done['results']==results
            else:
                done=read(root/'fit_completed.json');assert done['results']==results
                b=results[0];eligible=[r for r in results if r['fully_evaluated'] and not r['rejected'] and r['mean_physical_cost']<=b['mean_physical_cost']+.02*abs(b['mean_physical_cost'])]
                selected=min(eligible,key=lambda r:(r['mean_raw_cost'],r['policy']['id']!='fixed',r['policy']['id']))['policy']
                assert done['selected']==selected and read(root/'policy.json')==dict(selected,training_seed=seed)
                assert done['policy_hash']==digest(root/'policy.json') and done['training_bank_hash']==digest(bank)
                assert done['gradient_updates']==0 and not done['validation_access'] and not done['test_access']
    result=dict(passed=True,phase=phase,groups=groups,episodes=sum(g['episodes'] for g in groups),recorded_steps=sum(g['steps'] for g in groups),
        maximum_dynamics_error=max(g['maximum_dynamics_error'] for g in groups),source_hash=digest(Path(__file__)),
        scope='Recomputes raw costs and constraints, independently integrates dynamics, validates every horizon and retry, durable budgets and complete training selection. No new optimizer solves; no evidence of independent efficacy.')
    write(OUT/('audit_'+phase+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
