"""Independent horizon formulas, dynamics, cost and durable-budget reconciliation."""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
from failure_state_protocol import ROOT, OLD, OUT, REG, TASKS, read, write, sha, verify, bank, jobs
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from conservative_fixed_log_audit import integrate
from fixed_policy_branches import metrics
from gated_horizon_training_angular_diagnosis import model, boundary_checks


def expected(task,policy,ctx,original):
    if policy.startswith('fixed'):
        return int(policy[5:]),False
    if task=='vehicle':
        # Unmodified selected policy must exactly replay its previously audited trajectory.
        return original['horizon'],original['gate']['use_short']
    state=ctx['state'];angle=abs(state['theta'])
    outward=state['theta']*state['omega']>=0
    threshold=math.atan((5+1e-5)/(.8*9.81))
    enabled=angle>threshold+1e-10 and angle<math.pi/2-1e-10 and outward
    return int(policy.replace('certificate','')) if enabled else 30,bool(enabled)


def clean(trace):
    rows=copy.deepcopy(trace)
    for row in rows:
        row.pop('gate',None)
        for a in row['recovery']['attempts']:a.pop('solver_s',None)
    return rows


def main(phase):
    spec=verify();m=model();assert boundary_checks(m)
    output=OUT/('audit_'+phase+'.json')
    if output.exists():
        done=read(output)
        for p,h in done['hashes'].items():assert sha(Path(p))==h
        assert done['passed'];print('Existing audit verified');return
    hashes={str(REG):sha(REG)};groups=[]
    for task,seed,policy in jobs():
        folder=OUT/phase/('%s_s%d_%s'%(task,seed,policy));completion=folder/'completed.json'
        done=read(completion);assert done['passed'];hashes[str(completion)]=sha(completion)
        for p,h in done['hashes'].items():assert sha(Path(p))==h
        summary=read(folder/'summary.json');trace_folder=Path(summary['trace_folder']);reused=summary['reused']
        assert reused==done['reused'] and (task,seed,policy,phase)==tuple(summary[k] for k in ('task','seed','policy','phase'))
        cases=read(bank(task,seed))['cases'];episodes=summary['episodes']
        ids=spec['smoke_cases'][task][str(seed)] if phase=='smoke' else list(range(24))
        repeats=2 if phase=='smoke' else 1
        assert [(e['repeat'],e['case']) for e in episodes]==[(r,c) for r in range(repeats) for c in ids]
        calls=[json.loads(line) for line in (trace_folder/'solver_calls.jsonl').read_text().splitlines()]
        counts=read(trace_folder/'solver_attempts.json')
        assert len(calls)==counts['solve_attempts']==counts['solve_completed']
        assert counts['warmup_attempts']==len(episodes)
        assert sum(x['kind'].startswith('retry') for x in calls)==counts['retry_attempts']
        cursor=0;steps=0;cache={};error=0.;triggered=0
        selected=read(OLD/'train'/('vehicle_s%d'%seed)/'policy.json') if task=='vehicle' else None
        for e in episodes:
            cid,repeat=e['case'],e['repeat'];p=trace_folder/('r%d_trace_%02d.json'%(repeat,cid));trace=read(p)
            assert str(p) in done['hashes'] and sha(p)==done['hashes'][str(p)]
            audit_trace(task,cases[cid],trace);audit_context(task,cases[cid],trace,0)
            calculated=metrics(task,trace)
            assert all(e[k]==v for k,v in calculated.items())
            assert e['physical_constraint_cost']==calculated['performance_cost']+calculated['constraint_cost']
            initial=sum(not r['recovery']['attempts'][0]['success'] for r in trace)
            retries=sum(len(r['recovery']['attempts'])-1 for r in trace)
            switches=sum(a['horizon']!=b['horizon'] for a,b in zip(trace,trace[1:]))
            assert (initial,retries,switches)==(e['initial_failed_steps'],e['retries'],e['switches'])
            assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid and calls[cursor]['step']==-1;cursor+=1
            primary=read(OLD/'train'/('%s_s%d'%(task,seed))/'fixed'/('r0_trace_%02d.json'%cid))
            assert trace[0]['previous_state']==primary[0]['previous_state']
            original=read(OLD/'train'/('vehicle_s%d'%seed)/selected['id']/('r0_trace_%02d.json'%cid)) if policy=='selected' else None
            episode_triggered=0
            for step,row in enumerate(trace):
                h,on=expected(task,policy,row['policy_context'],original[step] if original else None)
                assert h==row['horizon']
                if not reused:assert row['gate']['triggered']==on
                episode_triggered+=on
                event=row['recovery'];attempts=event['attempts']
                assert 1<=len(attempts)<=3 and [a['kind'] for a in attempts]==['initial','retry_zero','retry_previous'][:len(attempts)]
                assert calls[cursor:cursor+len(attempts)]==attempts;cursor+=len(attempts)
                for i,a in enumerate(attempts):
                    accepted=a['finite'] and a['success'] and max(a['constraint_residual'],a['bound_residual'])<=1e-5
                    assert a['case']==cid and a['step']==step and a['accepted']==accepted
                    if i<len(attempts)-1:assert not accepted
                assert len(attempts)==3 or attempts[-1]['accepted']
                assert event['restored_original']==(not attempts[-1]['accepted'])
                assert event['recovered']==(not attempts[0]['accepted'] and attempts[-1]['accepted'])
                assert event['final_success']==row['solver_success']==(attempts[0]['success'] if event['restored_original'] else attempts[-1]['success'])
                # Original inherited traces are covered by the frozen all-candidate audit.
                if not reused:
                    key=json.dumps([row['previous_state'],row['input']],sort_keys=True)
                    if key not in cache:cache[key]=integrate(task,row['previous_state'],row['input'])
                    for name,value in cache[key].items():
                        np.testing.assert_allclose(row['state'][name],value,rtol=1e-7,atol=1e-7)
                        error=max(error,abs(row['state'][name]-value))
            if not reused:assert episode_triggered==e['intervention_steps']
            if policy=='selected':assert clean(trace)==clean(original)
            if policy==('fixed30' if task=='pendulum' else 'fixed25'):assert clean(trace)==clean(primary)
            if repeat:assert clean(trace)==clean(read(trace_folder/('r0_trace_%02d.json'%cid)))
            steps+=len(trace);triggered+=episode_triggered
        assert cursor==len(calls)
        meters=[read(p) for p in trace_folder.glob('attempt_*.json')]
        assert sum(a['step_calls'] for a in meters)==steps and sum(a['reset_calls'] for a in meters)==len(episodes)
        if reused:
            assert phase=='full' and summary['new_step_calls']==summary['new_reset_calls']==summary['new_solver_calls']==0
        else:
            assert summary['new_step_calls']==steps and summary['new_reset_calls']==len(episodes)
            assert summary['new_solver_calls']==counts['solve_attempts']
        groups.append(dict(task=task,seed=seed,policy=policy,reused=reused,episodes=len(episodes),recorded_steps=steps,
                           new_steps=0 if reused else steps,new_resets=0 if reused else len(episodes),
                           new_solver_calls=0 if reused else counts['solve_attempts'],intervention_steps=triggered,
                           new_numerical_integrations=len(cache),maximum_dynamics_error=error))
    for name,value in hashes.items():assert sha(Path(name))==value
    report=dict(passed=True,phase=phase,post_hoc=True,groups=groups,hashes=hashes,
                episodes=sum(g['episodes'] for g in groups),recorded_steps=sum(g['recorded_steps'] for g in groups),
                new_control_steps=sum(g['new_steps'] for g in groups),new_resets=sum(g['new_resets'] for g in groups),
                new_solver_calls=sum(g['new_solver_calls'] for g in groups),new_numerical_integrations=sum(g['new_numerical_integrations'] for g in groups),
                maximum_dynamics_error=max(g['maximum_dynamics_error'] for g in groups),test_accessed=False,
                scope='Audit of descriptive reused-training diagnosis. No new learner, validation or confirmation. Original reused trajectories inherit frozen all-candidate dynamics audit.')
    assert report['episodes']==(72 if phase=='smoke' else 576)
    write(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ('groups','hashes')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('smoke','full'),required=True);a=ap.parse_args();main(a.phase)
