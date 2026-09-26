"""Post-hoc closed-loop recovery diagnosis on all exposed validation scenarios.

Every selected primary and both frozen rounds, all seeds and both tasks. This is
not fresh validation; any later performance claim requires new scenario banks.
"""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from runtime import ART,ROOT,imports
from conservative_iteration import OUT as PREVIOUS,TASKS,BASE,BLOCK,verify
from conservative_canonical_reset import make_env
from conservative_policy_model import choose
from conservative_iteration_evaluate import arm_name
from conservative_iteration_audit import audit_context
from conservative_solver_recovery import install
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from fixed_policy_branches import metrics
from relative_policy_features import context
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read,digest
from run import write,weights_hash

OUT=ART/'results/solver_recovery_diagnosis_2026-09-25'


def register():
    verify();assert read(PREVIOUS/'validation_finish_status.json')['complete']
    OUT.mkdir(exist_ok=True)
    files=[Path(__file__).resolve(),Path(__file__).with_name('conservative_solver_recovery.py').resolve(),
        PREVIOUS/'solver_initialization_probe/array_audit.json',PREVIOUS/'validation_selection.json']
    for task in TASKS:
        files.append(PREVIOUS/'banks'/(task+'_validation.json'))
        for seed in range(3):
            files.append(model_dir(task,'fixed',seed)/'model.zip')
            for r in range(2):files.append(PREVIOUS/('%s_s%d_r%d'%(task,seed,r))/'policy.json')
    spec=dict(scope='Post-hoc mechanism diagnosis on exposed previous validation; never independent efficacy evidence.',
        conditions='2 tasks x3 seeds x(primary,round0,round1) xall24 previously exposed scenarios; no scene exclusions.',
        intervention='On solver failure or primal residual>1e-5, retry same NLP once with current-state constant and zero input; if still unusable retry once with current-state constant and previous input. Zero slack/algebraic initial guess. Accept first successful finite primal solution with residual<=1e-5. If neither usable restore original solution exactly.',
        unchanged='Policy, terminal, horizon hold, dynamics, costs, constraints, solver tolerances and scenario. No refitting. Original reset warmup has recovery disabled for same starting state.',
        fairness='Identical recovery enabled for all fixed and adaptive arms; initial failures, retries, final failures and added solve attempts separately counted.',
        metrics=['total_cost','physical_constraint_cost','success','constraint','initial_solver_failed_steps','initial_unusable_steps','recovered_steps','final_solver_failed_steps','raw_solve_attempts'],
        acceptance='Diagnose whether recovery removes the demonstrated seed2 failure and whether it helps or harms all other conditions. No success declaration or test unlock.',
        control='Every trace before first retry equals original trace exactly; no-retry episodes equal complete original. Smoke repeats full episodes and compares all nontiming fields.',
        smoke='vehicle seed2 case2 and pendulum seed0 case0; primary and round0, each twice. Same disclosed historical diagnostic scenes, no new selection.',
        timing='Recorded solver durations diagnostic only; parallel wall times do not support speed claims.',
        budget='Max432 episodes x150 steps, max2 retries per scored step; actual reset/step and raw solver attempts durably counted including failures. Constructor initialization separate.',
        workers=2,test_access=False,hashes={str(p):digest(p) for p in files})
    p=OUT/'protocol.json'
    if p.exists():
        registered=read(p)
        if registered!=spec:
            receipt=read(OUT/'restore_type_fix/source_amendment.json')
            for name,change in receipt['changed_sources'].items():
                assert registered['hashes'][name]==change['old'] and spec['hashes'][name]==change['new']
                spec['hashes'][name]=change['old']
        assert registered==spec
    else:write(p,spec)


def job(task,seed,family,smoke=False):
    register();h=BASE[task] if family=='primary' else 0;arm=arm_name(family,seed,h)
    dest=OUT/('smoke' if smoke else 'formal')/task/arm;dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'completed.json').exists():
        for p,hsh in read(dest/'completed.json')['hashes'].items():assert digest(Path(p))==hsh
        return
    assert not (dest/'solver_attempts.json').exists(),'Inspect interruption before rerun'
    bank=PREVIOUS/'banks'/(task+'_validation.json');cases=read(bank)['cases']
    env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,dest)
    source=model_dir(task,'fixed',seed);_,SAC,_=imports();model=SAC.load(str(source/'model.zip'))
    assert weights_hash(model)==read(source/'completed.json')['final_hash']
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
    policy_path=PREVIOUS/('%s_s%d_r%d'%(task,seed,int(family[-1])))/'policy.json' if family!='primary' else None
    policy=read(policy_path) if policy_path else None
    state=install(env.control_system.controller.mpc,dest)
    ids=[2 if task=='vehicle' else 0] if smoke else list(range(24))
    summaries=[];hashes={str(OUT/'protocol.json'):digest(OUT/'protocol.json')};first_traces={}
    for repeat in range(2 if smoke else 1):
        for cid in ids:
            case=cases[cid];state.update(enabled=False,case=cid,step=-1,events=[])
            env.reset(**copy.deepcopy(case));state['enabled']=True
            original_path=PREVIOUS/'evaluations/validation'/task/arm/('trace_%02d.json'%cid)
            old=read(original_path);trace=[];diverged=False
            for t in range(env.max_steps):
                state['step']=t;ctx=context(env,task)
                if t%BLOCK==0:selected=choose(policy,ctx['features']) if policy else BASE[task]
                _,done,row=observed_step(env,task,selected,case,t)
                event=state['events'][-1];assert len(state['events'])==t+1
                row['policy_context']=ctx
                if not diverged and len(event['attempts'])==1:assert t<len(old) and row==old[t],(task,seed,family,cid,t,'unmodified-prefix mismatch')
                elif not diverged:
                    assert row['previous_state']==old[t]['previous_state'] and ctx==old[t]['policy_context']
                diverged|=len(event['attempts'])>1
                row['recovery']=event;trace.append(row)
                if done:break
            assert done;state['enabled']=False
            audit_trace(task,case,trace);audit_context(task,case,trace,0)
            def strip(data):
                rows=copy.deepcopy(data)
                for r in rows:
                    for attempt in r['recovery']['attempts']:attempt.pop('solver_s')
                return rows
            if repeat:assert strip(trace)==strip(first_traces[cid]),'Smoke recovery repeat changed'
            else:first_traces[cid]=trace
            path=dest/('repeat%d_trace_%02d.json'%(repeat,cid));write(path,trace);hashes[str(path)]=digest(path);hashes[str(original_path)]=digest(original_path)
            m=metrics(task,trace);m.update(case=cid,repeat=repeat,physical_constraint_cost=m['performance_cost']+m['constraint_cost'],
                initial_solver_failed_steps=sum(not e['attempts'][0]['success'] for e in state['events']),
                initial_unusable_steps=sum(not e['attempts'][0]['accepted'] for e in state['events']),
                recovered_steps=sum(e['recovered'] for e in state['events']),
                retry_attempts=sum(len(e['attempts'])-1 for e in state['events']),
                original_metrics=metrics(task,old),exact_until_first_retry=True)
            summaries.append(m);print(json.dumps(dict(task=task,seed=seed,family=family,case=cid,repeat=repeat,cost=m['total_cost'],recovered=m['recovered_steps'])),flush=True)
    write(dest/'summary.json',dict(task=task,seed=seed,family=family,smoke=smoke,episodes=summaries,solver_counts=state['counts']))
    for p in [dest/'summary.json',dest/'solver_attempts.json',dest/'solver_calls.jsonl',bank,source/'model.zip']:
        hashes[str(p)]=digest(p)
    if policy_path:hashes[str(policy_path)]=digest(policy_path)
    write(dest/'completed.json',dict(passed=True,smoke=smoke,hashes=hashes,efficacy_claim=False))


def suite():
    register()
    for task,seed in [('vehicle',2),('pendulum',0)]:
        for family in ('primary','round0'):
            p=OUT/'smoke'/task/arm_name(family,seed,BASE[task] if family=='primary' else 0)/'completed.json'
            assert read(p)['passed']
            for name,h in read(p)['hashes'].items():assert digest(Path(name))==h
    lock=(OUT/'suite.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(active=True,pid=os.getpid(),completed=[],started=time.time());write(OUT/'status.json',status)
    def launch(j):
        task,seed,family=j;log=OUT/('%s_s%d_%s_%d.log'%(task,seed,family,time.time_ns()))
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--mode','job','--task',task,'--seed',str(seed),'--family',family],stdout=stream,stderr=subprocess.STDOUT,timeout=14400,cwd=str(ROOT))
        return dict(job=j,exit_code=result.returncode,log=str(log))
    jobs=[(task,seed,family) for task in TASKS for seed in range(3) for family in ('primary','round0','round1')]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(launch,j) for j in jobs]):
                status['completed'].append(future.result());write(OUT/'status.json',status)
        status.update(active=False,complete=all(j['exit_code']==0 for j in status['completed']),ended=time.time());write(OUT/'status.json',status)
        assert status['complete'],'Keep failed diagnostics; inspect logs'
    except BaseException as exc:
        status.update(active=False,complete=False,exception=repr(exc));write(OUT/'status.json',status);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','job','suite'],required=True)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--family',choices=['primary','round0','round1']);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.mode=='register':register()
    elif a.mode=='suite':suite()
    else:job(a.task,a.seed,a.family,a.smoke)
