"""Raw closed-loop policy search; new banks, immutable rules, durable attempts."""
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
import numpy as np
from gated_horizon_policy import BASE,PROFILES,candidates,decide
from conservative_canonical_reset import make_env
from conservative_solver_recovery import install
from relative_policy_features import context
from conservative_iteration_audit import audit_context
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from fixed_policy_branches import metrics
from min_q_eval_suite import model_dir
from run import write,snapshot,weights_hash
from paper_h_soft_probe import read,digest

OUT=ART/'results/gated_horizon_search_2026-09-25'
TASKS=('vehicle','pendulum')


def spec():
    return dict(method='Finite direct closed-loop training search over gated horizon shortening; method extension, not original SAC.',
        rationale='Observed signed-log regression/generalization and repeated-deployment failures motivate direct raw episode objective and a smaller policy class.',
        tasks=TASKS,seeds=[0,1,2],profiles=PROFILES,candidates={t:candidates(t) for t in TASKS},
        state='Current plant state, previous control, and <=50-step controller forecast only. Fixed H in first5 scored steps; recompute gate each subsequent control step. No five-step hold.',
        terminal='Reuse independent15k-step H25/H30 terminal for each seed unchanged. Both adaptive and fixed arms use identical bounded same-NLP recovery; reset warmup recovery disabled.',
        train_cases=24,smoke_cases=2,validation_cases=32,test_cases=64,
        seeds_for_banks=dict(train_base=2609251000,smoke_base=2609250990,validation_base=2609252000,test_base=2609253000,task_offset=100),
        training='Each seed evaluates baseline fixedH on all24 new training cases. Candidate order predetermined by code, same cases. Evaluate36 candidates; after a complete episode reject immediately if any per-case necessary condition below fails. Preserve that failure; unrun cases explicitly recorded.',
        necessary_per_case='Success>=fixed; constraint<=fixed; initial solver-failed steps<=fixed; final solver-failed steps<=fixed; physical+constraint cost<=fixed+.05*abs(fixed)+1e-8.',
        train_choice='Among candidates completing all24 cases and mean physical+constraint cost<=fixed+.02*abs(fixed), choose minimum mean raw total cost, with baseline fixed included. Exact ties choose baseline then candidate id. No transformed cost, critic, entropy, policy continuation labels or validation checkpoint choice.',
        validation='Freeze all3 fitted policies per task before outcomes. Evaluate32 fresh common cases. Fully re-evaluate independent seed0 H5..50 by5; nominate minimum raw cost among those no worse in success/constraint/initial and final failure rates than primary. If H changes give seeds1/2 independent15k terminal learning with original settings. Also full matched-terminal H5..50 at all3 seeds, same safety criterion per seed and minimum pooled cost.',
        efficacy='Both tasks and both nominated fixed comparators: all3 seeds preserve success counts, constraint counts, initial and final failed-step rates, physical+constraint cost within2%; actual within-episode H change required. Gain either each seed >=3% total-cost reduction and paired95% CI upper delta<0, or each seed >=10% measured decision-latency reduction with95% upper ratio<1 and total-cost noninferiority2%. Both timing repetitions must favor adaptation for speed claims. Same criteria on independently sealed test.',
        inference='10000 paired scenario bootstrap samples, seeds retained together; per-seed results also shown. Intervals conditional on3 trained policies. No test access unless validation efficacy gate passes and models/comparators freeze.',
        timing='Two serial randomized-order repeats,policy+controller including retries; simulator/writes excluded,reset separate. Report median,p95,deadline misses,episode totals. H is proxy only.',
        smoke='Baseline plus h5_p2_g5 on2 separately generated scenes per task, each replayed twice. No selection or pruning in smoke.',
        budgets='All baseline search/terminal learning inherited budget plus36 candidate searches,pruned failures,full validation and timing separately disclosed. Every step/reset and solver attempt durably logged. Baseline term learning same opportunity; extra adaptive search is not equal-budget superiority.',
        limitations='Structured policy may fail to exploit useful long H; finite36 candidates are not general global policy search. Training feasibility and geometric gates give no formal safety guarantee. Original masked50 NLP retained.',
        workers=2,test_access=False)


def freeze():
    OUT.mkdir(exist_ok=True)
    value=json.loads(json.dumps(spec()))
    files=[Path(__file__).resolve(),Path(__file__).with_name('gated_horizon_policy.py').resolve()]
    files += [Path(__file__).with_name(n).resolve() for n in ('conservative_solver_recovery.py','conservative_canonical_reset.py','relative_policy_features.py','runtime.py')]
    for task in TASKS:
        files.append(ART/'configs'/(task+'.json'))
        for seed in range(3):files += [model_dir(task,'fixed',seed)/n for n in ('model.zip','manifest.json','completed.json')]
    hashes={str(p):digest(p) for p in files}
    for name,data in [('protocol.json',value),('inputs_sha256.json',hashes)]:
        p=OUT/name
        if p.exists():assert read(p)==data,(name,'Frozen inputs changed')
        else:write(p,data)


def banks():
    freeze();dest=OUT/'banks';dest.mkdir(exist_ok=True)
    for ti,task in enumerate(TASKS):
        desired=[('train_s%d'%seed,2609251000+ti*100+seed,24) for seed in range(3)]
        desired += [('smoke',2609250990+ti,2),('validation',2609252000+ti*100,32),('test',2609253000+ti*100,64)]
        if all((dest/('%s_%s_bank.json'%(task,name))).exists() for name,_,_ in desired):continue
        env=make_env(task,0,aligned=True,scaled_obs=True);meter(env,dest)
        for name,seed,n in desired:
            p=dest/('%s_%s_bank.json'%(task,name))
            if p.exists():continue
            env.seed(seed);np.random.seed(seed);cases=[]
            for _ in range(n):env.reset();cases.append(snapshot(env))
            write(p,dict(task=task,split=name,seed=seed,cases=cases,test_sealed=name=='test'))
    hashes={str(p):digest(p) for p in sorted(dest.glob('*_bank.json'))}
    if (dest/'hashes.json').exists():assert read(dest/'hashes.json')==hashes
    else:write(dest/'hashes.json',hashes)


def case_metrics(task,trace):
    m=metrics(task,trace)
    return dict(m,physical_constraint_cost=m['performance_cost']+m['constraint_cost'],
        initial_failed_steps=sum(not r['recovery']['attempts'][0]['success'] for r in trace),
        retries=sum(len(r['recovery']['attempts'])-1 for r in trace),
        recovered_steps=sum(r['recovery']['recovered'] for r in trace),
        switches=sum(a['horizon']!=b['horizon'] for a,b in zip(trace,trace[1:])))


def violation(a,b):
    reasons=[]
    if a['success']<b['success']:reasons.append('success')
    if a['constraint']>b['constraint']:reasons.append('constraint')
    if a['initial_failed_steps']>b['initial_failed_steps']:reasons.append('initial_solver_failures')
    if a['solver_failure_steps']>b['solver_failure_steps']:reasons.append('final_solver_failures')
    if a['physical_constraint_cost']>b['physical_constraint_cost']+.05*abs(b['physical_constraint_cost'])+1e-8:reasons.append('per_case_physical_cost')
    return reasons


def train(task,seed,smoke=False):
    freeze();assert read(OUT/'split_audit.json')['passed'],'New splits must be audited before training'
    root=OUT/('smoke' if smoke else 'train')/('%s_s%d'%(task,seed));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    bank=OUT/'banks'/('%s_%s_bank.json'%(task,'smoke' if smoke else 'train_s%d'%seed))
    assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
    cases=read(bank)['cases'];source=model_dir(task,'fixed',seed);_,SAC,_=imports()
    model=SAC.load(str(source/'model.zip'));assert weights_hash(model)==read(source/'completed.json')['final_hash']
    terminal=model.policy_tf.get_mpc_vfn_weights_and_biases();model.sess.close()
    fixed=dict(id='fixed',task=task)
    choices=[fixed]+([c for c in candidates(task) if c['id']=='h5_p2_g5'] if smoke else candidates(task))
    results=[];baseline=None
    for policy in choices:
        dest=root/policy['id'];dest.mkdir(exist_ok=True);completion=dest/'completed.json'
        if completion.exists():
            d=read(completion)
            for p,h in d['hashes'].items():assert digest(Path(p))==h
            result=d['result'];results.append(result)
            if policy['id']=='fixed':baseline=result['episodes']
            continue
        assert not (dest/'solver_attempts.json').exists(),'Inspect interrupted candidate before resuming'
        env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,dest)
        env.set_value_function_weights_and_biases(*terminal);state=install(env.control_system.controller.mpc,dest)
        rows=[];rejected=[];hashes={str(bank):digest(bank),str(OUT/'inputs_sha256.json'):digest(OUT/'inputs_sha256.json')}
        repeats=2 if smoke else 1
        for repeat in range(repeats):
            for cid,case in enumerate(cases):
                state.update(enabled=False,events=[],case=cid,step=-1);env.reset(**copy.deepcopy(case));state['enabled']=True;trace=[]
                for t in range(env.max_steps):
                    state['step']=t;ctx=context(env,task);h,gate=decide(policy,ctx)
                    _,done,row=observed_step(env,task,h,case,t);row.update(policy_context=ctx,gate=gate,recovery=state['events'][-1]);trace.append(row)
                    if done:break
                assert done;audit_trace(task,case,trace);audit_context(task,case,trace,0)
                path=dest/('r%d_trace_%02d.json'%(repeat,cid))
                if repeat:
                    def clean(data):
                        d=copy.deepcopy(data)
                        for r in d:
                            for a in r['recovery']['attempts']:a.pop('solver_s')
                        return d
                    assert clean(trace)==clean(read(dest/('r0_trace_%02d.json'%cid)))
                write(path,trace);hashes[str(path)]=digest(path)
                m=case_metrics(task,trace);m.update(case=cid,repeat=repeat);rows.append(m)
                if baseline is not None and not smoke:
                    reasons=violation(m,baseline[cid])
                    if reasons:rejected.append(dict(case=cid,reasons=reasons));break
            if rejected:break
        state['enabled']=False
        result=dict(policy=policy,episodes=rows,rejected=rejected,unrun_cases=list(range(len(rows),len(cases))) if not smoke else [],
            fully_evaluated=len(rows)==len(cases)*repeats,mean_raw_cost=float(np.mean([m['total_cost'] for m in rows])),
            mean_physical_cost=float(np.mean([m['physical_constraint_cost'] for m in rows])))
        for p in (dest/'solver_attempts.json',dest/'solver_calls.jsonl'):hashes[str(p)]=digest(p)
        write(completion,dict(result=result,hashes=hashes));results.append(result)
        if policy['id']=='fixed':baseline=rows
        print(json.dumps(dict(task=task,seed=seed,candidate=policy['id'],cases=len(rows),rejected=rejected,cost=result['mean_raw_cost'])),flush=True)
    if smoke:
        write(root/'smoke_completed.json',dict(passed=True,candidates=len(results),results=results));return
    base=results[0];eligible=[r for r in results if r['fully_evaluated'] and not r['rejected'] and r['mean_physical_cost']<=base['mean_physical_cost']+.02*abs(base['mean_physical_cost'])]
    chosen=min(eligible,key=lambda r:(r['mean_raw_cost'],r['policy']['id']!='fixed',r['policy']['id']))
    write(root/'policy.json',dict(chosen['policy'],training_seed=seed))
    write(root/'fit_completed.json',dict(task=task,seed=seed,selected=chosen['policy'],results=results,
        policy_hash=digest(root/'policy.json'),training_bank_hash=digest(bank),gradient_updates=0,
        selection='Full raw training episode mean among preregistered admissible policies only',validation_access=False,test_access=False))


def suite():
    freeze()
    for task in TASKS:assert read(OUT/'smoke'/('%s_s0'%task)/'smoke_completed.json')['passed']
    lock=(OUT/'suite.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),active=True,completed=[],started=time.time());write(OUT/'status.json',state)
    def launch(j):
        task,seed=j;log=OUT/('train_%s_s%d_%d.log'%(task,seed,time.time_ns()))
        with log.open('w') as f:
            r=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--mode','train','--task',task,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),timeout=86400)
        return dict(task=task,seed=seed,exit_code=r.returncode,log=str(log))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(launch,(t,s)) for t in TASKS for s in range(3)]):
                state['completed'].append(future.result());write(OUT/'status.json',state)
        state.update(active=False,complete=all(j['exit_code']==0 for j in state['completed']),ended=time.time());write(OUT/'status.json',state)
        assert state['complete'],'Inspect failed training jobs'
    except BaseException as exc:
        state.update(active=False,complete=False,exception=repr(exc));write(OUT/'status.json',state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['freeze','banks','train','suite'],required=True)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='banks':banks()
    elif a.mode=='suite':suite()
    else:train(a.task,a.seed,a.smoke)
