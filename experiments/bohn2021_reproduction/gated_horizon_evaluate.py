"""Frozen validation and sealed confirmation, with full ten-H fixed grids."""
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
from gated_horizon_search import OUT,TASKS,freeze,case_metrics
from gated_horizon_policy import BASE,decide
from conservative_canonical_reset import make_env
from conservative_solver_recovery import install
from conservative_iteration_evaluate import grid_model as inherited_grid,arm_name
from relative_policy_features import context
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from min_q_eval_suite import model_dir
from runtime import ROOT,imports
from run import write,weights_hash
from paper_h_soft_probe import read,digest
from gated_horizon_shared_reset import install as install_shared
from gated_horizon_amendment import registration as amended_registration,verify as amendment

HS=list(range(5,51,5))


def grid_model(task,h,seed):
    if h==BASE[task]:return model_dir(task,'fixed',seed)
    if seed==0:return inherited_grid(task,h,0)
    return OUT/'extra_fixed'/('%s_h%d_s%d'%(task,h,seed))


def jobs():
    return [(t,f,s,h) for t in TASKS for s in range(3) for f,h in
            [('adaptive',0),('primary',BASE[t])]+[('matched',h) for h in HS if h!=BASE[t]]]+[
            (t,'grid',0,h) for t in TASKS for h in HS if h!=BASE[t]]


def register():
    freeze()
    assert not (OUT/'shared_initialization_amendment/pending.json').exists(),'Shared initialization amendment must be completed and audited before evaluation'
    paths=[Path(__file__).resolve(),Path(__file__).with_name('gated_horizon_audit.py').resolve(),Path(__file__).with_name('gated_horizon_select.py').resolve(),OUT/'protocol.json',OUT/'inputs_sha256.json']
    for task in TASKS:
        for h in HS:
            source=grid_model(task,h,0);m=read(source/'manifest.json');d=read(source/'completed.json')
            assert m['adaptations']==dict(aligned=True,scaled_obs=True,ent_coef='1.0',no_online_value=False,batch_size=256,buffer_size=1000000)
            assert m['steps']==d['steps']==15000 and m['fixed_horizon']==h and m['seed']==0 and d['status']=='complete'
            paths += [source/n for n in ('model.zip','manifest.json','completed.json')]
    value=dict(hashes={str(p):digest(p) for p in paths},jobs=jobs(),
        rule='No adaptive validation tuning. Full10 H independent seed0 and full10 H matched terminal at all3 seeds. Safety includes both initial and final failed-step rates. Minimum raw cost, ties smaller H.',
        test='Requires passing validation effect gate, independently audited traces, frozen comparator/model hashes and explicit confirmation registration.',
        recovery='Identical bounded original-NLP retry policy for all arms, off during author H50 reset warmup.',workers=2)
    value=json.loads(json.dumps(value));path=OUT/'evaluation_registration.json'
    if path.exists():amended_registration(path,value)
    else:
        assert not (OUT/'evaluations').exists(),'Freeze evaluator before any validation outcomes'
        write(path,value)


def freeze_policies():
    register();audit=read(OUT/'audit_train.json');assert audit['passed']
    hashes={str(OUT/'audit_train.json'):digest(OUT/'audit_train.json')}
    for task in TASKS:
        for seed in range(3):
            folder=OUT/'train'/('%s_s%d'%(task,seed));fit=read(folder/'fit_completed.json')
            assert fit['policy_hash']==digest(folder/'policy.json')
            for p in (folder/'policy.json',folder/'fit_completed.json'):hashes[str(p)]=digest(p)
    path=OUT/'fitted_policy_registration.json';value=dict(hashes=hashes,validation_access=False,test_access=False)
    if path.exists():assert read(path)==value
    else:
        assert not (OUT/'evaluations').exists();write(path,value)


def evaluate(task,family,seed,h,split):
    freeze_policies()
    if split=='test':
        confirm=read(OUT/'confirmation_registration.json');assert confirm['validation_gate_passed']
        assert [task,family,seed,h] in confirm['jobs']
        for p,value in confirm['hashes'].items():assert digest(Path(p))==value
    bank=OUT/'banks'/('%s_%s_bank.json'%(task,split));assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
    dest=OUT/'evaluations'/split/task/arm_name(family,seed,h);dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'completed.json').exists():
        for p,v in read(dest/'completed.json')['hashes'].items():assert digest(Path(p))==v
        return
    assert not (dest/'solver_attempts.json').exists(),'Audit interrupted evaluation before resuming'
    source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
    manifest=read(source/'manifest.json');done=read(source/'completed.json')
    assert done['status']=='complete' and done['steps']==15000 and manifest['seed']==seed
    policy_path=OUT/'train'/('%s_s%d'%(task,seed))/'policy.json' if family=='adaptive' else None
    policy=read(policy_path) if policy_path else dict(id='fixed',task=task)
    env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,dest)
    _,SAC,_=imports();model=SAC.load(str(source/'model.zip'));assert weights_hash(model)==done['final_hash']
    own_terminal=model.policy_tf.get_mpc_vfn_weights_and_biases();env.set_value_function_weights_and_biases(*own_terminal);model.sess.close()
    install_shared(env,task,seed,own_terminal,independent_terminal=family=='grid' and h!=BASE[task])
    recovery=install(env.control_system.controller.mpc,dest);rows=[]
    hashes={str(bank):digest(bank),str(source/'model.zip'):digest(source/'model.zip'),str(OUT/'evaluation_registration.json'):digest(OUT/'evaluation_registration.json')}
    receipt=amendment();hashes[str(receipt)]=digest(receipt);initial_states=[]
    if policy_path:hashes[str(policy_path)]=digest(policy_path)
    for cid,case in enumerate(read(bank)['cases']):
        recovery.update(enabled=False,events=[],case=cid,step=-1);env.reset(**copy.deepcopy(case));recovery['enabled']=True;trace=[]
        initial_states.append(copy.deepcopy(dict(state=env.control_system.current_state,input=env.control_system.controller.current_input)))
        for t in range(env.max_steps):
            recovery['step']=t;ctx=context(env,task)
            selected,gate=decide(policy,ctx) if policy_path else (h,dict(use_short=False,reason='fixed',fixed_h=h))
            _,terminated,row=observed_step(env,task,selected,case,t)
            row.update(policy_context=ctx,gate=gate,recovery=recovery['events'][-1]);trace.append(row)
            if terminated:break
        assert terminated;audit_trace(task,case,trace);audit_context(task,case,trace,0)
        p=dest/('r0_trace_%02d.json'%cid);write(p,trace);hashes[str(p)]=digest(p)
        m=case_metrics(task,trace);m.update(case=cid,repeat=0);rows.append(m)
        print(json.dumps(dict(task=task,arm=arm_name(family,seed,h),case=cid,cost=m['total_cost'])),flush=True)
    import numpy as np
    result=dict(policy=policy,episodes=rows,rejected=[],unrun_cases=[],fully_evaluated=True,
        mean_raw_cost=float(np.mean([e['total_cost'] for e in rows])),mean_physical_cost=float(np.mean([e['physical_constraint_cost'] for e in rows])))
    summary=dict(task=task,family=family,seed=seed,h=h,split=split,episodes=rows,solver_counts=recovery['counts'])
    write(dest/'summary.json',summary)
    p=dest/'initial_states.json';write(p,initial_states);hashes[str(p)]=digest(p)
    for p in (dest/'summary.json',dest/'solver_attempts.json',dest/'solver_calls.jsonl'):hashes[str(p)]=digest(p)
    write(dest/'completed.json',dict(passed=True,result=result,hashes=hashes,model_hash=done['final_hash']))


def suite(split):
    freeze_policies();lock=(OUT/(split+'_suite.lock')).open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=jobs() if split=='validation' else read(OUT/'confirmation_registration.json')['jobs']
    path=OUT/(split+'_status.json');state=dict(pid=os.getpid(),active=True,completed=[],split=split,started=time.time());write(path,state)
    def launch(j):
        task,family,seed,h=j;log=OUT/('eval_%s_%s_%s_%d.log'%(split,task,arm_name(family,seed,h),time.time_ns()))
        with log.open('w') as stream:
            r=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--mode','job','--split',split,'--task',task,'--family',family,'--seed',str(seed),'--h',str(h)],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
        return dict(job=j,exit_code=r.returncode,log=str(log))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(launch,j) for j in selected]):state['completed'].append(f.result());write(path,state)
        state.update(active=False,complete=all(j['exit_code']==0 for j in state['completed']),ended=time.time());write(path,state);assert state['complete']
    except BaseException as exc:
        state.update(active=False,complete=False,exception=repr(exc));write(path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','freeze-policies','job','suite'],required=True)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--family',choices=['adaptive','primary','matched','grid']);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--h',type=int,default=0)
    ap.add_argument('--split',choices=['validation','test'],default='validation');a=ap.parse_args()
    if a.mode=='register':register()
    elif a.mode=='freeze-policies':freeze_policies()
    elif a.mode=='job':evaluate(a.task,a.family,a.seed,a.h,a.split)
    else:suite(a.split)
