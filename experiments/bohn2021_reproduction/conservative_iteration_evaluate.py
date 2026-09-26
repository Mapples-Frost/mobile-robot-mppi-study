"""New-bank evaluation for frozen conservative policies and strong fixed grids."""
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
from conservative_iteration import OUT,TASKS,HS,BASE,BLOCK,verify,protocol
from runtime import ROOT,ART,imports
from conservative_canonical_reset import make_env
import numpy as np
from run import write,snapshot,weights_hash
from paper_h_soft_probe import read,digest
from min_q_eval_suite import model_dir
from conservative_policy_model import choose
from relative_policy_features import context
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from fixed_policy_branches import metrics


def grid_model(task,h,seed):
    if h==BASE[task]:return model_dir(task,'fixed',seed)
    if seed:return OUT/'extra_fixed'/('%s_h%d_s%d'%(task,h,seed))
    name='%s_fixed_h%d'%(task,h)
    candidates=[ART/'results'/group/name for group in ('paper_defaults','paper_exact_grid_2026-09-23')]
    found=[p for p in candidates if (p/'completed.json').exists()]
    assert len(found)==1,(task,h,found)
    return found[0]


def arm_name(family,seed,h):return '%s_h%d_s%d'%(family,h,seed)


def jobs():
    result=[]
    for task in TASKS:
        for seed in range(3):
            result.append((task,'primary',seed,BASE[task]))
            result += [(task,'round%d'%r,seed,0) for r in range(2)]
            result += [(task,'matched',seed,h) for h in HS if h!=BASE[task]]
        result += [(task,'grid',0,h) for h in range(5,51,5) if h!=BASE[task]]
    return result


def register():
    verify()
    path=OUT/'evaluation_registration.json'
    files=[Path(__file__),Path(__file__).with_name('conservative_iteration_audit.py'),
           Path(__file__).with_name('conservative_policy_model.py'),OUT/'protocol.json',OUT/'learner_registration.json']
    for task in TASKS:
        for h in range(5,51,5):
            source=grid_model(task,h,0)
            files += [source/n for n in ('model.zip','manifest.json','completed.json')]
    value=dict(hashes={str(p):digest(p) for p in files},validation_jobs=jobs(),
        fixed_selection='Among seed0 independently trained H5..50 with success>=primary, constraints<=primary and solver-failure rate<=primary, minimize mean total cost; ties smaller H. Primary always eligible. Additional seeds1/2 get15k steps if selected H differs.',
        matched_selection='One common H among all seven, require no safety degradation per seed relative to primary, choose minimum pooled total cost; ties shorter H.',
        adaptive_selection='Select common round by pooled total cost only after all comparator validation exists and constraints in protocol hold. Compare to both selected independent and selected matched-terminal fixed policies.',
        tie_tolerance=0,scope='Validation selection only; independent test requires separate sealed confirmation registration')
    value=json.loads(json.dumps(value))
    if path.exists():
        registered=read(path)
        canonical=lambda mapping:{str(Path(p).resolve()):h for p,h in mapping.items()}
        registered['hashes']=canonical(registered['hashes'])
        value['hashes']=canonical(value['hashes'])
        assert registered==value
    else:write(path,value)


def banks():
    register()
    dest=OUT/'banks';dest.mkdir(exist_ok=True)
    for task in TASKS:
        pending=[s for s in ('validation','test') if not (dest/(task+'_'+s+'.json')).exists()]
        if not pending:continue
        env=make_env(task,0,aligned=True,scaled_obs=True);meter(env,dest)
        for split in pending:
            params=protocol()['splits'];scene_seed=params[split+'_seed_base']+TASKS.index(task)*100
            env.seed(scene_seed);np.random.seed(scene_seed);cases=[]
            for _ in range(params[split+'_cases']):env.reset();cases.append(snapshot(env))
            write(dest/(task+'_'+split+'.json'),dict(seed=scene_seed,split=split,cases=cases,
                use='Sealed outcome evaluation; no policy decisions collected' if split=='test' else 'Model/baseline selection'))
    path=dest/'hashes.json';data={str(p):digest(p) for p in dest.glob('*_*.json') if p.name.startswith(('vehicle_','pendulum_'))}
    if path.exists():assert read(path)==data
    else:write(path,data)


def evaluate(task,family,seed,h,split):
    register()
    if split=='test':
        confirm=read(OUT/'confirmation_registration.json')
        assert confirm['validation_gate_passed']
        assert [task,family,seed,h] in confirm['jobs']
        for p,value in confirm['hashes'].items():assert digest(Path(p))==value
    bank=OUT/'banks'/(task+'_'+split+'.json')
    assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
    dest=OUT/'evaluations'/split/task/arm_name(family,seed,h)
    dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'completed.json').exists():
        for p,value in read(dest/'completed.json')['hashes'].items():assert digest(Path(p))==value
        return
    policy=None
    source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
    policy_path=None
    if family.startswith('round'):
        folder=OUT/('%s_s%d_r%d'%(task,seed,int(family[-1])))
        assert read(folder/'collection_audit.json')['passed']
        fit=read(folder/'fit_completed.json');policy_path=folder/'policy.json'
        assert fit['policy_hash']==digest(policy_path)
        policy=read(policy_path)
    manifest=read(source/'manifest.json');done=read(source/'completed.json')
    assert done['status']=='complete' and done['steps']==15000 and manifest['seed']==seed
    env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,dest)
    _,SAC,_=imports();model=SAC.load(str(source/'model.zip'))
    assert weights_hash(model)==done['final_hash']
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
    rows=[];hashes={str(bank):digest(bank),str(source/'model.zip'):digest(source/'model.zip')}
    if policy_path:hashes[str(policy_path)]=digest(policy_path)
    for cid,case in enumerate(read(bank)['cases']):
        path=dest/('trace_%02d.json'%cid)
        if not path.exists():
            env.reset(**copy.deepcopy(case));trace=[]
            for t in range(env.max_steps):
                ctx=context(env,task)
                if t%BLOCK==0:selected=choose(policy,ctx['features']) if policy else h
                _,terminated,row=observed_step(env,task,selected,case,t)
                row['policy_context']=ctx;trace.append(row)
                if terminated:break
            assert terminated
            audit_trace(task,case,trace);audit_context(task,case,trace,0);write(path,trace)
        trace=read(path);audit_trace(task,case,trace);audit_context(task,case,trace,0)
        for t,row in enumerate(trace):
            if t%BLOCK==0:selected=choose(policy,row['policy_context']['features']) if policy else h
            assert row['horizon']==selected
        m=metrics(task,trace);m.update(case=cid,physical_constraint_cost=m['performance_cost']+m['constraint_cost'],
            unique_horizons=len(set(r['horizon'] for r in trace)),horizon_switches=sum(trace[k]['horizon']!=trace[k-1]['horizon'] for k in range(1,len(trace))))
        rows.append(m);hashes[str(path)]=digest(path)
        print(json.dumps(dict(task=task,arm=arm_name(family,seed,h),case=cid,cost=m['total_cost'])),flush=True)
    attempts=[read(p) for p in dest.glob('attempt_*.json')]
    summary=dict(task=task,family=family,seed=seed,h=h,split=split,episodes=rows,
        explicit_step_calls=sum(a['step_calls'] for a in attempts),explicit_reset_calls=sum(a['reset_calls'] for a in attempts))
    write(dest/'summary.json',summary);hashes[str(dest/'summary.json')]=digest(dest/'summary.json')
    write(dest/'completed.json',dict(passed=True,hashes=hashes,model_hash=done['final_hash'],
        evaluation_registration_hash=digest(OUT/'evaluation_registration.json')))


def suite(split):
    register()
    selected=jobs() if split=='validation' else read(OUT/'confirmation_registration.json')['jobs']
    def launch(j):
        task,family,seed,h=j
        log=OUT/('eval_%s_%s_%s_%d.log'%(split,task,arm_name(family,seed,h),time.time_ns()))
        cmd=[sys.executable,'-u',__file__,'--mode','job','--split',split,'--task',task,
             '--family',family,'--seed',str(seed),'--h',str(h)]
        with log.open('w') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
        return dict(job=j,exit_code=r.returncode,log=str(log))
    status=dict(active=True,pid=os.getpid(),finished=[],split=split);path=OUT/(split+'_status.json');write(path,status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(launch,j) for j in selected]):
            status['finished'].append(f.result());write(path,status)
    status.update(active=False,complete=all(j['exit_code']==0 for j in status['finished']));write(path,status)
    assert status['complete'],'Inspect evaluation errors before continuing'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','banks','job','suite'],required=True)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--family',choices=['primary','grid','matched','round0','round1'])
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--h',type=int,default=0);ap.add_argument('--split',choices=['validation','test'],default='validation')
    a=ap.parse_args()
    if a.mode=='register':register()
    elif a.mode=='banks':banks()
    elif a.mode=='job':evaluate(a.task,a.family,a.seed,a.h,a.split)
    else:suite(a.split)
