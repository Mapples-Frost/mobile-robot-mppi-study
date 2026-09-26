"""Serial measured decision latency for frozen selected policies, exact replay."""
import argparse
import copy
import fcntl
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS,BASE,BLOCK,verify
from conservative_iteration_evaluate import grid_model,arm_name
from conservative_policy_model import choose
from relative_policy_features import context
from runtime import imports
from conservative_canonical_reset import make_env
from run import write,weights_hash
from paper_h_soft_probe import read,digest
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from min_q_eval_suite import model_dir


def register():
    files=[Path(__file__),Path(__file__).with_name('conservative_iteration_select.py'),OUT/'protocol.json']
    data=dict(hashes={str(p):digest(p) for p in files},repeats=2,order_seed=2609247000,
        timing='At every control step: feature extraction and five-step policy choice when due, plus controller.get_action. Excludes simulator, audit, durable writes and reset. Untimed audit context on held-action steps is excluded because not needed for deployment.',
        warmup='One untimed full feature/choice per model before first measured episode; every reset has author H50 warmup, measured separately.',
        exactness='Strip timing only and compare every remaining raw trace field exactly with efficacy evaluation.',
        environment='Single-thread libraries, one timing process, no concurrent experiment Python processes except waiting monitor. WSL/host scheduling remains a limitation.',
        inference='Portable float32 NumPy, includes conversion of saved lists into arrays; same implementation used during validation. No unmeasured optimized inference substituted.',
        scope='Validation timing selects no policy. Confirmation timing uses separately frozen test job list.',
        diagnostic_validation='Optional separate all-candidate output: both trained rounds and both nominated fixed comparators, all seeds, including ineligible rounds. Descriptive only; never changes policy selection or test gate.')
    p=OUT/'timing_registration.json'
    if p.exists():
        registered=read(p)
        canonical=lambda mapping:{str(Path(path).resolve()):h for path,h in mapping.items()}
        registered['hashes']=canonical(registered['hashes'])
        data['hashes']=canonical(data['hashes'])
        assert registered==data
    else:write(p,data)


def idle():
    conflicts=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name)==os.getpid():continue
        try:args=path.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if args and 'python' in Path(args[0]).name and any('experiments/' in a and a.endswith('.py') for a in args[1:]):
            conflicts.append(dict(pid=int(path.parent.name),args=args))
    assert not conflicts,conflicts


def jobs(split):
    if split=='test':return read(OUT/'confirmation_registration.json')['jobs']
    selection=read(OUT/'validation_selection.json')
    assert selection['selection_eligible']
    result=[]
    for task in TASKS:
        s=selection['selections'][task]
        for seed in range(3):
            result.append((task,'round%d'%s['round'],seed,0))
            result.append((task,'primary' if s['independent_h']==BASE[task] else 'grid',seed,s['independent_h']))
            result.append((task,'primary' if s['matched_h']==BASE[task] else 'matched',seed,s['matched_h']))
    return sorted(set(tuple(j) for j in result))


def all_candidate_jobs():
    selection=read(OUT/'validation_selection.json')
    result=[]
    for task in TASKS:
        s=selection['selections'][task]
        for seed in range(3):
            result += [(task,'round%d'%r,seed,0) for r in range(2)]
            result.append((task,'primary' if s['independent_h']==BASE[task] else 'grid',seed,s['independent_h']))
            result.append((task,'primary' if s['matched_h']==BASE[task] else 'matched',seed,s['matched_h']))
    return sorted(set(result))


def run(split,all_candidates=False):
    verify();register();idle()
    assert not all_candidates or split=='validation','All-candidate diagnostics never open test'
    dest=OUT/('timing_validation_all_candidates' if all_candidates else 'timing_'+split);dest.mkdir(exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    conditions=all_candidate_jobs() if all_candidates else jobs(split)
    if not (dest/'environment.json').exists():
        write(dest/'environment.json',dict(platform=platform.platform(),python=sys.version,
            executable=sys.executable,thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
            cpuinfo=Path('/proc/cpuinfo').read_text(),start=time.time()))
    _,SAC,_=imports()
    for repeat in range(2):
        for index in np.random.RandomState(2609247000+repeat).permutation(len(conditions)):
            idle();task,family,seed,h=conditions[index]
            folder=dest/('r%d_%s_%s'%(repeat,task,arm_name(family,seed,h)));folder.mkdir(exist_ok=True)
            if (folder/'completed.json').exists():
                for p,value in read(folder/'completed.json')['hashes'].items():assert digest(Path(p))==value
                continue
            source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
            policy_path=OUT/('%s_s%d_r%d'%(task,seed,int(family[-1])))/'policy.json' if family.startswith('round') else None
            policy=read(policy_path) if policy_path else None
            bank=OUT/'banks'/(task+'_'+split+'.json')
            reference=OUT/'evaluations'/split/task/arm_name(family,seed,h)
            done=read(reference/'completed.json');assert done['passed']
            for p,value in done['hashes'].items():assert digest(Path(p))==value
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,folder)
            model=SAC.load(str(source/'model.zip'));assert weights_hash(model)==read(source/'completed.json')['final_hash']
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
            controller=env.control_system.controller;original=controller.get_action;measured=[]
            def timed(*args,**kw):
                start=time.perf_counter();value=original(*args,**kw);measured.append(time.perf_counter()-start);return value
            controller.get_action=timed
            episodes=[];hashes={str(bank):digest(bank),str(reference/'completed.json'):digest(reference/'completed.json'),str(source/'model.zip'):digest(source/'model.zip')}
            if policy_path:hashes[str(policy_path)]=digest(policy_path)
            for cid,case in enumerate(read(bank)['cases']):
                path=folder/('trace_%02d.json'%cid)
                expected=read(reference/('trace_%02d.json'%cid))
                if path.exists():
                    trace=read(path);reset_info=read(folder/('reset_%02d.json'%cid))
                else:
                    start=time.perf_counter();env.reset(**copy.deepcopy(case));reset_elapsed=time.perf_counter()-start
                    assert len(measured)==1
                    reset_info=dict(reset_s=reset_elapsed,controller_s=measured.pop())
                    if cid==0 and policy:choose(policy,context(env,task)['features'])
                    trace=[]
                    for t in range(env.max_steps):
                        # Audit-only context has no RNG side effects and is outside timing.
                        ctx=context(env,task)
                        start=time.perf_counter()
                        if t%BLOCK==0:selected=choose(policy,context(env,task)['features']) if policy else h
                        selection_s=time.perf_counter()-start
                        _,terminated,row=observed_step(env,task,selected,case,t)
                        assert len(measured)==1
                        controller_s=measured.pop();row['policy_context']=ctx
                        row['timing']=dict(selection_s=selection_s,controller_s=controller_s,decision_s=selection_s+controller_s)
                        trace.append(row)
                        if terminated:break
                    assert terminated
                    write(path,trace);write(folder/('reset_%02d.json'%cid),reset_info)
                assert [{k:v for k,v in r.items() if k!='timing'} for r in trace]==expected
                audit_trace(task,case,trace)
                times=np.array([r['timing']['decision_s'] for r in trace])
                episodes.append(dict(case=cid,steps=len(trace),termination=trace[-1]['termination'],
                    cost=sum(-r['reward'] for r in trace),decision_mean_s=float(times.mean()),
                    decision_total_s=float(times.sum()),decision_median_s=float(np.median(times)),
                    decision_p95_s=float(np.percentile(times,95)),
                    deadline_exceed_steps=int(np.sum(times>(.1 if task=='vehicle' else .04))),reset=reset_info))
                hashes[str(path)]=digest(path)
                hashes[str(folder/('reset_%02d.json'%cid))]=digest(folder/('reset_%02d.json'%cid))
            write(folder/'summary.json',dict(task=task,family=family,seed=seed,h=h,repeat=repeat,split=split,episodes=episodes))
            hashes[str(folder/'summary.json')]=digest(folder/'summary.json')
            write(folder/'completed.json',dict(passed=True,exact_replay=True,hashes=hashes))
            print(folder.name+' measured and replay audited',flush=True)
    paths=list(dest.glob('r*/completed.json'));assert len(paths)==2*len(conditions)
    write(dest/'completed.json',dict(passed=True,conditions=len(paths),hashes={str(p):digest(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],default='validation');ap.add_argument('--register',action='store_true');ap.add_argument('--all-candidates',action='store_true')
    a=ap.parse_args()
    if a.register:register()
    else:run(a.split,a.all_candidates)
