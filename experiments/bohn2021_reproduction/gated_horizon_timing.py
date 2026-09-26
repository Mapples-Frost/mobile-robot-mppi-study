"""Serial end-to-end decision timing with measured logging subtraction and exact replay."""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import os
import platform
import sys
import time
from pathlib import Path
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE,decide,candidates
from gated_horizon_evaluate import grid_model,arm_name,freeze_policies
from conservative_canonical_reset import make_env
import conservative_solver_recovery as recovery_module
from conservative_iteration_timing import idle
from relative_policy_features import context
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from min_q_eval_suite import model_dir
from runtime import imports
from run import write,weights_hash
from paper_h_soft_probe import read,digest
from gated_horizon_shared_reset import install as install_shared
from gated_horizon_amendment import registration as amended_registration,verify as amendment
import numpy as np


class LoggingTimer:
    """Measure only two recovery logging boundaries, leaving solver code unchanged."""
    def __init__(self,folder):self.folder=folder;self.seconds=0.;self.operations=0
    def __enter__(self):
        self.original=recovery_module.write
        def measured(*args,**kwargs):
            start=time.perf_counter()
            try:return self.original(*args,**kwargs)
            finally:self.seconds+=time.perf_counter()-start;self.operations+=1
        recovery_module.write=measured
        return self
    def __exit__(self,*args):recovery_module.write=self.original
    def __truediv__(self,name):
        if name!='solver_calls.jsonl':return self.folder/name
        owner=self
        class Append:
            @contextmanager
            def open(self,mode):
                assert mode=='a';start=time.perf_counter()
                try:
                    with (owner.folder/name).open(mode) as stream:yield stream
                finally:owner.seconds+=time.perf_counter()-start;owner.operations+=1
        return Append()


def strip_timing(trace):
    result=copy.deepcopy(trace)
    for row in result:
        row.pop('timing',None)
        for attempt in row['recovery']['attempts']:attempt.pop('solver_s')
    return result


def register():
    freeze();p=OUT/'timing_registration.json'
    paths=[Path(__file__).resolve(),OUT/'protocol.json',OUT/'evaluation_registration.json']
    value=dict(hashes={str(p):digest(p) for p in paths},repeats=2,order_seed=2609257000,
        measurement='Every step measures policy context+gate for adaptive arms and controller.get_action including all attempts, feasibility checks, retry initialization and restoration. Subtract measured recovery JSON counter writes and JSONL append serialization/open/write/close intervals. Retain gross and logging times separately. Simulator, audit context, trace writes and environment call meter outside controller are excluded. Reset measured separately.',
        comparison='Same timing implementation for all arms. Two randomized serial repeats; all6 trained policies and both nominated comparator families, even when method fails efficacy. Duplicated identical fixed arms measured once.',
        exactness='Strip top-level timing and solver_s fields only; remaining trace must equal saved validation/test trace exactly. No warmed-up optimized policy implementation substituted.',
        smoke='Same2 smoke scenes per task, fixed and widest H5 gate,2 repeats; may coexist with training, used only to check instrumentation and replay, never efficacy.',
        environment='Single-thread numerical libraries, no concurrent experiment Python jobs during formal timing. WSL host scheduling and clock instrumentation remain limitations.',test_access=False)
    if p.exists():amended_registration(p,value)
    else:
        assert not (OUT/'evaluations').exists(),'Freeze timing implementation before validation outcomes'
        write(p,value)


def jobs(split):
    if split=='test':
        c=read(OUT/'confirmation_registration.json');assert c['validation_gate_passed']
        for p,h in c['hashes'].items():assert digest(Path(p))==h
        return [tuple(j) for j in c['jobs']]
    selected=read(OUT/'baseline_selection.json');assert not selected['pending_independent_baselines']
    for p,h in selected['hashes'].items():assert digest(Path(p))==h
    result=[]
    for task,n in selected['nominations'].items():
        for seed in range(3):
            result.append((task,'adaptive',seed,0))
            for label,family in [('independent','grid'),('matched','matched')]:
                h=n[label+'_h'];result.append((task,'primary' if h==BASE[task] else family,seed,h))
    return sorted(set(result))


def run(split,smoke=False):
    register()
    if not smoke:freeze_policies();idle()
    dest=OUT/('timing_smoke' if smoke else 'timing_'+split);dest.mkdir(exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    conditions=[(t,f,0,BASE[t] if f=='primary' else 0) for t in TASKS for f in ('primary','smoke_candidate')] if smoke else jobs(split)
    if not (dest/'environment.json').exists():
        write(dest/'environment.json',dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
            thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
            cpuinfo=Path('/proc/cpuinfo').read_text(),start=time.time(),smoke_not_efficacy=smoke))
    _,SAC,_=imports()
    for repeat in range(2):
        for index in np.random.RandomState(2609257000+repeat).permutation(len(conditions)):
            if not smoke:idle()
            task,family,seed,h=conditions[index]
            folder=dest/('r%d_%s_%s'%(repeat,task,arm_name(family,seed,h)));folder.mkdir(exist_ok=True)
            if (folder/'completed.json').exists():
                for p,value in read(folder/'completed.json')['hashes'].items():assert digest(Path(p))==value
                continue
            assert not (folder/'solver_attempts.json').exists(),'Inspect interrupted timing condition before resuming'
            source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
            if smoke:
                policy=next(c for c in candidates(task) if c['id']=='h5_p2_g5') if family=='smoke_candidate' else None
                reference=OUT/'smoke'/('%s_s0'%task)/(policy['id'] if policy else 'fixed');bank=OUT/'banks'/(task+'_smoke_bank.json')
            else:
                policy=read(OUT/'train'/('%s_s%d'%(task,seed))/'policy.json') if family=='adaptive' else None
                reference=OUT/'evaluations'/split/task/arm_name(family,seed,h);bank=OUT/'banks'/('%s_%s_bank.json'%(task,split))
            assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
            for p,value in read(reference/'completed.json')['hashes'].items():assert digest(Path(p))==value
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,folder)
            model=SAC.load(str(source/'model.zip'));assert weights_hash(model)==read(source/'completed.json')['final_hash']
            own_terminal=model.policy_tf.get_mpc_vfn_weights_and_biases();env.set_value_function_weights_and_biases(*own_terminal);model.sess.close()
            install_shared(env,task,seed,own_terminal,independent_terminal=family=='grid' and h!=BASE[task])
            controller=env.control_system.controller;original=controller.get_action;measured=[]
            with LoggingTimer(folder) as logging:
                recovery=recovery_module.install(controller.mpc,logging)
                def timed(*args,**kwargs):
                    before=logging.seconds;start=time.perf_counter();value=original(*args,**kwargs);gross=time.perf_counter()-start
                    logged=logging.seconds-before;assert 0<=logged<gross
                    measured.append(dict(controller_gross_s=gross,logging_s=logged,controller_s=gross-logged));return value
                controller.get_action=timed;episodes=[]
                hashes={str(bank):digest(bank),str(reference/'completed.json'):digest(reference/'completed.json'),str(source/'model.zip'):digest(source/'model.zip'),str(OUT/'timing_registration.json'):digest(OUT/'timing_registration.json')}
                receipt=amendment();hashes[str(receipt)]=digest(receipt)
                for cid,case in enumerate(read(bank)['cases']):
                    recovery.update(enabled=False,events=[],case=cid,step=-1)
                    start=time.perf_counter();env.reset(**copy.deepcopy(case));elapsed=time.perf_counter()-start
                    assert len(measured)==1;reset=dict(reset_gross_s=elapsed,**measured.pop());recovery['enabled']=True;trace=[]
                    if cid==0 and policy:decide(policy,context(env,task))
                    for t in range(env.max_steps):
                        recovery['step']=t;ctx=context(env,task)
                        start=time.perf_counter()
                        if policy:selected,gate=decide(policy,context(env,task))
                        else:selected,gate=h,dict(use_short=False,reason='fixed') if smoke else dict(use_short=False,reason='fixed',fixed_h=h)
                        selection_s=time.perf_counter()-start
                        _,terminated,row=observed_step(env,task,selected,case,t);assert len(measured)==1
                        timing=measured.pop();timing.update(selection_s=selection_s,decision_s=selection_s+timing['controller_s'],decision_gross_s=selection_s+timing['controller_gross_s'])
                        row.update(policy_context=ctx,gate=gate,recovery=recovery['events'][-1],timing=timing);trace.append(row)
                        if terminated:break
                    assert terminated;audit_trace(task,case,trace)
                    assert strip_timing(trace)==strip_timing(read(reference/('r0_trace_%02d.json'%cid)))
                    p=folder/('r0_trace_%02d.json'%cid);write(p,trace);hashes[str(p)]=digest(p)
                    p=folder/('reset_%02d.json'%cid);write(p,reset);hashes[str(p)]=digest(p)
                    times=np.asarray([r['timing']['decision_s'] for r in trace])
                    episodes.append(dict(case=cid,steps=len(trace),cost=sum(-r['reward'] for r in trace),reset=reset,
                        decision_total_s=float(times.sum()),decision_mean_s=float(times.mean()),decision_median_s=float(np.median(times)),decision_p95_s=float(np.percentile(times,95)),
                        deadline_exceed_steps=int(np.sum(times>(.1 if task=='vehicle' else .04))),
                        decision_gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),logging_total_s=sum(r['timing']['logging_s'] for r in trace)))
                assert logging.operations==1+3*recovery['counts']['solve_completed']
                write(folder/'summary.json',dict(task=task,family=family,seed=seed,h=h,split=split,repeat=repeat,smoke=smoke,episodes=episodes,
                    solver_counts=recovery['counts'],logging_operations=logging.operations,logging_total_s=logging.seconds))
            for p in (folder/'summary.json',folder/'solver_attempts.json',folder/'solver_calls.jsonl'):hashes[str(p)]=digest(p)
            write(folder/'completed.json',dict(passed=True,exact_replay=True,hashes=hashes));print(folder.name+' replay and timing accounting passed',flush=True)
    paths=list(dest.glob('r*/completed.json'));assert len(paths)==2*len(conditions)
    write(dest/'completed.json',dict(passed=True,conditions=len(paths),smoke=smoke,hashes={str(p):digest(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--split',choices=['validation','test'],default='validation');a=ap.parse_args()
    if a.register:register()
    else:run(a.split,a.smoke)
