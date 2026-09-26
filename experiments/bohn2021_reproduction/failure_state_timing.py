"""Conditional serial timing of all pendulum diagnostic arms, exact replay.

Only runs after the full diagnostic safety/physical-cost screen. This measures
training-case diagnosis and does not open validation/test or prove reproduction.
"""
import argparse
import copy
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from failure_state_protocol import ROOT, SCRIPTS, OLD, OUT, REG, read, write, sha, verify, bank, source_model
from failure_state_policy import decide
from runtime import imports
from run import weights_hash
from conservative_canonical_reset import make_env
from branch_calibration_run import meter, observed_step
from relative_policy_features import context
from gated_horizon_timing import LoggingTimer
from conservative_iteration_timing import idle
import conservative_solver_recovery as recovery_module

TIMING_REG=OUT/'timing_registration.json'


def clean(trace):
    rows=copy.deepcopy(trace)
    for row in rows:
        row.pop('timing',None);row.pop('gate',None)
        for a in row['recovery']['attempts']:a.pop('solver_s',None)
    return rows


def register():
    spec=verify()
    files=[Path(__file__),SCRIPTS/'failure_state_timing_audit.py',REG,
           SCRIPTS/'gated_horizon_timing.py',SCRIPTS/'conservative_iteration_timing.py',
           SCRIPTS/'conservative_solver_recovery.py',OUT/'serialization_repair/amendment.json']
    value=dict(source_hashes={str(p.resolve()):sha(p) for p in files},jobs=spec['timing_jobs'],
               diagnostic_only=True,case_ids=list(range(24)),repeats=2,
               requirement='Full diagnosis and independent audit passed; at least one certificate policy meets frozen all-seed safety/physical screen. Measure all4 policies regardless of candidate ranking. No running experiment Python processes.',
               measurement='Current-state certificate or fixed choice plus full controller.get_action including retries and prediction bookkeeping; subtract measured recovery log I/O, retain gross and log times. Simulator, external trace/audit/context instrumentation excluded. Reset separate. Selected certificate requires only current theta/omega, no unused full preview read inside its measured selection.',
               frozen_order=True,reference='Exact raw state/action/cost/observation/context/recovery replay of complete training diagnosis, ignoring timing and gate metadata only.',
               summary='All conditions/repeats/cases and net/gross decision distributions, deadlines40ms, retry costs. No new CI or efficacy test. Timing cannot serve independent confirmation.',test_access=False)
    if TIMING_REG.exists():
        original=read(TIMING_REG)
        repair=OUT/'timing_controller_repair/amendment.json'
        if repair.exists():
            amend=read(repair);assert sha(TIMING_REG)==amend['original_timing_registration_sha256']
            for p,h in amend['archived_files'].items():assert sha(ROOT/p)==h
            expected=dict(original['source_hashes'])
            for p,record in amend['changed_sources'].items():
                assert expected[p]==record['old_sha256']
                expected[p]=record['new_sha256']
            assert value['source_hashes']==expected
            assert {k:v for k,v in value.items() if k!='source_hashes'}=={k:v for k,v in original.items() if k!='source_hashes'}
        else:assert original==value
    else:write(TIMING_REG,value)
    return value


def run():
    registered=register();idle()
    s=read(OUT/'status.json');assert s['complete'] and not s['active']
    assert not (Path('/proc')/str(s['pid'])/'cmdline').exists()
    d=read(OUT/'delivery/report.json');assert d['passed'] and d['timing_screen_candidates']
    assert read(OUT/'audit_full.json')['passed']
    for p,h in d['hashes'].items():assert sha(Path(p))==h
    for p,h in registered['source_hashes'].items():assert sha(Path(p))==h
    dest=OUT/'timing';dest.mkdir(exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=OUT/'timing_status.json'
    if statepath.exists():
        previous=read(statepath)
        if previous.get('complete'):
            for p,h in previous['output_hashes'].items():assert sha(Path(p))==h
            print('Existing timing verified');return
        assert not (Path('/proc')/str(previous['pid'])/'cmdline').exists()
        write(OUT/('previous_timing_status_%d.json'%time.time_ns()),previous)
    state=dict(pid=os.getpid(),active=True,complete=False,started=time.time(),completed=[],registration_hash=sha(TIMING_REG))
    write(statepath,state)
    _,SAC,_=imports()
    try:
        for j in registered['jobs']:
            idle();repeat=j['repeat'];task,seed,policy=j['job'];assert task=='pendulum'
            folder=dest/('r%d_%s_s%d_%s'%(repeat,task,seed,policy));folder.mkdir(exist_ok=True)
            completion=folder/'completed.json'
            if completion.exists():
                for p,h in read(completion)['hashes'].items():assert sha(Path(p))==h
                state['completed'].append(str(completion));write(statepath,state);continue
            assert not list(folder.glob('attempt_*.json')) and not (folder/'solver_attempts.json').exists(),'Inspect partial timing before resume'
            full=OUT/'full'/('%s_s%d_%s'%(task,seed,policy));summary=read(full/'summary.json');reference=Path(summary['trace_folder'])
            for p,h in read(full/'completed.json')['hashes'].items():assert sha(Path(p))==h
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,folder)
            model=SAC.load(str(source_model(task,seed)/'model.zip'))
            assert weights_hash(model)==read(source_model(task,seed)/'completed.json')['final_hash']
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
            controller=env.control_system.controller;original=controller.get_action;measured=[]
            hashes={str(TIMING_REG):sha(TIMING_REG),str(full/'completed.json'):sha(full/'completed.json'),
                    str(bank(task,seed)):sha(bank(task,seed))}
            if not (dest/'environment.json').exists():
                write(dest/'environment.json',dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
                      cpuinfo=Path('/proc/cpuinfo').read_text(),started=time.time(),
                      threads={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
                      limitation='WSL host activity is not fully controlled; two serial repeats, not independent scenes.'))
            with LoggingTimer(folder) as logging:
                recovery=recovery_module.install(controller.mpc,logging)
                def timed(*args,**kwargs):
                    before=logging.seconds;start=time.perf_counter();result=original(*args,**kwargs);gross=time.perf_counter()-start
                    logs=logging.seconds-before;assert 0<logs<gross
                    measured.append(dict(controller_gross_s=gross,logging_s=logs,controller_s=gross-logs));return result
                controller.get_action=timed;episodes=[]
                for cid,case in enumerate(read(bank(task,seed))['cases']):
                    recovery.update(enabled=False,events=[],case=cid,step=-1)
                    start=time.perf_counter();env.reset(**copy.deepcopy(case));reset_s=time.perf_counter()-start
                    assert len(measured)==1;reset=dict(reset_gross_s=reset_s,**measured.pop());recovery['enabled']=True
                    trace=[]
                    for t in range(env.max_steps):
                        recovery['step']=t;ctx=context(env,task)
                        start=time.perf_counter()
                        selected,gate=decide(task,policy,{'state':env.control_system.current_state})
                        selection_s=time.perf_counter()-start
                        _,done,row=observed_step(env,task,selected,case,t);assert len(measured)==1
                        timing=measured.pop();timing.update(selection_s=selection_s,decision_s=selection_s+timing['controller_s'],
                                                          decision_gross_s=selection_s+timing['controller_gross_s'])
                        row.update(policy_context=ctx,gate=gate,recovery=recovery['events'][-1],timing=timing);trace.append(row)
                        if done:break
                    assert done
                    path=folder/('r0_trace_%02d.json'%cid);write(path,trace)
                    assert clean(trace)==clean(read(reference/('r0_trace_%02d.json'%cid)))
                    hashes[str(path)]=sha(path);reset_path=folder/('reset_%02d.json'%cid);write(reset_path,reset);hashes[str(reset_path)]=sha(reset_path)
                    values=np.array([r['timing']['decision_s'] for r in trace])
                    episodes.append(dict(case=cid,steps=len(trace),cost=sum(-r['reward'] for r in trace),reset=reset,
                         decision_total_s=float(values.sum()),decision_mean_s=float(values.mean()),decision_median_s=float(np.median(values)),
                         decision_p95_s=float(np.percentile(values,95)),deadline_exceed_steps=int(sum(values>.04)),
                         gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),logging_total_s=sum(r['timing']['logging_s'] for r in trace)))
                    write(folder/'progress.json',dict(pid=os.getpid(),episodes=len(episodes),expected=24))
                assert logging.operations==1+3*recovery['counts']['solve_completed']
                write(folder/'summary.json',dict(task=task,seed=seed,policy=policy,repeat=repeat,episodes=episodes,
                      reference=str(reference),solver_counts=recovery['counts'],logging_operations=logging.operations,logging_total_s=logging.seconds))
            for p in list(folder.glob('attempt_*.json'))+[folder/n for n in ('summary.json','solver_attempts.json','solver_calls.jsonl')]:hashes[str(p)]=sha(p)
            write(completion,dict(passed=True,exact_replay=True,hashes=hashes));state['completed'].append(str(completion));write(statepath,state)
            print(folder.name+' complete',flush=True)
        assert len(state['completed'])==24
        write(dest/'completed.json',dict(passed=True,conditions=24,hashes={p:sha(Path(p)) for p in state['completed']}))
        state.update(active=False,complete=True,ended=time.time(),output_hashes={str(dest/'completed.json'):sha(dest/'completed.json')})
        write(statepath,state)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(statepath,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:register();print('Timing registered, no measurements')
    else:run()
