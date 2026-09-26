"""All-seed diagnostic rollouts; no learned update, selection or test access."""
import argparse
import copy
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path
from failure_state_protocol import (ROOT, ART, OLD, OUT, REG, TASKS, bank, source_model,
                                    old_folder, read, write, sha, verify)
from failure_state_policy import decide, policies
from runtime import imports
from conservative_canonical_reset import make_env
from conservative_solver_recovery import install
from branch_calibration_run import meter, observed_step
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from relative_policy_features import context
from gated_horizon_search import case_metrics
from run import weights_hash


def strip(trace, gates=True):
    result = copy.deepcopy(trace)
    for r in result:
        r.pop('timing', None)
        if gates:
            r.pop('gate', None)
        for a in r['recovery']['attempts']:
            a.pop('solver_s', None)
    return result


def location(phase, task, seed, policy):
    return OUT / phase / ('%s_s%d_%s' % (task, seed, policy))


def job(phase, task, seed, policy):
    spec = verify()
    assert task in TASKS and seed in range(3) and policy in policies(task)
    assert phase in ('smoke', 'full')
    folder = location(phase, task, seed, policy)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = folder / 'completed.json'
    if completed.exists():
        for name, value in read(completed)['hashes'].items():
            assert sha(Path(name)) == value
        print('Verified existing completion ' + folder.name, flush=True)
        return
    assert not list(folder.glob('attempt_*.json')) and not (folder/'solver_attempts.json').exists(), 'Partial attempt needs explicit recovery audit'
    if phase == 'full':
        assert read(OUT / 'audit_smoke.json')['passed']
    case_ids = spec['smoke_cases'][task][str(seed)] if phase == 'smoke' else list(range(24))
    cases = read(bank(task, seed))['cases']
    inherited = old_folder(task, seed, policy)
    base_hashes = {str(REG):sha(REG), str(bank(task, seed)):sha(bank(task, seed)),
                   str(source_model(task, seed)/'model.zip'):sha(source_model(task, seed)/'model.zip')}
    if phase == 'full' and inherited:
        done = read(inherited/'completed.json')
        for name,value in done['hashes'].items():
            assert sha(Path(name)) == value
        base_hashes.update(done['hashes'])
        base_hashes[str(inherited/'completed.json')] = sha(inherited/'completed.json')
        summary = dict(task=task,seed=seed,policy=policy,phase=phase,reused=True,trace_folder=str(inherited),
                       episodes=done['result']['episodes'],case_ids=case_ids,repeats=1,
                       new_step_calls=0,new_reset_calls=0,new_solver_calls=0,
                       note='Reference to already audited training traces; no new simulation or timing.')
        write(folder/'summary.json',summary)
        base_hashes[str(folder/'summary.json')] = sha(folder/'summary.json')
        write(completed,dict(passed=True,hashes=base_hashes,reused=True))
        print('Reused original '+folder.name,flush=True)
        return
    modelpath = source_model(task, seed)
    _, SAC, _ = imports()
    model = SAC.load(str(modelpath/'model.zip'))
    assert weights_hash(model) == read(modelpath/'completed.json')['final_hash']
    terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    terminal_hash = sha(modelpath/'model.zip')
    model.sess.close()
    selected = read(OLD/'train'/('vehicle_s%d'%seed)/'policy.json') if task=='vehicle' and policy=='selected' else None
    env = make_env(task, seed, aligned=True, scaled_obs=True)
    counts = meter(env, folder)
    env.set_value_function_weights_and_biases(*terminal)
    state = install(env.control_system.controller.mpc, folder)
    write(folder/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
          threads={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
          started=time.time(),post_hoc=True,not_formal_timing=True))
    episodes=[]
    repeats=2 if phase=='smoke' else 1
    for repeat in range(repeats):
        for cid in case_ids:
            case=cases[cid]
            state.update(enabled=False,events=[],case=cid,step=-1)
            env.reset(**copy.deepcopy(case));state['enabled']=True
            trace=[]
            # Canonical old reference defines identical scored initial state.
            reference=OLD/'train'/('%s_s%d'%(task,seed))/'fixed'/('r0_trace_%02d.json'%cid)
            initial=read(reference)[0]['previous_state']
            assert env.control_system.current_state==initial
            for t in range(env.max_steps):
                state['step']=t;ctx=context(env,task)
                h,gate=decide(task,policy,ctx,selected)
                _,terminated,row=observed_step(env,task,h,case,t)
                row.update(policy_context=ctx,gate=gate,recovery=state['events'][-1]);trace.append(row)
                if terminated:break
            assert terminated
            path=folder/('r%d_trace_%02d.json'%(repeat,cid))
            # Save complete raw data before any audit, so an audit failure retains evidence.
            write(path,trace)
            audit_trace(task,case,trace);audit_context(task,case,trace,0)
            if repeat:
                assert strip(trace,False)==strip(read(folder/('r0_trace_%02d.json'%cid)),False)
            if inherited:
                assert strip(trace)==strip(read(inherited/('r0_trace_%02d.json'%cid)))
            base_hashes[str(path)]=sha(path)
            m=case_metrics(task,trace)
            m.update(case=cid,repeat=repeat,intervention_steps=sum(r['gate']['triggered'] for r in trace))
            episodes.append(m)
            write(folder/'progress.json',dict(task=task,seed=seed,policy=policy,phase=phase,pid=os.getpid(),
                  complete_episodes=len(episodes),expected_episodes=len(case_ids)*repeats,
                  steps=counts['step_calls'],resets=counts['reset_calls']))
            print(json.dumps(dict(task=task,seed=seed,policy=policy,repeat=repeat,case=cid,steps=m['steps'],
                  cost=m['total_cost'],intervention_steps=m['intervention_steps'])),flush=True)
    state['enabled']=False
    assert sha(modelpath/'model.zip')==terminal_hash
    summary=dict(task=task,seed=seed,policy=policy,phase=phase,reused=False,trace_folder=str(folder),
                 episodes=episodes,case_ids=case_ids,repeats=repeats,new_step_calls=counts['step_calls'],
                 new_reset_calls=counts['reset_calls'],new_solver_calls=state['counts']['solve_attempts'],
                 solver_counts=state['counts'],source_terminal_hash=terminal_hash)
    write(folder/'summary.json',summary)
    for p in list(folder.glob('attempt_*.json'))+[folder/n for n in ('summary.json','solver_attempts.json','solver_calls.jsonl','environment.json')]:
        base_hashes[str(p)]=sha(p)
    write(completed,dict(passed=True,hashes=base_hashes,reused=False,exact_replay=phase=='smoke'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('smoke','full'),required=True)
    ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--policy',required=True);a=ap.parse_args();job(a.phase,a.task,a.seed,a.policy)
