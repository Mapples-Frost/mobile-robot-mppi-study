"""Unchanged author fixed-H training with complete durable transition logging.

Called only for a newly nominated fixed H lacking independent seeds1/2.
Instrumentation is read-only and does not generate future TVPs or consume RNG.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import run
from run import serial,write
from paper_h_soft_probe import digest,read
from latency_tree_evaluation_spec import OUT,verify_evaluation as verify


def main():
    verify()
    args=sys.argv[1:]
    smoke='--instrumentation-smoke' in args
    if smoke:
        args.remove('--instrumentation-smoke')
        sys.argv=[sys.argv[0]]+args
    task=args[args.index('--task')+1];seed=int(args[args.index('--seed')+1]);h=int(args[args.index('--fixed-horizon')+1])
    dest=Path(args[args.index('--out')+1])
    steps=300 if smoke else 15000
    if smoke:
        assert dest.parent==OUT/'extra_fixed_smoke'
        expected_bank=OUT/('smoke_'+task)/'train_bank.json'
    else:
        selected=read(OUT/'baseline_selection.json')
        assert dict(task=task,seed=seed,h=h,training_steps=15000) in selected['pending_independent_baselines']
        expected_bank=OUT/'banks'/(task+'_validation_bank.json')
    assert int(args[args.index('--steps')+1])==steps and '--no-online-value' not in args
    assert '--test-bank' in args and Path(args[args.index('--test-bank')+1])==expected_bank
    if (dest/'completed.json').exists():
        assert (dest/'instrumentation_completed.json').exists();return
    dest.mkdir(parents=True,exist_ok=True)
    assert not list(dest.glob('transitions_*.jsonl')),'Inspect interrupted training before restart'
    attempt=str(time.time_ns());stream=(dest/('transitions_'+attempt+'.jsonl')).open('w')
    reset_stream=(dest/('resets_'+attempt+'.jsonl')).open('w')
    counts=dict(step_attempts=0,step_completed=0,reset_attempts=0,reset_completed=0,environment_constructions=0,pid=os.getpid())
    ledger=dest/('instrumentation_attempt_'+attempt+'.json');write(ledger,counts)
    original=run.make_env
    def logline(f,data):
        f.write(json.dumps(data,default=serial,allow_nan=False)+'\n');f.flush()
    def make(*args,**kwargs):
        counts['environment_constructions']+=1;write(ledger,counts)
        env=original(*args,**kwargs);base_reset=env.reset;base_step=env.step
        def reset(**kw):
            counts['reset_attempts']+=1;write(ledger,counts)
            result=base_reset(**kw);c=env.control_system
            logline(reset_stream,dict(reset=counts['reset_attempts'],state_before_warmup=copy.deepcopy(c.history['state'][0]),
                state_after_warmup=copy.deepcopy(c.current_state),observation=result,
                available_tvps={k:copy.deepcopy(v.values) for k,v in c.tvps.items()},
                object_noise_seed=copy.deepcopy(getattr(c.controller,'object_noise_seed',None))))
            counts['reset_completed']+=1;write(ledger,counts);return result
        def step(action):
            counts['step_attempts']+=1;write(ledger,counts)
            c=env.control_system;previous=copy.deepcopy(c.current_state);before=env.get_observation().copy();t=env.steps_count
            obs,reward,done,info=base_step(action)
            logline(stream,dict(index=counts['step_attempts'],reset=counts['reset_attempts'],t=t,
                previous_state=previous,observation=before,requested_action=action,executed_horizon=info['executed_horizon'],
                state=copy.deepcopy(c.current_state),input=copy.deepcopy(c.controller.current_input),
                next_observation=obs,reward=reward,done=done,termination=info.get('termination'),
                performance=info['reward/performance'],compute=info['reward/computation'],constraint=info['reward/constraint'],
                solver_success=info['solver_success'],mpc_forecasts=copy.deepcopy(c.controller.history['tvp'][-1])))
            counts['step_completed']+=1;write(ledger,counts);return obs,reward,done,info
        env.reset=reset;env.step=step;return env
    run.make_env=make
    try:run.main()
    finally:stream.close();reset_stream.close()
    assert counts['step_attempts']==counts['step_completed'] and counts['reset_attempts']==counts['reset_completed']
    paths=[dest/('transitions_'+attempt+'.jsonl'),dest/('resets_'+attempt+'.jsonl'),ledger,dest/'completed.json',Path(__file__)]
    write(dest/'instrumentation_completed.json',dict(passed=True,counts=counts,training_steps=steps,smoke=smoke,
        extra_evaluation_steps=counts['step_completed']-steps,
        hashes={str(p):digest(p) for p in paths},
        scope='Every explicit call after constructor; includes final validation with/without terminal. No intermediate validation checkpoint selection.'))


if __name__=='__main__':main()
