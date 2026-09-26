"""Frozen evaluation, physical cost audit and solver budget accounting."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from runtime import ART,imports
from optimized_runtime import install_terminal,make_env
from mechanism_probe import checked_step
from run import write,weights_hash


def evaluate(model,env,cases,out,fixed_horizon,use_value,terminal_weights=None):
    task='vehicle' if env.control_system.controller.mpc.model.n_x==3 else 'pendulum'
    w,b=model.policy_tf.get_mpc_vfn_weights_and_biases() if terminal_weights is None else terminal_weights
    if not use_value:w,b=[np.zeros_like(a) for a in w],[np.zeros_like(a) for a in b]
    env.set_value_function_weights_and_biases(w,b)
    rows=[]
    for j,case in enumerate(cases):
        obs=env.reset(**copy.deepcopy(case));initial=copy.deepcopy(env.control_system.current_state)
        trace=[]
        while True:
            start=time.perf_counter()
            h=fixed_horizon if fixed_horizon is not None else int(np.clip(np.rint(model.predict(obs,deterministic=True)[0][0]),1,50))
            obs,done,row=checked_step(env,task,h)
            row.update(env.optimized_solver_info)
            row['elapsed_s']=time.perf_counter()-start
            row['reward']=-row['cost'];trace.append(row)
            if done:break
        write(out/('trace_%02d.json'%j),trace)
        rows.append({'episode':j,'initial_state':initial,'steps':len(trace),'termination':trace[-1]['termination'],
            'performance_cost':sum(r['performance'] for r in trace),'computation_cost':sum(r['compute'] for r in trace),
            'constraint_cost':sum(r['constraint'] for r in trace),'total_cost':sum(r['cost'] for r in trace),
            'discounted_cost':sum(.97**t*r['cost'] for t,r in enumerate(trace)),
            'mean_horizon':float(np.mean([r['horizon'] for r in trace])),
            'solver_failure_steps':sum(not r['solver_success'] for r in trace),
            'solver_calls':sum(r['solver_calls'] for r in trace),
            'max_constraint_residual':max(r['max_constraint_residual'] for r in trace),
            'mean_step_s':float(np.mean([r['elapsed_s'] for r in trace]))})
    write(out/'summary.json',{'episodes':rows,'mean_total_cost':float(np.mean([r['total_cost'] for r in rows])),
        'constraint_episodes':sum(r['termination']=='constraint' for r in rows),
        'goal_episodes':sum(r['termination']=='goal' for r in rows),'terminal_value':use_value,
        'physical_cost_and_bounds_verified':True,'computation_cost_is_h_proxy':True})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--no-value',action='store_true')
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if (args.out/'completed.json').exists():return
    args.out.mkdir(exist_ok=True)
    spec=json.loads((args.model_dir/'manifest.json').read_text());task=spec['task']
    install_terminal(task)
    env=make_env(task,926,spec['fixed_horizon']);_,SAC,_=imports()
    model=SAC.load(str(args.model_dir/'model.zip'),env=env);before=weights_hash(model)
    cases=json.loads(args.bank.read_text())['cases']
    evaluate(model,env,cases,args.out,spec['fixed_horizon'],not args.no_value)
    assert before==weights_hash(model)
    write(args.out/'completed.json',{'frozen':True,'weights_sha256':before,'bank':str(args.bank),
        'episodes':len(cases),'physical_costs_verified':True})


if __name__=='__main__':main()
