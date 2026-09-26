"""Predeclared matched comparison; complete fixed-H grid and fresh holdout."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from runtime import ART,ROOT,make_env
from run import write,snapshot

OUT=ART/'results/optimized';SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def prepare():
    OUT.mkdir(exist_ok=True)
    jobs=[(task,s,None) for task in ['vehicle','pendulum'] for s in range(3)]
    jobs += [(task,0,h) for task in ['vehicle','pendulum'] for h in range(5,51,5)]
    protocol={'status':'predeclared','training_steps':15000,'jobs':jobs,
        'selection':'All final 15000-step checkpoints. No seed/checkpoint selection on holdout.',
        'fixed_comparators':'H5..50 step5; each trains its own terminal value on 15000 transitions. Grid seed0; best validation fixed H replicated with seeds1,2 before holdout.',
        'reward':'Unchanged paper performance, H cost and early-constraint-termination cost; vehicle lambdaH=.001 lambdaC=2; pendulum .003,10.',
        'hyperparameters':{'gamma':.97,'rho':.97,'lr':.0003,'batch':256,'replay':1000000,'entropy':1.,'reward_scale':{'vehicle':.3,'pendulum':.6},'actor':[32,32],'critics':[256,256],'tau':.005,'nstep_terminal':32},
        'extensions':['PSD full quadratic in relative current reference error','terminal prediction endpoint reference','32D vehicle/9D pendulum MPC-available context','deterministic 3-start vehicle/1-start pendulum solver','zero-terminal reset warmup'],
        'unchanged':['plant dynamics','physical input/state constraints','obstacle uncertainty process','episode lengths','horizon range1..50','SAC continuous action rounding'],
        'remaining_reconstruction_assumptions':['initial distributions','feature choices','learning_starts100','time_awareTrue'],
        'validation':'Original separate 10-case validation bank; used only to choose which fixed H receives additional seeds.',
        'holdout':{'seed':26091704,'cases_per_task':20,'new':True,'terminal_modes':['learned','zero']},
        'computation':'Report solver calls and elapsed time; H proxy is preserved but cannot establish actual speedup.',
        'failure_handling':'Log any failed job; no silent candidate deletion.',
        'hard_timeout_seconds_per_job':18000}
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',protocol)
    for task in ['vehicle','pendulum']:
        p=OUT/(task+'_holdout_bank.json')
        if p.exists():continue
        env=make_env(task,26091704,aligned=True,scaled_obs=True);np.random.seed(26091704)
        cases=[]
        for _ in range(20):env.reset();cases.append(snapshot(env))
        write(p,{'seed':26091704,'purpose':'Fresh holdout, not used for choosing extensions or checkpoints','cases':cases})
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SCRIPTS/n for n in ['optimized_runtime.py','optimized_run.py','optimized_evaluate.py','optimized_suite.py','run.py','runtime.py']]})
    return jobs


def train(job):
    task,seed,h=job;name=task+('_rl_s%d'%seed if h is None else '_fixed_h%d_s%d'%(h,seed))
    folder=OUT/name
    if (folder/'completed.json').exists():return {'name':name,'exit_code':0,'reused':True}
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_run.py'),'--task',task,'--seed',str(seed),
        '--steps','15000','--out',str(folder),'--aligned','--scaled-obs','--batch-size','256',
        '--buffer-size','1000000','--ent-coef','1.0','--test-bank',str(ART/'configs'/(task+'_validation_bank.json'))]
    if h is not None:cmd+=['--fixed-horizon',str(h)]
    with open(OUT/(name+'.log'),'a') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    row={'name':name,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row


def evaluate(job):
    folder,value=job;spec=json.loads((folder/'manifest.json').read_text());task=spec['task']
    dest=folder/('holdout_value' if value else 'holdout_no_value')
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_evaluate.py'),'--model-dir',str(folder),
        '--bank',str(OUT/(task+'_holdout_bank.json')),'--out',str(dest)]
    if not value:cmd.append('--no-value')
    with open(OUT/(folder.name+'_holdout_'+str(value)+'.log'),'a') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    row={'name':folder.name,'value':value,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    jobs=prepare()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(train,jobs))
    write(OUT/'grid_completed.json',{'jobs':results,'complete':all(r['exit_code']==0 for r in results)})
    assert all(r['exit_code']==0 for r in results)
    replication=[];selected={}
    for task in ['vehicle','pendulum']:
        candidates=[]
        for h in range(5,51,5):
            s=json.loads((OUT/('%s_fixed_h%d_s0'%(task,h))/'eval_value/summary.json').read_text())
            candidates.append((s['mean_total_cost'],h))
        cost,h=min(candidates);selected[task]={'h':h,'validation_cost':cost,'all_candidates':candidates}
        replication += [(task,s,h) for s in [1,2]]
    write(OUT/'validation_selected_fixed.json',selected)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:more=list(pool.map(train,replication))
    assert all(r['exit_code']==0 for r in more)
    write(OUT/'training_completed.json',{'jobs':results+more,'complete':True,'training_runs':30,'transitions':450000})
    folders=[OUT/r['name'] for r in results+more]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:ev=list(pool.map(evaluate,[(f,v) for f in folders for v in [True,False]]))
    write(OUT/'holdout_completed.json',{'jobs':ev,'complete':all(r['exit_code']==0 for r in ev)})
    assert all(r['exit_code']==0 for r in ev)


if __name__=='__main__':main()
