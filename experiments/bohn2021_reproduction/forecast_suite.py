"""Matched full-reference-preview experiment, fresh holdout, no seed selection."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ROOT,ART,make_env
from run import write,snapshot

OUT=ART/'results/forecast_refinement';SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def train(seed):
    folder=OUT/('pendulum_forecast_s%d'%seed)
    cmd=[sys.executable,'-u',str(SCRIPTS/'forecast_run.py'),'--task','pendulum','--seed',str(seed),'--steps','15000',
        '--out',str(folder),'--aligned','--scaled-obs','--batch-size','256','--buffer-size','1000000','--ent-coef','0.01',
        '--test-bank',str(ART/'configs/pendulum_validation_bank.json')]
    with open(OUT/(folder.name+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,folder.name
    print(json.dumps({'trained':folder.name}),flush=True);return folder


def evaluate(job):
    folder,mode=job;dest=OUT/'evaluations'/folder.name/mode;dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(SCRIPTS/'forecast_evaluate.py'),'--model-dir',str(folder),'--bank',str(OUT/'holdout_bank.json'),'--out',str(dest)]
    if mode=='no_value':cmd.append('--no-value')
    with open(OUT/(folder.name+'_'+mode+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,(folder.name,mode)
    row={'name':folder.name,'mode':mode,'exit_code':r.returncode};print(json.dumps(row),flush=True);return row


def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',{
        'question':'Can complete MPC-known reference preview improve H selection beyond sparse preview?',
        'intervention':'Only actor observation expands from9 to56; terminal representation and training unchanged.',
        'base_entropy':.01,'choice_basis':'Most stable validation seed costs in previous entropy experiment; previous holdout also inspected and disclosed.',
        'seeds':[0,1,2],'steps':15000,'new_runs':3,'transitions':45000,
        'comparators':'Sparse preview alpha.01 and fixedH30, each3seeds; both terminal modes.',
        'new_holdout_seed':26091706,'independent_scenes':20,'evaluation_episodes':360,
        'selection':'All final checkpoints, no seed selection; new holdout not used to change the protocol.',
        'fidelity':'Hypothesis improvement, not exact paper implementation.'})
    if not (OUT/'holdout_bank.json').exists():
        env=make_env('pendulum',26091706,aligned=True,scaled_obs=True);np.random.seed(26091706);cases=[]
        for _ in range(20):env.reset();cases.append(snapshot(env))
        write(OUT/'holdout_bank.json',{'seed':26091706,'cases':cases})
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SCRIPTS/n for n in ['forecast_runtime.py','forecast_run.py','forecast_evaluate.py','forecast_suite.py','optimized_runtime.py','optimized_evaluate.py','run.py','runtime.py']]})
    with ThreadPoolExecutor(max_workers=3) as pool:folders=list(pool.map(train,range(3)))
    write(OUT/'training_completed.json',{'complete':True,'folders':[str(f) for f in folders]})
    folders += [ART/'results/policy_refinement'/('pendulum_alpha0p01_s%d'%s) for s in range(3)]
    folders += [ART/'results/optimized'/('pendulum_fixed_h30_s%d'%s) for s in range(3)]
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,[(f,m) for f in folders for m in ['value','no_value']]))
    write(OUT/'evaluation_completed.json',{'complete':True,'jobs':rows})


if __name__=='__main__':main()
