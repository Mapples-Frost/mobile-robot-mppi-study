"""Predeclared near-upright scenario study; full fixed grid, all seeds retained."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART,ROOT,make_env
from recoverable_runtime import OUT,install_distribution,prepare_config
from run import write,snapshot

SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def train(job):
    seed,h=job;name='pendulum_rl_s%d'%seed if h is None else 'pendulum_fixed_h%d_s%d'%(h,seed);folder=OUT/name
    if (folder/'completed.json').exists():return {'name':name,'seed':seed,'h':h,'exit_code':0}
    cmd=[sys.executable,'-u',str(SCRIPTS/'recoverable_run.py'),'--task','pendulum','--seed',str(seed),'--steps','15000','--out',str(folder),
        '--aligned','--scaled-obs','--batch-size','256','--buffer-size','1000000','--ent-coef','0.01','--test-bank',str(OUT/'validation_bank.json')]
    if h is not None:cmd+=['--fixed-horizon',str(h)]
    with open(OUT/(name+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,name
    row={'name':name,'seed':seed,'h':h,'exit_code':0};print(json.dumps(row),flush=True);return row


def evaluate(job):
    folder,mode=job;dest=folder/('holdout_'+mode)
    cmd=[sys.executable,'-u',str(SCRIPTS/'recoverable_evaluate.py'),'--model-dir',str(folder),'--bank',str(OUT/'holdout_bank.json'),'--out',str(dest)]
    if mode=='no_value':cmd.append('--no-value')
    with open(OUT/(folder.name+'_'+mode+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,(folder.name,mode)
    row={'name':folder.name,'mode':mode,'exit_code':0};print(json.dumps(row),flush=True);return row


def main():
    prepare_config();install_distribution()
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',{
        'question':'Do physically difficult initial states obscure the adaptive-horizon learning problem?',
        'change':'Only reset distribution narrows: theta±.35rad,omega±.5,pos±.3,v±.5; no rejection sampling or outcome-based deletion.',
        'base':'Full-reference-preview,alpha.01,PSD terminal,15k transitions; rewards,dynamics,constraints,reference process unchanged.',
        'not_original_distribution':True,'not_formal_viability_certificate':True,
        'jobs':'RLseeds0,1,2; fixedH5..50step5seed0; replicate validation-best fixed withseeds1,2.',
        'training_runs':15,'training_transitions':225000,'validation':{'seed':26091708,'n':10},'holdout':{'seed':26091709,'n':30},
        'evaluation_episodes':900,'selection':'All final models. No test-based checkpoints, seeds, or discarded failures. Original broad-distribution results stay reported.'})
    for label,seed,n in [('validation',26091708,10),('holdout',26091709,30)]:
        p=OUT/(label+'_bank.json')
        if p.exists():continue
        env=make_env('pendulum',seed,aligned=True,scaled_obs=True);np.random.seed(seed);cases=[]
        for _ in range(n):env.reset();cases.append(snapshot(env))
        write(p,{'seed':seed,'cases':cases})
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [SCRIPTS/n for n in ['recoverable_runtime.py','recoverable_run.py','recoverable_evaluate.py','recoverable_suite.py','forecast_runtime.py','optimized_runtime.py','run.py','runtime.py']]})
    jobs=[(s,None) for s in range(3)]+[(0,h) for h in range(5,51,5)]
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(train,jobs))
    candidates=[(json.loads((OUT/('pendulum_fixed_h%d_s0'%h)/'eval_value/summary.json').read_text())['mean_total_cost'],h) for h in range(5,51,5)]
    cost,h=min(candidates);write(OUT/'selected_fixed.json',{'h':h,'validation_cost':cost,'candidates':candidates})
    with ThreadPoolExecutor(max_workers=2) as pool:rows+=list(pool.map(train,[(s,h) for s in [1,2]]))
    write(OUT/'training_completed.json',{'complete':True,'jobs':rows})
    with ThreadPoolExecutor(max_workers=8) as pool:ev=list(pool.map(evaluate,[(OUT/r['name'],m) for r in rows for m in ['value','no_value']]))
    write(OUT/'evaluation_completed.json',{'complete':True,'jobs':ev})


if __name__=='__main__':main()
