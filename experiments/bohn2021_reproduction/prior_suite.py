"""Riccati-initialized joint learning with a matched complete fixed-H grid."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART,ROOT,make_env
import recoverable_runtime as distribution
from riccati_terminal_probe import prior
from run import write,snapshot

OUT=ART/'results/prior_refinement';SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def train(job):
    seed,h=job;name='pendulum_rl_s%d'%seed if h is None else 'pendulum_fixed_h%d_s%d'%(h,seed);folder=OUT/name
    if (folder/'completed.json').exists():return {'name':name,'seed':seed,'h':h,'exit_code':0}
    cmd=[sys.executable,'-u',str(SCRIPTS/'prior_run.py'),'--task','pendulum','--seed',str(seed),'--steps','15000','--out',str(folder),
        '--aligned','--scaled-obs','--batch-size','256','--buffer-size','1000000','--ent-coef','0.01','--test-bank',str(OUT/'validation_bank.json')]
    if h is not None:cmd+=['--fixed-horizon',str(h)]
    with open(OUT/(name+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,name
    row={'name':name,'seed':seed,'h':h,'exit_code':0};print(json.dumps(row),flush=True);return row


def evaluate(job):
    folder,mode=job;dest=folder/('holdout_'+mode)
    cmd=[sys.executable,'-u',str(SCRIPTS/'prior_evaluate.py'),'--model-dir',str(folder),'--bank',str(OUT/'holdout_bank.json'),'--out',str(dest)]
    if mode=='no_value':cmd.append('--no-value')
    with open(OUT/(folder.name+'_'+mode+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,(folder.name,mode)
    row={'name':folder.name,'mode':mode,'exit_code':0};print(json.dumps(row),flush=True);return row


def main():
    distribution.OUT=OUT;distribution.prepare_config();distribution.install_distribution()
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',{
        'question':'Does a physically informed terminal initialization enable useful short horizons and improve joint learning?',
        'change':'Only PSD terminal kernel and bias initial values use discounted discrete Riccati local solution;32step joint learning continues.',
        'base':'Near-upright distribution,full-reference-preview,alpha.01; dynamics/rewards/budgets unchanged.',
        'training_runs':15,'training_transitions':225000,'seeds':[0,1,2],'fixed_grid':list(range(5,51,5)),
        'validation':'Same near-upright validation10 from previous distribution study, no holdout selection.',
        'holdout':{'seed':26091710,'n':30},'evaluation_episodes':900,
        'selection':'All final checkpoints; independently trained fixed grid and replicate validation-best H to3seeds.',
        'limitation':'Model-derived prior explicitly extends paper method; stationary-reference local quadratic is not global constrained value.'})
    if not (OUT/'validation_bank.json').exists():write(OUT/'validation_bank.json',json.loads((ART/'results/recoverable_distribution/validation_bank.json').read_text()))
    if not (OUT/'holdout_bank.json').exists():
        env=make_env('pendulum',26091710,aligned=True,scaled_obs=True);np.random.seed(26091710);cases=[]
        for _ in range(30):env.reset();cases.append(snapshot(env))
        write(OUT/'holdout_bank.json',{'seed':26091710,'cases':cases})
    write(OUT/'prior.json',prior()[2])
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [SCRIPTS/n for n in ['prior_run.py','prior_evaluate.py','prior_suite.py','recoverable_runtime.py','riccati_terminal_probe.py','forecast_runtime.py','optimized_runtime.py','run.py','runtime.py']]})
    jobs=[(s,None) for s in range(3)]+[(0,h) for h in range(5,51,5)]
    with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(train,jobs))
    candidates=[(json.loads((OUT/('pendulum_fixed_h%d_s0'%h)/'eval_value/summary.json').read_text())['mean_total_cost'],h) for h in range(5,51,5)]
    cost,h=min(candidates);write(OUT/'selected_fixed.json',{'h':h,'validation_cost':cost,'candidates':candidates})
    with ThreadPoolExecutor(max_workers=2) as pool:rows+=list(pool.map(train,[(s,h) for s in [1,2]]))
    write(OUT/'training_completed.json',{'complete':True,'jobs':rows})
    with ThreadPoolExecutor(max_workers=6) as pool:ev=list(pool.map(evaluate,[(OUT/r['name'],m) for r in rows for m in ['value','no_value']]))
    write(OUT/'evaluation_completed.json',{'complete':True,'jobs':ev})


if __name__=='__main__':main()
