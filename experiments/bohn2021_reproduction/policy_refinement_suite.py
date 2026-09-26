"""Predeclared single-factor entropy refinement, separate fresh holdout."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from runtime import ROOT, ART, make_env
from run import write, snapshot

OUT=ART/'results/policy_refinement';SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def train(job):
    alpha,seed=job;name='pendulum_alpha%s_s%d'%(str(alpha).replace('.','p'),seed);folder=OUT/name
    if (folder/'completed.json').exists():return {'name':name,'alpha':alpha,'seed':seed,'exit_code':0}
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_run.py'),'--task','pendulum','--seed',str(seed),
        '--steps','15000','--out',str(folder),'--aligned','--scaled-obs','--batch-size','256',
        '--buffer-size','1000000','--ent-coef',str(alpha),'--test-bank',str(ART/'configs/pendulum_validation_bank.json')]
    with open(OUT/(name+'.log'),'a') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert p.returncode==0,name
    result={'name':name,'alpha':alpha,'seed':seed,'exit_code':p.returncode};print(json.dumps(result),flush=True);return result


def evaluate(job):
    folder,mode=job;dest=OUT/'evaluations'/folder.name/mode
    dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_evaluate.py'),'--model-dir',str(folder),
        '--bank',str(OUT/'holdout_bank.json'),'--out',str(dest)]
    if mode=='no_value':cmd.append('--no-value')
    with open(OUT/(folder.name+'_'+mode+'.log'),'a') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert p.returncode==0,(folder.name,mode)
    row={'name':folder.name,'mode':mode,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row


def main():
    OUT.mkdir(exist_ok=True)
    protocol={'question':'Does fixed entropy=1 prevent reliable fine horizon choices under the reconstructed pendulum distribution?',
        'intervention':'Only entropy coefficient changes to .1 or .01; performance/H/constraint reward and scaling remain unchanged.',
        'baseline_entropy':1.,'candidate_entropy':[.1,.01],'training_seeds':[0,1,2],
        'steps_per_model':15000,'new_training_runs':6,'new_transitions':90000,
        'unchanged':['optimized runtime','terminal representation and joint updates','actor/critic architecture','Adam learning rate','batch256','physical environments','H1..50','nstep32','gamma/rho .97'],
        'evaluation':'All final models and all original alpha1 models plus selected fixedH30 seeds0,1,2, both terminal modes, on fresh20 scenes seed26091705.',
        'new_independent_scenes':20,'new_evaluation_episodes':480,
        'selection':'Report both coefficients and every seed; no checkpoint selection or deleted failures. Do not evaluate until all new training finishes.',
        'claim_limit':'Improvement experiment, explicitly not paper-exact hyperparameters; entropy and real control return differ.',
        'provenance':'Initiated after completed optimized pendulum holdout and validation-only frozen H interventions; vehicle original round continues.'}
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',protocol)
    if not (OUT/'holdout_bank.json').exists():
        env=make_env('pendulum',26091705,aligned=True,scaled_obs=True);np.random.seed(26091705);cases=[]
        for _ in range(20):env.reset();cases.append(snapshot(env))
        write(OUT/'holdout_bank.json',{'seed':26091705,'purpose':'New holdout for entropy-only refinement','cases':cases})
    if not (OUT/'source_hashes.json').exists():
        write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [SCRIPTS/n for n in ['optimized_runtime.py','optimized_run.py','optimized_evaluate.py','run.py','runtime.py','policy_refinement_suite.py']]})
    with ThreadPoolExecutor(max_workers=6) as pool:trained=list(pool.map(train,[(a,s) for a in [.1,.01] for s in range(3)]))
    write(OUT/'training_completed.json',{'complete':True,'jobs':trained})
    folders=[OUT/r['name'] for r in trained]
    folders += [ART/'results/optimized'/('pendulum_rl_s%d'%s) for s in range(3)]
    folders += [ART/'results/optimized'/('pendulum_fixed_h30_s%d'%s) for s in range(3)]
    with ThreadPoolExecutor(max_workers=6) as pool:ev=list(pool.map(evaluate,[(f,m) for f in folders for m in ['value','no_value']]))
    write(OUT/'evaluation_completed.json',{'complete':True,'jobs':ev})


if __name__=='__main__':main()
