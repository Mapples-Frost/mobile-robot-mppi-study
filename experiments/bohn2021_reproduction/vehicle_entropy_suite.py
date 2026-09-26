"""Vehicle entropy-only follow-up after the complete original matched result."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART,ROOT,make_env
from run import write,snapshot

OUT=ART/'results/vehicle_entropy';SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def train(job):
    alpha,seed=job;name='vehicle_alpha%s_s%d'%(str(alpha).replace('.','p'),seed);folder=OUT/name
    if (folder/'completed.json').exists():return folder
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_run.py'),'--task','vehicle','--seed',str(seed),'--steps','15000','--out',str(folder),
        '--aligned','--scaled-obs','--batch-size','256','--buffer-size','1000000','--ent-coef',str(alpha),
        '--test-bank',str(ART/'configs/vehicle_validation_bank.json')]
    with open(OUT/(name+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,name
    print(json.dumps({'trained':name}),flush=True);return folder


def evaluate(job):
    folder,mode=job;dest=OUT/'evaluations'/folder.name/mode;dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(SCRIPTS/'optimized_evaluate.py'),'--model-dir',str(folder),'--bank',str(OUT/'holdout_bank.json'),'--out',str(dest)]
    if mode=='no_value':cmd.append('--no-value')
    with open(OUT/(folder.name+'_'+mode+'.log'),'a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    assert r.returncode==0,(folder.name,mode)
    row={'name':folder.name,'mode':mode,'exit_code':0};print(json.dumps(row),flush=True);return row


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((ART/'results/optimized/finalized.json').read_text())['complete']
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',{
        'question':'Does reduced entropy improve vehicle horizon selection after the matched original result?',
        'change':'Only SAC entropy coefficient1→.1/.01. Reward,32Dobservations,terminal,multistart,dynamics and15k budget unchanged.',
        'alphas':[.1,.01],'seeds':[0,1,2],'new_runs':6,'new_transitions':90000,
        'comparators':'All originalalpha1seeds and validation-selected fixedH10seeds0,1,2.',
        'holdout':{'seed':26091711,'n':20},'episodes':480,
        'selection':'All final seeds and both coefficients reported; no checkpoint or test-driven model selection.',
        'fidelity':'Explicit hyperparameter refinement, not exact paper setting.'})
    if not (OUT/'holdout_bank.json').exists():
        env=make_env('vehicle',26091711,aligned=True,scaled_obs=True);np.random.seed(26091711);cases=[]
        for _ in range(20):env.reset();cases.append(snapshot(env))
        write(OUT/'holdout_bank.json',{'seed':26091711,'cases':cases})
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [SCRIPTS/n for n in ['vehicle_entropy_suite.py','optimized_run.py','optimized_runtime.py','optimized_evaluate.py','run.py','runtime.py']]})
    with ThreadPoolExecutor(max_workers=6) as pool:folders=list(pool.map(train,[(a,s) for a in [.1,.01] for s in range(3)]))
    write(OUT/'training_completed.json',{'complete':True,'folders':[str(f) for f in folders]})
    folders += [ART/'results/optimized'/('vehicle_rl_s%d'%s) for s in range(3)]
    folders += [ART/'results/optimized'/('vehicle_fixed_h10_s%d'%s) for s in range(3)]
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(evaluate,[(f,m) for f in folders for m in ['value','no_value']]))
    write(OUT/'evaluation_completed.json',{'complete':True,'jobs':rows})


if __name__=='__main__':main()
