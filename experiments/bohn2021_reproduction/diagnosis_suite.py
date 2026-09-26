"""Predeclared, one-factor diagnostic runs on a separate validation bank."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART,ROOT,make_env
from run import snapshot,write


def create_banks():
    for task in ['vehicle','pendulum']:
        for name,seed,count in [('validation',26091701,10),('holdout',26091702,20)]:
            path=ART/'configs'/('%s_%s_bank.json'%(task,name))
            if path.exists():continue
            env=make_env(task,seed);np.random.seed(seed)
            cases=[]
            for _ in range(count):
                env.reset();cases.append(snapshot(env))
            write(path,{'seed':seed,'purpose':name,'cases':cases,
                        'provenance':'Fresh independent bank, never used for training.'})


def main():
    create_banks()
    out=ART/'results/diagnosis';out.mkdir(exist_ok=True)
    variants={'aligned':['--aligned'],'scaled':['--scaled-obs'],
              'fixed_entropy':['--ent-coef','1.0'],'no_online_value':['--no-online-value']}
    write(out/'protocol.json',{'variants':variants,'task':'vehicle','seed':1,'training_steps':15000,
        'selection':'Lowest mean validation cost, with zero collisions preferred; no old test checkpoint selection.',
        'next_stage':'Replicate selected reconstruction on seeds 0,1,2 and both tasks; evaluate final models and fixed baselines on fresh holdout.',
        'fidelity':'Original SAC actor and critic, continuous rounded H, original MPC, 32-step terminal learning. Each diagnostic changes only the named factor.'})
    def run(name,options):
        dest=out/('vehicle_'+name+'_s1')
        cmd=[sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/run.py'),
             '--task','vehicle','--seed','1','--steps','15000','--out',str(dest),
             '--test-bank',str(ART/'configs/vehicle_validation_bank.json')]+options
        with open(out/(name+'.log'),'w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        return {'name':name,'exit_code':proc.returncode}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=list(pool.map(lambda item:run(*item),variants.items()))
    write(out/'suite_completed.json',{'jobs':jobs,'complete':all(j['exit_code']==0 for j in jobs)})


if __name__=='__main__':main()
