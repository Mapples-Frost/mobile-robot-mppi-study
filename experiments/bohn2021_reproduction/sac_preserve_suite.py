"""Two-worker scheduler; frozen protocol, all seeds and checkpoints retained."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import numpy as np
from sac_preserve_common import OUT,ARMS,LEGACY,MODERN,prepare,verify,write,read

AGENT=str(Path(__file__).with_name('sac_preserve_agent.py'))
WORKER=str(Path(__file__).with_name('sac_preserve_worker.py'))


def launch(jobs,seed):
    jobs=list(jobs);np.random.RandomState(seed).shuffle(jobs)
    def run(job):
        runtime,script,args=job
        name='_'.join(args).replace('--','')
        path=OUT/'logs'/(name+'.log');path.parent.mkdir(exist_ok=True)
        with path.open('a') as f:
            proc=subprocess.run([runtime,'-u',script]+args,stdout=f,stderr=subprocess.STDOUT,timeout=21600)
        assert proc.returncode==0,(args,proc.returncode,str(path))
        print('complete',args,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))


def evaluation_jobs(split):
    names=['%s_s%d_15000'%(arm,s) for arm in ARMS for s in range(3)]
    if split=='validation':
        names+=['pretrained_s%d'%s for s in range(3)]
        names+=['%s_s%d_%d'%(arm,s,step) for arm in ARMS for s in range(3) for step in [5000,10000]]
    names+=['rule','fixed25','value_teacher']
    jobs=[(MODERN,AGENT,['--evaluate',name,'--split',split]) for name in names]
    if split=='holdout':jobs += [(LEGACY,WORKER,['%s_s%d'%(arm,s)]) for arm in ['plain','teacher'] for s in range(3)]
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare()
    if not a.prepare:
        assert read(OUT/'smoke.json')['passed']
        launch([(MODERN,AGENT,['--pretrain',str(s)]) for s in range(3)],260919201)
        launch([(MODERN,AGENT,['--train',arm,'--seed',str(s)]) for arm in ARMS for s in range(3)],260919202)
        launch(evaluation_jobs('validation'),260919203)
        # No learned checkpoint selection or tuning from validation results.
        write(OUT/'validation_complete.json',{'selection':'None; final15000 for every arm/seed, all intermediate results retained.'})
        launch(evaluation_jobs('holdout'),260919204)
        verify();write(OUT/'completed.json',{'models':9,'validation_conditions':33,'holdout_conditions':18,'SAC_joint_updates':9*14901})
