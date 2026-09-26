"""Core-method reconstruction: aligned labels and fixed observation units."""
import argparse
import fcntl
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from runtime import ART,ROOT
from run import write

ap=argparse.ArgumentParser()
ap.add_argument('--fixed',action='store_true')
ap.add_argument('--workers',type=int,default=6)
ap.add_argument('--horizons',type=int,nargs='+')
args=ap.parse_args()
out=ART/'results/refined';out.mkdir(exist_ok=True)
jobs=[]
for task in ['vehicle','pendulum']:
    if args.fixed:
        for h in ([5,10,15] if task=='vehicle' else [20,30,40]):
            if args.horizons and h not in args.horizons:continue
            jobs.append((task,0,h))
    else:
        for seed in range(3):jobs.append((task,seed,None))
plan={'jobs':jobs,'steps_per_job':15000,'aligned':True,'scaled_obs':True,'entropy':'auto',
      'candidate_selection':'Validation only; final checkpoint after 15000 transitions, not best test checkpoint.',
      'method':'Original author SAC + AHMPC + joint 32-step quadratic terminal learning.',
      'changes':'Synchronize TVP with physical clock, correct Bellman stage cost, physical terminal constraint precedence, invertible fixed observation transform.'}
prefix='fixed_extra' if args.horizons else ('fixed' if args.fixed else 'rl')
write(out/(prefix+'_protocol.json'),plan)
def run(job):
    task,seed,h=job
    name=task+('_rl_s%d'%seed if h is None else '_fixed_h%d'%h)
    folder=out/name;folder.mkdir(exist_ok=True)
    lock=open(folder/'launcher.lock','a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    if (folder/'completed.json').exists():
        return {'name':name,'exit_code':0,'reused':True}
    cmd=[sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/run.py'),
         '--task',task,'--seed',str(seed),'--steps','15000','--out',str(out/name),
         '--aligned','--scaled-obs','--test-bank',str(ART/'configs'/(task+'_validation_bank.json'))]
    if h is not None:cmd+=['--fixed-horizon',str(h)]
    with open(out/(name+'.log'),'w') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    return {'name':name,'exit_code':proc.returncode}
with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,jobs))
write(out/(prefix+'_completed.json'),{'jobs':results,'complete':all(r['exit_code']==0 for r in results)})
