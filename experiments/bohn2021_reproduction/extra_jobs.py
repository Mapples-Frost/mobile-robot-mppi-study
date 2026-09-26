"""Use spare memory for selected jobs already present in the main queue."""
import json
import argparse
import subprocess
import sys
from runtime import ROOT,ART

out=ART/'results/full'
p=argparse.ArgumentParser();p.add_argument('--horizons',nargs='+',type=int,default=[45,50]);args=p.parse_args()
assert all(h in range(5,51,5) for h in args.horizons)
jobs=[(task,h) for h in args.horizons for task in ['pendulum','vehicle']]
active=[]
for task,h in jobs:
    name=task+'_fixed_h'+str(h)
    log=open(out/(name+'.extra.log'),'a')
    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/run.py'),
        '--task',task,'--fixed-horizon',str(h),'--seed','0','--steps','15000','--out',str(out/name)],
        cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
    active.append((name,proc,log))
result=[]
for name,proc,log in active:
    try:code=proc.wait(timeout=10800)
    except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=30)
    log.close();result.append({'name':name,'exit_code':code})
(out/('extra_jobs_'+'_'.join(map(str,args.horizons))+'_completed.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
