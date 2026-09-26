"""Evaluate the six completed RL runs with two bounded workers."""
import concurrent.futures
import json
import subprocess
import sys
import time
from runtime import ART,ROOT


def job(task,seed):
    name=task+'_rl_s'+str(seed);folder=ART/'results/full'/name
    deadline=time.monotonic()+10800
    while not (folder/'completed.json').exists():
        if time.monotonic()>deadline:return {'run':name,'error':'training completion timeout'}
        time.sleep(10)
    with open(ART/'results/full'/(name+'.curve.log'),'a') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/checkpoint_eval.py'),
                '--task',task,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,timeout=10800,cwd=str(ROOT))
    return {'run':name,'exit_code':result.returncode}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(job,t,s) for s in range(3) for t in ['pendulum','vehicle']]
    results=[f.result() for f in futures]
(ART/'results/full/curve_suite_completed.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
