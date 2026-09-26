"""Bounded job queue for both tasks, 3 SAC seeds, and all 10 fixed horizons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from runtime import ART,ROOT


def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--steps',type=int,default=15000)
    p.add_argument('--job-timeout',type=int,default=10800)
    args=p.parse_args()
    out=ART/'results'/'full';out.mkdir(exist_ok=True)
    lock=out/'suite.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    # Precompute shared banks serially to avoid same-task workers racing on reset.
    from run import bank
    for task in ['pendulum','vehicle']:
        bank(task)
    jobs=[]
    # RL first, then interleave the two tasks' fixed-horizon baselines.
    for seed in range(3):
        for task in ['pendulum','vehicle']:
            jobs.append({'task':task,'seed':seed,'horizon':None,'name':task+'_rl_s'+str(seed)})
    for h in range(5,51,5):
        for task in ['pendulum','vehicle']:
            jobs.append({'task':task,'seed':0,'horizon':h,'name':task+'_fixed_h'+str(h)})
    plan={'jobs':jobs,'steps_per_job':args.steps,'total_training_steps':args.steps*len(jobs),
          'test_episodes_per_job':10,'terminal_ablation':True,'workers':args.workers,
          'timing_note':'Parallel training and evaluation; latency values are telemetry, not a fair compute benchmark.'}
    save(out/'plan.json',plan)
    pending=[j for j in jobs if not (out/j['name']/'completed.json').exists()]
    completed=[j['name'] for j in jobs if (out/j['name']/'completed.json').exists()]
    active=[];failed=[];start=time.time()
    try:
        while pending or active:
            while pending and len(active)<args.workers:
                job=pending.pop(0);dest=out/job['name']
                cmd=[sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/run.py'),
                     '--task',job['task'],'--steps',str(args.steps),'--seed',str(job['seed']),'--out',str(dest)]
                if job['horizon'] is not None:cmd+=['--fixed-horizon',str(job['horizon'])]
                log=open(out/(job['name']+'.log'),'a')
                proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                active.append({'job':job,'proc':proc,'log':log,'start':time.time()})
            for a in active[:]:
                code=a['proc'].poll()
                if code is None and time.time()-a['start']>args.job_timeout:
                    a['proc'].terminate();code=a['proc'].wait(timeout=30)
                    failed.append({'name':a['job']['name'],'reason':'hard timeout'})
                if code is not None:
                    name=a['job']['name'];a['log'].close();active.remove(a)
                    if code==0 and (out/name/'completed.json').exists():completed.append(name)
                    elif not any(f['name']==name for f in failed):failed.append({'name':name,'exit_code':code})
            status={'pid':os.getpid(),'elapsed_s':time.time()-start,'completed':completed,'failed':failed,
                    'pending':len(pending),'active':[{'name':a['job']['name'],'pid':a['proc'].pid,'elapsed_s':time.time()-a['start']} for a in active]}
            save(out/'status.json',status)
            if pending or active:time.sleep(5)
        save(out/'suite_completed.json',dict(status,status='complete' if not failed else 'incomplete'))
    finally:
        lock.unlink()


if __name__=='__main__':main()
