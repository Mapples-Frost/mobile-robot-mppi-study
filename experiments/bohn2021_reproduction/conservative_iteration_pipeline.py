"""Resumable two-round collector/fitter; evaluation is a separate gated stage."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from conservative_iteration import OUT, ROOT, TASKS, verify
from paper_h_soft_probe import read, digest
from run import write


def main():
    verify()
    lock=(OUT/'pipeline.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    legacy=Path('/home/mapples/.local/share/bohn2021-python37/bin/python')
    modern=ROOT/'.venv/bin/python'
    scripts=ROOT/'experiments/bohn2021_reproduction'
    for task in TASKS:
        done=read(OUT/('smoke_'+task)/'smoke_fit_completed.json')
        assert done['inference_reload_passed']
    state=dict(pid=os.getpid(),active=True,started=time.time(),stages=[],
               launcher_hash=digest(Path(__file__)))
    write(OUT/'pipeline_status.json',state)
    jobs=[]
    for round_id in range(2):
        jobs.append(('collect_r%d'%round_id,[str(legacy),'-u',str(scripts/'conservative_iteration.py'),
                    '--mode','suite','--round',str(round_id)]))
        for task in TASKS:
            for seed in range(3):
                jobs.append(('fit_%s_s%d_r%d'%(task,seed,round_id),[str(modern),'-u',
                    str(scripts/'conservative_policy_model.py'),'--task',task,'--seed',str(seed),'--round',str(round_id)]))
    for label,cmd in jobs:
        log=OUT/('pipeline_%s_%d.log'%(label,time.time_ns()))
        state['current']=dict(label=label,command=cmd,log=str(log))
        write(OUT/'pipeline_status.json',state)
        with log.open('w') as stream:r=subprocess.run(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=86400)
        state['stages'].append(dict(label=label,exit_code=r.returncode,log=str(log)))
        write(OUT/'pipeline_status.json',state)
        if r.returncode:
            state.update(active=False,complete=False,failed=label)
            write(OUT/'pipeline_status.json',state)
            raise SystemExit(r.returncode)
    state.update(active=False,complete=True,ended=time.time())
    write(OUT/'pipeline_status.json',state)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        path=OUT/'pipeline_status.json'
        if path.exists():
            state=read(path)
            if state.get('pid')==os.getpid():
                state.update(active=False,complete=False,ended=time.time(),exception=repr(exc))
                write(path,state)
        raise
