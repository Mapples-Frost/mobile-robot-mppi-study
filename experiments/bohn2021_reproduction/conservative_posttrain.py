"""Continue after active training: independent audit, banks, validation, selection.

Never opens test. An interrupted/failed prerequisite stops this continuation.
"""
import fcntl
import os
import subprocess
import time
from pathlib import Path
from conservative_iteration import OUT,ROOT,TASKS,verify
from paper_h_soft_probe import read,digest
from run import write


def main():
    verify()
    lock=(OUT/'posttrain.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),active=True,started=time.time(),stage='waiting_for_training',completed=[],
               source_hash=digest(Path(__file__)))
    status_path=OUT/'posttrain_status.json';write(status_path,state)
    while True:
        train=read(OUT/'pipeline_status.json')
        if not train['active']:
            assert train.get('complete'),train
            break
        try:os.kill(train['pid'],0)
        except ProcessLookupError:raise RuntimeError('Training launcher disappeared without final status; inspect logs')
        time.sleep(10)
    scripts=ROOT/'experiments/bohn2021_reproduction';modern=str(ROOT/'.venv/bin/python')
    legacy='/home/mapples/.local/share/bohn2021-python37/bin/python'
    jobs=[]
    for task in TASKS:
        for seed in range(3):
            for r in range(2):
                jobs.append(('audit_%s_s%d_r%d'%(task,seed,r),[modern,str(scripts/'conservative_iteration_audit.py'),
                            '--task',task,'--seed',str(seed),'--round',str(r)]))
    jobs += [('banks',[legacy,str(scripts/'conservative_iteration_evaluate.py'),'--mode','banks']),
             ('validation',[legacy,'-u',str(scripts/'conservative_iteration_evaluate.py'),'--mode','suite','--split','validation']),
             ('select',[modern,str(scripts/'conservative_iteration_select.py')])]
    for label,cmd in jobs:
        log=OUT/('posttrain_%s_%d.log'%(label,time.time_ns()))
        state['stage']=label;state['command']=cmd;state['log']=str(log);write(status_path,state)
        with log.open('w') as stream:r=subprocess.run(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=86400)
        state['completed'].append(dict(stage=label,exit_code=r.returncode,log=str(log)));write(status_path,state)
        if r.returncode:
            state.update(active=False,complete=False,failed=label);write(status_path,state)
            raise SystemExit(r.returncode)
    state.update(active=False,complete=True,ended=time.time());write(status_path,state)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        path=OUT/'posttrain_status.json'
        if path.exists():
            state=read(path)
            if state.get('pid')==os.getpid():
                state.update(active=False,complete=False,ended=time.time(),exception=repr(exc))
                write(path,state)
        raise
