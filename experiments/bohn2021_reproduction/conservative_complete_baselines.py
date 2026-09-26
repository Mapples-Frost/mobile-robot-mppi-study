"""Complete nominated independent fixed-H seeds, then validation selection.

No method/search changes and no test access. Registration precedes validation.
"""
import argparse
import fcntl
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from conservative_iteration import OUT, ROOT, TASKS, BASE, verify
from conservative_iteration_evaluate import grid_model
from paper_h_soft_probe import read, digest
from run import write

SCRIPTS=ROOT/'experiments/bohn2021_reproduction'
LEGACY='/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN=str(ROOT/'.venv/bin/python')
SETTINGS=dict(aligned=True,scaled_obs=True,ent_coef='1.0',no_online_value=False,batch_size=256,buffer_size=1000000)


def register():
    verify()
    paths=[Path(__file__).resolve(),SCRIPTS/'conservative_fixed_train.py',SCRIPTS/'run.py',
           SCRIPTS/'conservative_iteration_evaluate.py',SCRIPTS/'conservative_iteration_select.py',
           SCRIPTS/'conservative_iteration_report.py',OUT/'protocol.json']
    for task in TASKS:
        for h in range(5,51,5):
            source=grid_model(task,h,0);m=read(source/'manifest.json')
            assert m['adaptations']==SETTINGS and m['steps']==15000 and m['fixed_horizon']==h
            paths += [source/'manifest.json',source/'completed.json',source/'model.zip']
    spec=dict(hashes={str(p):digest(p) for p in paths},settings=SETTINGS,training_steps=15000,
        seeds=[1,2],workers=2,final_checkpoint_only=True,
        trigger='Only pending_independent_baselines from frozen validation selection; no additional H search.',
        extra_validation='24 validation scenes with/without terminal emitted by training wrapper; then registered grid evaluation24. All calls charged.',
        smoke='Each task completed300-step instrumentation smoke with its original primary H and original smoke training bank.',
        limits='Preserves existing15k terminal-training opportunity; adaptive branch simulation remains extra, not equal-budget superiority.',
        test_access=False)
    path=OUT/'baseline_completion_registration.json'
    if path.exists():assert read(path)==spec
    else:
        assert not (OUT/'evaluations').exists(),'Register before validation outcomes'
        write(path,spec)


def command(j):
    task,seed,h=j['task'],j['seed'],j['h']
    assert task in TASKS and seed in (1,2) and h in range(5,51,5)
    assert h!=BASE[task] and j['training_steps']==15000
    dest=grid_model(task,h,seed)
    assert dest.parent==OUT/'extra_fixed'
    return [LEGACY,'-u',str(SCRIPTS/'conservative_fixed_train.py'),
        '--task',task,'--seed',str(seed),'--fixed-horizon',str(h),'--steps','15000',
        '--out',str(dest),'--test-bank',str(OUT/'banks'/(task+'_validation.json')),
        '--eval-episodes','24','--aligned','--scaled-obs','--ent-coef','1.0',
        '--batch-size','256','--buffer-size','1000000']


def run(wait):
    register()
    lock=(OUT/'baseline_completion.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'baseline_completion_status.json'
    state=dict(pid=os.getpid(),active=True,started=time.time(),stage='waiting_for_posttrain',completed=[])
    write(path,state)
    try:
        deadline=time.monotonic()+86400
        while True:
            p=read(OUT/'posttrain_status.json')
            if not p['active']:
                assert p.get('complete'),p
                break
            assert wait,'Posttrain still active'
            proc=Path('/proc')/str(p['pid'])/'cmdline'
            assert proc.exists() and b'conservative_posttrain.py' in proc.read_bytes(),'Posttrain process missing: inspect logs, do not restart automatically'
            assert time.monotonic()<deadline,'24h waiting limit reached; training process was not killed'
            time.sleep(10)
        assert read(OUT/'validation_status.json')['complete']
        nomination=read(OUT/'baseline_selection.json')
        for p,h in nomination['hashes'].items():assert digest(Path(p))==h
        state['nomination_hash']=digest(OUT/'baseline_selection.json')
        pending=nomination['pending_independent_baselines']
        for task in TASKS:
            smoke=OUT/'extra_fixed_smoke'/('%s_h%d_s0'%(task,BASE[task]))
            done=read(smoke/'instrumentation_completed.json');assert done['passed'] and done['smoke']
            for p,h in done['hashes'].items():assert digest(Path(p))==h
        state.update(stage='independent_fixed_training',jobs=pending);write(path,state)
        def launch(j):
            label='%s_h%d_s%d'%(j['task'],j['h'],j['seed'])
            log=OUT/('complete_fixed_%s_%d.log'%(label,time.time_ns()))
            with log.open('w') as stream:
                result=subprocess.run(command(j),cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=28800)
            assert result.returncode==0,(j,str(log),result.returncode)
            dest=grid_model(j['task'],j['h'],j['seed'])
            m=read(dest/'manifest.json');d=read(dest/'completed.json');i=read(dest/'instrumentation_completed.json')
            assert m['adaptations']==SETTINGS and m['seed']==j['seed'] and m['fixed_horizon']==j['h']
            assert d['status']=='complete' and d['steps']==15000 and d['weights_changed'] and d['evaluation_frozen']
            assert i['passed'] and not i['smoke'] and i['training_steps']==15000
            for p,h in i['hashes'].items():assert digest(Path(p))==h
            return dict(job=j,exit_code=0,log=str(log),completed_hash=digest(dest/'completed.json'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(launch,j) for j in pending]):
                state['completed'].append(future.result());write(path,state)
        jobs=[]
        for j in pending:
            jobs.append(('eval_%s_h%d_s%d'%(j['task'],j['h'],j['seed']),[LEGACY,'-u',
                str(SCRIPTS/'conservative_iteration_evaluate.py'),'--mode','job','--split','validation',
                '--task',j['task'],'--family','grid','--seed',str(j['seed']),'--h',str(j['h'])]))
        jobs += [('select',[MODERN,str(SCRIPTS/'conservative_iteration_select.py')]),
                 ('report',[MODERN,str(SCRIPTS/'conservative_iteration_report.py')])]
        for label,cmd in jobs:
            log=OUT/('baseline_%s_%d.log'%(label,time.time_ns()))
            state.update(stage=label,command=cmd,log=str(log));write(path,state)
            with log.open('w') as stream:
                result=subprocess.run(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
            assert result.returncode==0,(label,result.returncode,str(log))
            state['completed'].append(dict(stage=label,exit_code=0,log=str(log)));write(path,state)
        state.update(active=False,complete=True,ended=time.time(),test_opened=False);write(path,state)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(path,state)
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    if a.register:register()
    else:run(a.wait)
