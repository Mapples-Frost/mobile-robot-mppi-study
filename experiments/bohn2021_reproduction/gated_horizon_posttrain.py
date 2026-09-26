"""Continue only the known live search; audit, freeze and validate, never open test."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from gated_horizon_search import OUT,freeze
from gated_horizon_evaluate import register
from runtime import ROOT
from paper_h_soft_probe import read,digest
from run import write
from gated_horizon_amendment import registration as amended_registration

LEGACY='/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN=str(ROOT/'.venv/bin/python')
SCRIPTS=ROOT/'experiments/bohn2021_reproduction'


def registration():
    freeze();register()
    paths=[Path(__file__).resolve(),SCRIPTS/'gated_horizon_validation_audit.py',OUT/'evaluation_registration.json']
    value=dict(hashes={str(p):digest(p) for p in paths},scope='Wait for registered running search, independent training audit, freeze all6 policies, full84 validation conditions, independent validation audit, fixed nomination. Never reads test outcomes.',test_access=False)
    path=OUT/'posttrain_registration.json'
    if path.exists():amended_registration(path,value)
    else:assert not (OUT/'evaluations').exists();write(path,value)


def run():
    registration();lock=(OUT/'posttrain.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'posttrain_status.json'
    assert not path.exists(),'Inspect existing continuation before restarting'
    state=dict(pid=os.getpid(),active=True,stage='waiting_for_registered_search',completed=[],started=time.time());write(path,state)
    try:
        deadline=time.monotonic()+172800
        while True:
            s=read(OUT/'status.json')
            if not s['active']:assert s.get('complete'),s;break
            process=Path('/proc')/str(s['pid'])/'cmdline'
            assert process.exists() and b'gated_horizon_search.py' in process.read_bytes(),'Registered search missing; do not duplicate'
            assert time.monotonic()<deadline,'Wait limit exceeded; search not killed'
            time.sleep(15)
        work=[('audit_train',MODERN,'gated_horizon_audit.py',[]),
              ('freeze_policies',MODERN,'gated_horizon_evaluate.py',['--mode','freeze-policies']),
              ('validation',LEGACY,'gated_horizon_evaluate.py',['--mode','suite']),
              ('audit_validation',MODERN,'gated_horizon_validation_audit.py',[]),
              ('nominate_baselines',MODERN,'gated_horizon_select.py',[])]
        for label,python,script,args in work:
            log=OUT/('posttrain_%s_%d.log'%(label,time.time_ns()));cmd=[python,'-u',str(SCRIPTS/script)]+args
            state.update(stage=label,command=cmd,log=str(log));write(path,state)
            with log.open('w') as stream:r=subprocess.run(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=172800)
            assert r.returncode==0,(label,r.returncode,str(log))
            state['completed'].append(dict(stage=label,exit_code=0,log=str(log)));write(path,state)
        state.update(active=False,complete=True,ended=time.time(),test_opened=False);write(path,state)
    except BaseException as exc:
        state.update(active=False,complete=False,exception=repr(exc),ended=time.time());write(path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:registration()
    else:run()
