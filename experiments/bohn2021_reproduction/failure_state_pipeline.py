"""Bounded smoke -> audit -> all-case diagnosis -> audit -> report continuation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from failure_state_protocol import ROOT, SCRIPTS, OUT, REG, LEGACY, MODERN, read, write, sha, verify, jobs


def main(smoke_only=False):
    verify();lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=OUT/'status.json'
    if statepath.exists():
        old=read(statepath)
        assert not (Path('/proc')/str(old['pid'])/'cmdline').exists(),'Existing controller live'
        if old.get('complete') and (smoke_only or not old.get('smoke_only')):
            for p,h in old['output_hashes'].items():assert sha(Path(p))==h
            print('Complete pipeline verified');return
        # Explicit preservation is required; partial conditions are still rejected by worker.
        write(OUT/('previous_status_%d.json'%time.time_ns()),old)
    state=dict(pid=os.getpid(),active=True,complete=False,smoke_only=smoke_only,started=time.time(),
               registration_hash=sha(REG),stage='smoke',completed=[],test_access=False)
    write(statepath,state)
    def command(label,cmd):
        log=OUT/(label+'_'+str(time.time_ns())+'.log')
        with log.open('w') as stream:
            p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            code=p.wait(timeout=14400)
        return dict(label=label,exit_code=code,pid=p.pid,log=str(log))
    def stage(name):
        state['stage']=name;write(statepath,state);print(name,flush=True)
    try:
        phases=['smoke'] if smoke_only else ['smoke','full']
        for phase in phases:
            stage(phase)
            if not (OUT/('audit_'+phase+'.json')).exists():
                def worker(j):
                    task,seed,policy=j
                    return command('%s_%s_s%d_%s'%(phase,task,seed,policy),[LEGACY,'-u',str(SCRIPTS/'failure_state_run.py'),
                                   '--phase',phase,'--task',task,'--seed',str(seed),'--policy',policy])
                # Start at most two jobs, and stop dispatching on the first error.
                # Already-running jobs finish and remain fully accounted.
                iterator=iter(jobs());failed_dispatch=False
                with ThreadPoolExecutor(max_workers=2) as pool:
                    active={pool.submit(worker,next(iterator)) for _ in range(2)}
                    while active:
                        finished,active=wait(active,return_when=FIRST_COMPLETED)
                        for future in finished:
                            r=future.result();state['completed'].append(r);write(statepath,state);print(json.dumps(r),flush=True)
                            failed_dispatch=failed_dispatch or r['exit_code']!=0
                        while not failed_dispatch and len(active)<2:
                            try:j=next(iterator)
                            except StopIteration:break
                            active.add(pool.submit(worker,j))
                failed=[r for r in state['completed'] if r['label'].startswith(phase+'_') and r['exit_code']!=0]
                assert not failed,failed
            stage('audit_'+phase)
            result=command('audit_'+phase,[MODERN,str(SCRIPTS/'failure_state_audit.py'),'--phase',phase])
            state['completed'].append(result);write(statepath,state);assert result['exit_code']==0,result
        if not smoke_only:
            stage('delivery');result=command('delivery',[MODERN,str(SCRIPTS/'failure_state_report.py')])
            state['completed'].append(result);write(statepath,state);assert result['exit_code']==0,result
        outputs=[OUT/'audit_smoke.json']+([] if smoke_only else [OUT/'audit_full.json',OUT/'delivery/report.json'])
        state.update(active=False,complete=True,ended=time.time(),output_hashes={str(p):sha(p) for p in outputs})
        write(statepath,state);print('Diagnostic pipeline complete; no timing or test automatically launched',flush=True)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(statepath,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();main(a.smoke_only)
