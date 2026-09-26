"""Serial timing and audited reporting after all registered training/validation finishes."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from gated_horizon_search import OUT,freeze
from paper_h_soft_probe import read,digest
from run import write
from runtime import ROOT
from gated_horizon_amendment import registration as amended_registration

SCRIPTS=ROOT/'experiments/bohn2021_reproduction'
MODERN=str(ROOT/'.venv/bin/python')


def registration():
    freeze()
    paths=[Path(__file__).resolve()]+[SCRIPTS/n for n in ('gated_horizon_timing.py','gated_horizon_timing_audit.py',
        'gated_horizon_report.py','gated_horizon_budget.py','conservative_iteration_timing.py')]
    paths += [OUT/n for n in ('effect_registration.json','timing_registration.json','baseline_completion_registration.json','environment/legacy.json','environment/audit.json')]
    value=dict(hashes={str(p):digest(p) for p in paths},purpose='Operational continuation only. No changes to training, baselines, policies, splits, criteria or bootstrap.',
        prerequisites='All known search/posttrain/baseline controllers terminal and complete, absent from process table; passing validation audit and no pending fixed seeds.',
        timing='Run registered serial timing in this legacy-Python process, so idle detection excludes only itself. Wait for other experiment Python processes to finish.',
        results='All-condition validation report, both timing repetitions, independent timing audit, final effect report and full interrupted-work budget.',test_access=False)
    p=OUT/'validation_finish_registration.json'
    if p.exists():amended_registration(p,value)
    else:write(p,value)
    return digest(p)


def main(wait):
    registered=registration();lock=(OUT/'validation_finish.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'validation_finish_status.json'
    if path.exists():
        old=read(path)
        if old.get('complete'):
            for p,h in old['output_hashes'].items():assert digest(Path(p))==h
            print('Complete outputs verified; no repeat timing',flush=True);return
        assert not (Path('/proc')/str(old['pid'])/'cmdline').exists(),'Prior continuation still exists'
        write(OUT/('validation_finish_previous_%d.json'%time.time_ns()),old)
    state=dict(pid=os.getpid(),active=True,started=time.time(),stage='waiting_for_baselines',completed=[],registration_hash=registered);write(path,state)
    def stage(label):state.update(stage=label);write(path,state);print(label,flush=True)
    def command(label,script):
        stage(label);log=OUT/('finish_%s_%d.log'%(label,time.time_ns()));state['log']=str(log);write(path,state)
        with log.open('w') as stream:r=subprocess.run([MODERN,str(SCRIPTS/script)],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
        state['completed'].append(dict(stage=label,exit_code=r.returncode,log=str(log)));write(path,state)
        assert r.returncode==0,(label,r.returncode,str(log))
    deadline=time.monotonic()+172800
    try:
        while True:
            pending=[]
            for name,script in [('status.json','gated_horizon_search.py'),('posttrain_status.json','gated_horizon_posttrain.py'),('baseline_completion_status.json','gated_horizon_complete_baselines.py')]:
                s=read(OUT/name);p=Path('/proc')/str(s['pid'])/'cmdline';alive=p.exists() and script.encode() in p.read_bytes()
                if s['active']:assert alive,('Prerequisite vanished',name,s['pid']);pending.append(name)
                else:
                    assert s.get('complete'),(name,s)
                    if alive:pending.append(name)
            if not pending:break
            assert wait,('Prerequisites running',pending)
            assert time.monotonic()<deadline,'48h wait limit, prerequisite processes not killed'
            time.sleep(15)
        assert read(OUT/'audit_validation.json')['passed']
        assert not read(OUT/'baseline_selection.json')['pending_independent_baselines']
        command('untimed_validation_report','gated_horizon_report.py')
        from gated_horizon_timing import idle,run
        stage('waiting_for_serial_host')
        while True:
            try:idle();break
            except AssertionError as exc:
                assert wait,repr(exc);assert time.monotonic()<deadline,'48h serial-host wait limit'
                state['idle_conflicts']=repr(exc);write(path,state);time.sleep(15)
        stage('serial_validation_timing');run('validation')
        command('timing_audit','gated_horizon_timing_audit.py')
        command('effect_report','gated_horizon_report.py')
        command('budget','gated_horizon_budget.py')
        outputs=[OUT/n for n in ('audit_validation.json','audit_timing_validation.json','timing_validation/completed.json',
            'validation_delivery/effect_gate.json','validation_delivery/all_conditions.csv','validation_delivery/all_episodes.csv','validation_delivery/all_comparisons.csv','budget/snapshot.json')]
        state.update(active=False,complete=True,ended=time.time(),test_opened=False,output_hashes={str(p):digest(p) for p in outputs});write(path,state)
        print('Validation timing and reports complete; test remains sealed',flush=True)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    if a.register:print(registration())
    else:main(a.wait)
