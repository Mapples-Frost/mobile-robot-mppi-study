"""Resume-safe validation audit and serial timing continuation; no test access.

Run with legacy Python. Existing training/evaluation workers are not restarted.
Timing runs inside this process so its idle check excludes only itself, never
an arbitrary parent monitor. Any failed prerequisite stops this continuation.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from conservative_iteration import OUT, ROOT, verify
from paper_h_soft_probe import read, digest
from run import write

SCRIPTS=ROOT/'experiments/bohn2021_reproduction'
MODERN=str(ROOT/'.venv/bin/python')


def registration():
    files=[Path(__file__).resolve()]+[SCRIPTS/name for name in (
        'conservative_validation_inventory.py','conservative_candidate_report.py',
        'conservative_fixed_log_audit.py','conservative_iteration_timing.py',
        'conservative_iteration_report.py')]
    result=dict(hashes={str(p):digest(p) for p in files},test_access=False,
        purpose='Operational continuation only; no change to selection, reward, policy or effect thresholds.',
        timing_route='Eligible selection: registered selected validation timing. Ineligible selection: registered separate all-candidate descriptive timing.',
        prerequisites=['complete posttrain','complete baseline completion','all added fixed training logs independently audited','complete validation inventory'],
        confirmation='Never automatically freezes or opens test; completed reports must be inspected first.')
    path=OUT/'validation_finish_registration.json'
    if path.exists():assert read(path)==result
    else:write(path,result)
    return digest(path)


def main(wait=False):
    verify();registration_hash=registration()
    lock=(OUT/'validation_finish.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/'validation_finish_status.json'
    state=dict(pid=os.getpid(),active=True,started=time.time(),stage='waiting_for_baselines',completed=[],
        registration_hash=registration_hash,source_hash=digest(Path(__file__)))
    if status.exists():
        old=read(status)
        if old.get('complete'):
            for p,h in old['output_hashes'].items():assert digest(Path(p))==h
            print('Already completed; verified outputs, no repeat timing',flush=True);return
        archive=OUT/('validation_finish_previous_%d.json'%time.time_ns());write(archive,old)
    write(status,state)
    deadline=time.monotonic()+86400

    def stage(name):
        state.update(stage=name);write(status,state);print(name,flush=True)

    def command(label,args):
        stage(label)
        log=OUT/('finish_%s_%d.log'%(label,time.time_ns()))
        state['current_log']=str(log);write(status,state)
        with log.open('w') as stream:
            result=subprocess.run([MODERN]+[str(a) for a in args],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
        state['completed'].append(dict(stage=label,exit_code=result.returncode,log=str(log)));write(status,state)
        assert result.returncode==0,(label,result.returncode,str(log))

    try:
        while True:
            pending=[]
            for filename,script in [('posttrain_status.json','conservative_posttrain.py'),('baseline_completion_status.json','conservative_complete_baselines.py')]:
                d=read(OUT/filename)
                proc=Path('/proc')/str(d['pid'])/'cmdline'
                alive=proc.exists() and script.encode() in proc.read_bytes()
                if d['active']:
                    assert alive,('Prerequisite vanished; inspect before restart',filename)
                    pending.append(filename)
                else:
                    assert d.get('complete'),(filename,d)
                    if alive:pending.append(filename)
            if not pending:break
            assert wait,('Prerequisites still active',pending)
            assert time.monotonic()<deadline,'24h waiting limit; prerequisites were not killed'
            time.sleep(10)
        command('protocol_audit',[SCRIPTS/'conservative_protocol_audit.py'])
        for folder in sorted((OUT/'extra_fixed').glob('*')):
            if (folder/'instrumentation_completed.json').exists():
                command('audit_'+folder.name,[SCRIPTS/'conservative_fixed_log_audit.py','--folder',folder])
        command('validation_inventory',[SCRIPTS/'conservative_validation_inventory.py'])
        command('all_candidate_report',[SCRIPTS/'conservative_candidate_report.py'])
        selection=read(OUT/'validation_selection.json')
        state['selection_hash']=digest(OUT/'validation_selection.json')
        all_candidates=not selection['selection_eligible']
        state['timing_route']='all_candidates' if all_candidates else 'selected'
        stage('waiting_for_serial_timing')
        from conservative_iteration_timing import idle, run
        while True:
            try:idle();break
            except AssertionError as exc:
                assert wait,repr(exc)
                assert time.monotonic()<deadline,'24h limit waiting for idle timing host'
                state['idle_conflicts']=repr(exc);write(status,state);time.sleep(10)
        stage('serial_validation_timing_'+state['timing_route'])
        run('validation',all_candidates=all_candidates)
        if all_candidates:
            command('all_candidate_timed_report',[SCRIPTS/'conservative_candidate_report.py','--timing'])
        command('registered_effect_report',[SCRIPTS/'conservative_iteration_report.py'])
        command('budget',[SCRIPTS/'conservative_budget.py'])
        final_paths=[OUT/'validation_inventory/coverage_audit.json',OUT/'validation_delivery/effect_gate.json',
            OUT/'all_candidate_delivery/report.json',
            OUT/('timing_validation_all_candidates' if all_candidates else 'timing_validation')/'completed.json']
        if all_candidates:final_paths.append(OUT/'all_candidate_delivery_timed/report.json')
        state.update(active=False,complete=True,ended=time.time(),test_opened=False,
            output_hashes={str(p):digest(p) for p in final_paths})
        write(status,state);print('Validation audit and timing complete; test remains sealed',flush=True)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(status,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:print(registration())
    else:main(a.wait)
