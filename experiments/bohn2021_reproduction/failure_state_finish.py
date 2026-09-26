"""Wait for audited diagnosis; conditionally run frozen serial timing and audit."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from failure_state_protocol import ROOT, SCRIPTS, OUT, REG, MODERN, read, write, sha, verify


def register():
    verify()
    paths=[Path(__file__),SCRIPTS/'failure_state_timing.py',SCRIPTS/'failure_state_timing_audit.py',
           OUT/'timing_registration.json',REG,OUT/'serialization_repair/amendment.json']
    data=dict(source_hashes={str(p.resolve()):sha(p) for p in paths},
              prerequisite='Completed source pipeline, actual controller PID absent, full audit and delivery hashes pass.',
              conditional='No timing when diagnostic all-seed safety/physical screen is empty. Otherwise run all4 pendulum arms using previously registered order and source.',
              runtime='Timing executes in this legacy Python process; no waiting parent Python experiment process. Independent timing audit starts only after measurements finish.',
              test_access=False,goal_completion=False)
    path=OUT/'finish_registration.json'
    if path.exists():
        original=read(path);expected=dict(original['source_hashes'])
        timing_amendment=OUT/'timing_controller_repair/amendment.json'
        if timing_amendment.exists():
            amended=read(timing_amendment)
            assert amended['original_timing_registration_sha256']==sha(OUT/'timing_registration.json')
            for name,h in amended['archived_files'].items():assert sha(ROOT/name)==h
            allowed={str(SCRIPTS/'failure_state_timing.py'),str(SCRIPTS/'failure_state_timing_audit.py')}
            assert set(amended['changed_sources'])==allowed
            for name,item in amended['changed_sources'].items():
                assert expected[name]==item['old_sha256']
                expected[name]=item['new_sha256']
        finish_amendment=OUT/'timing_controller_repair/finish_amendment.json'
        if finish_amendment.exists():
            amended=read(finish_amendment)
            assert amended['original_finish_registration_sha256']==sha(path)
            assert amended['timing_amendment_sha256']==sha(timing_amendment)
            assert amended['registration_only'] and not amended['execution_logic_changed']
            for name,h in amended['archived_files'].items():assert sha(ROOT/name)==h
            name=str(Path(__file__).resolve());item=amended['changed_source']
            assert item['path']==name and expected[name]==item['old_sha256']
            assert sha(ROOT/item['archive'])==item['old_sha256']
            expected[name]=item['new_sha256']
        assert data['source_hashes']==expected
        assert {k:v for k,v in data.items() if k!='source_hashes'}=={k:v for k,v in original.items() if k!='source_hashes'}
    else:write(path,data)
    return data


def main():
    reg=register();lock=(OUT/'finish.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'finish_status.json'
    if path.exists():
        old=read(path)
        if old.get('complete'):
            for p,h in old['output_hashes'].items():assert sha(Path(p))==h
            print('Finish already complete');return
        assert not (Path('/proc')/str(old['pid'])/'cmdline').exists(),'Prior finish controller still exists'
        write(OUT/('previous_finish_status_%d.json'%time.time_ns()),old)
    state=dict(pid=os.getpid(),active=True,complete=False,stage='waiting_for_full_diagnosis',
               started=time.time(),test_access=False,goal_complete=False)
    write(path,state);deadline=time.monotonic()+86400
    try:
        while True:
            assert time.monotonic()<deadline,'24-hour wait bound; no process killed'
            s=read(OUT/'status.json');proc=Path('/proc')/str(s['pid'])/'cmdline'
            live=proc.exists() and b'failure_state_pipeline.py' in proc.read_bytes()
            if s['active']:
                assert live,('Source controller disappeared before terminal state',s['pid'])
                time.sleep(10);continue
            assert s['complete'],('Source diagnosis failed',s.get('exception'))
            if live:time.sleep(2);continue
            break
        for p,h in s['output_hashes'].items():assert sha(Path(p))==h
        d=read(OUT/'delivery/report.json');assert d['passed'] and read(OUT/'audit_full.json')['passed']
        for p,h in d['hashes'].items():assert sha(Path(p))==h
        outputs=[OUT/'delivery/report.json',OUT/'audit_full.json']
        if d['timing_screen_candidates']:
            state.update(stage='waiting_for_idle_host',timing_screen_candidates=d['timing_screen_candidates']);write(path,state)
            from failure_state_timing import idle,run
            while True:
                try:idle();break
                except AssertionError as exc:
                    assert time.monotonic()<deadline,'24-hour idle wait limit'
                    state['idle_conflicts']=repr(exc);write(path,state);time.sleep(10)
            state['stage']='serial_diagnostic_timing';write(path,state);run()
            state['stage']='independent_timing_audit';write(path,state)
            log=OUT/('timing_audit_%d.log'%time.time_ns())
            with log.open('w') as f:
                result=subprocess.run([MODERN,str(SCRIPTS/'failure_state_timing_audit.py')],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,timeout=14400)
            state.update(audit_log=str(log),audit_exit_code=result.returncode);write(path,state)
            assert result.returncode==0,('Timing audit failed',str(log))
            outputs.append(OUT/'timing_delivery/report.json')
            state['timing_performed']=True
        else:
            state.update(stage='screen_failed_no_timing',timing_performed=False,
                         reason='No certificate candidate meets all3-seed safety, physical-cost NI2%, and intervention screen. No speed claim.')
        state.update(active=False,complete=True,ended=time.time(),output_hashes={str(p):sha(p) for p in outputs})
        write(path,state);print(json.dumps({k:v for k,v in state.items() if k not in ('output_hashes','idle_conflicts')},indent=2))
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:register();print('Conditional finish registered')
    else:main()
