"""Finish audit/report and predetermined replay checks when recovery completes."""
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from paper_grid_recover import ROOT, ART, OUT, REC, read, write, no_competing_run


def main():
    scripts = Path(__file__).parent
    logdir = REC / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    with (REC / 'finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REC / 'delivery_complete.json').exists():
            return
        audit_script = scripts / 'paper_grid_audit_report.py'
        import hashlib
        audit_hash = hashlib.sha256(audit_script.read_bytes()).hexdigest()
        write(REC / 'finish_registration.json', {'audit_sha256': audit_hash,
              'replay_models': ['pendulum_fixed_h45', 'vehicle_fixed_h45'],
              'replay_case': 0, 'purpose': 'Predetermined first scene, one new fixed-H model per task; no selection by performance.',
              'time_epoch': time.time()})
        while not (REC / 'completed.json').exists():
            status = read(REC / 'status.json')
            if status['phase'] == 'failed':
                raise RuntimeError(status)
            time.sleep(30)
        no_competing_run()
        assert hashlib.sha256(audit_script.read_bytes()).hexdigest() == audit_hash
        commands = [('final_cost_audit', [str(audit_script)])]
        for task in ['pendulum', 'vehicle']:
            commands.append(('replay_' + task, [str(scripts / 'check_replay.py'),
                '--run', str(OUT / (task + '_fixed_h45')),
                '--bank', str(ART / 'configs' / (task + '_holdout_bank.json')),
                '--evaluation-dir', 'holdout_value',
                '--audit-out', str(REC / ('replay_' + task + '.json'))]))
        try:
            write(REC / 'finish_status.json', {'phase': 'audit', 'epoch': time.time()})
            for label, args in commands:
                with (logdir / (label + '.log')).open('a') as stream:
                    subprocess.check_call([sys.executable, '-u'] + args, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
                print('complete', label, flush=True)
            write(REC / 'delivery_complete.json', {'passed': True, 'epoch': time.time(),
                   'reconstruction_performance_success': 'See report; completion is not algorithm superiority.',
                   'report': str((ART / 'report/paper_exact_grid_2026-09-23/report.md').relative_to(ROOT))})
            write(REC / 'finish_status.json', {'phase': 'complete', 'epoch': time.time()})
        except BaseException as error:
            write(REC / 'finish_status.json', {'phase': 'failed', 'error': repr(error), 'epoch': time.time()})
            raise


if __name__ == '__main__':
    main()
