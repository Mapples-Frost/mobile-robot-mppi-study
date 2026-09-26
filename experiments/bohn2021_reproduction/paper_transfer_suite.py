"""Serial stages, two MPC workers, idle wait before imports/simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from paper_transfer_common import (OUT, ROOT, LEGACY, DOMAINS, LEARNED, CONTROLS,
                                   FIXED, prepare, verify, read, write, bank, metrics)


def blockers():
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(p.parent.name)
            if pid == os.getpid():
                continue
            args = p.read_bytes().decode('utf-8', 'replace').split('\0')
        except OSError:
            continue
        tokens = {Path(a).name for a in args if a}
        if tokens & {'kh_ladder_sweep.py', 'kh_ladder_roll.py', 'kh_backfill.sh'}:
            found.append(pid)
    return found


def idle(wait):
    while True:
        busy = blockers()
        if not busy:
            return
        write(OUT/'status.json', {'phase': 'waiting_for_measured_delay_and_backfill',
                                 'blocker_pids': busy, 'checked_epoch': time.time(),
                                 'new_simulation_started': False})
        if not wait:
            raise SystemExit('Measured-delay study/backfill active; use --wait-for-idle')
        time.sleep(30)


def command(args, name):
    if blockers():
        raise RuntimeError('Measured-delay workload appeared; preserve results and stop dispatch')
    log = OUT/'logs'/(name+'.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:
        p = subprocess.run([LEGACY, '-u']+args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=21600)
    if p.returncode:
        raise RuntimeError('Job failed (see %s): %s' % (log, args))


def evaluate(jobs, split, seed):
    jobs = list(jobs)
    random.Random(seed).shuffle(jobs)
    write(OUT/'status.json', {'phase': split, 'jobs': len(jobs), 'started_epoch': time.time()})
    worker = str(Path(__file__).with_name('paper_transfer_worker.py'))
    def one(job):
        domain, name = job
        command([worker, '--split', split, '--domain', domain, '--name', name], split+'_'+domain+'_'+name)
        print('complete', split, domain, name, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, jobs))


def select():
    choices = {}
    for domain in DOMAINS:
        ranked = []
        for name in FIXED:
            eps = read(OUT/'evaluation/validation'/domain/name/'completed.json')['episodes']
            key = (sum(e['physical_failure'] for e in eps),
                   sum(e['solver_failures'] for e in eps)/sum(e['steps'] for e in eps),
                   sum(e['raw_cost'] for e in eps)/len(eps), int(name.split('_')[1]))
            ranked.append((key, name))
        ranked.sort()
        choices[domain] = {'name': ranked[0][1], 'ranking': [{'name': n, 'key': k} for k, n in ranked]}
    write(OUT/'selection.json', choices)
    return choices


def check():
    b = bank(260923399, 2)
    assert b == bank(260923399, 2)
    assert b != bank(260923398, 2)
    for i in range(2):
        a = b['domains']['redraw_broad_600'][i]
        short = b['domains']['redraw_broad_100'][i]
        assert a['case'] == short['case'] and a['steps']==600 and short['steps']==100
        assert a['case']['state'] == b['domains']['plateau_broad_600'][i]['case']['state']
        assert a['case']['tvp'] == b['domains']['redraw_near_600'][i]['case']['tvp']
    # Early failures must retain the unexecuted constant offset separately.
    row = dict(cost=989.775, performance=-.3, compute=.075, constraint=990.,
               termination='constraint', solver_success=False, horizon=25)
    m = metrics([row], 100)
    assert abs(m['adjusted_cost']-(m['physical_cost']+m['H_cost']+m['failure_penalty']+m['unexecuted_offset'])) < 1e-9
    print('Bank pairing, determinism and short-episode early-failure accounting passed.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--wait-for-idle', action='store_true')
    a = p.parse_args()
    if a.check:
        check()
        return
    prepare()
    if a.prepare:
        return
    with (OUT/'suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This transfer suite already has a runner.')
        if (OUT/'completed.json').exists():
            return
        idle(a.wait_for_idle)
        verify()
        report = str(Path(__file__).with_name('paper_transfer_report.py'))
        try:
            smoke = [(d, n) for d in ['plateau_near_600', 'redraw_broad_100']
                     for n in ['fixed_25', 'rule_baseline', 'value_s0']]
            evaluate(smoke, 'smoke', 260923501)
            command([report, '--smoke'], 'smoke_audit')
            evaluate([(d, n) for d in DOMAINS for n in LEARNED+CONTROLS+FIXED], 'validation', 260923502)
            selected = select()
            evaluate([(d, n) for d in DOMAINS for n in LEARNED+CONTROLS+[selected[d]['name']]], 'holdout', 260923503)
            command([report], 'final_audit')
            write(OUT/'status.json', {'phase': 'completed', 'completed_epoch': time.time()})
        except Exception as error:
            write(OUT/'status.json', {'phase': 'failed', 'error': str(error), 'epoch': time.time()})
            raise


if __name__ == '__main__':
    main()
