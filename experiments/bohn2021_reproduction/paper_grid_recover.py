"""Preserve interrupted attempts and finish the registered paper fixed-H grid.

Run under the pinned legacy Python; --prepare only inventories/freezes inputs.
No original experiment implementation or already completed model is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/paper_exact_grid_2026-09-23'
REC = OUT / 'recovery_2026-09-24'
OLD = ART / 'results/paper_defaults'
SCRIPTS = Path(__file__).parent
JOBS = [(task, h) for task, existing in [('vehicle', {5, 10, 15}),
        ('pendulum', {20, 30, 40})] for h in range(5, 51, 5) if h not in existing]
MUTEX = threading.Lock()


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def name(job):
    return '%s_fixed_h%d' % job


def no_competing_run():
    forbidden = {'kh_ladder_sweep.py', 'kh_ladder_roll.py', 'kh_backfill.sh',
                 'paper_exact_grid_suite.py', 'paper_exact_grid_evaluate.py',
                 'paper_transfer_suite.py', 'sac_preserve_suite.py'}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            tokens = {Path(a).name for a in p.read_bytes().decode().split('\0') if a}
        except (OSError, UnicodeError):
            continue
        if tokens & forbidden:
            raise RuntimeError('Conflicting process: ' + str(p))


def verify():
    for path, digest in read(REC / 'frozen_inputs.json').items():
        assert sha(ROOT / path) == digest, path


def prepare():
    no_competing_run()
    REC.mkdir(parents=True, exist_ok=True)
    if (REC / 'frozen_inputs.json').exists():
        verify()
        return
    # Check core runtime against the saved pre-continuation audit.
    historical = read(OLD / 'audit.json')['script_sha256']
    for leaf in ['run.py', 'runtime.py', 'evaluate_saved.py']:
        p = SCRIPTS / leaf
        assert sha(p) == historical[str(p.relative_to(ROOT))], leaf
    inventory = []
    for job in JOBS:
        p = OUT / name(job)
        progress = read(p / 'progress.json') if (p / 'progress.json').exists() else {}
        state = 'complete' if (p / 'completed.json').exists() else 'interrupted' if p.exists() else 'not_started'
        inventory.append({'name': p.name, 'state': state,
                          'saved_progress': progress,
                          'files': {str(f.relative_to(OUT)): sha(f) for f in p.rglob('*') if f.is_file()}})
    write(REC / 'inventory.json', inventory)
    paths = [SCRIPTS / n for n in ['run.py', 'runtime.py', 'evaluate_saved.py',
                                   'paper_grid_recover.py', 'paper_exact_grid_suite.py']]
    paths += list((ART / 'configs').glob('*.json'))
    for repo in ['gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon']:
        paths += list((ART / 'sources' / repo).rglob('*.py'))
    paths += [OUT / 'protocol.json', OLD / 'protocol.json']
    for p in OLD.iterdir():
        if p.is_dir() and (p / 'completed.json').exists():
            paths += [p / n for n in ['model.zip', 'manifest.json', 'completed.json']]
    for entry in inventory:
        if entry['state'] == 'complete':
            paths += [OUT / entry['name'] / n for n in ['model.zip', 'manifest.json', 'completed.json']]
    write(REC / 'frozen_inputs.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    write(REC / 'protocol.json', {
        'scope': 'Finish all 14 registered fixed-H continuation runs, then both frozen holdout modes; no new algorithm or tuning.',
        'workers': 2, 'training_steps_per_model': 15000,
        'recovery': 'Archive every incomplete directory and its log; restart original seed from step 0. Full models without completion markers are also retained in archive, not silently certified.',
        'unchanged': 'Original run.py arguments, reward, bank, 32-step joint terminal learning, all H and seeds retained.',
        'evaluation': 'All 26 original+continuation models, both terminal modes, 20 shared cases per task; 1040 episode conditions total, 560 newly needed.',
        'selection': 'Validation mean total cost selects fixed H, tie by smaller H. Also show test-minimum grid point as a descriptive retrospective envelope, not an independent selected comparator.',
        'limits': 'Historical holdout already exposed; this completes an existing comparison, not fresh confirmatory evidence. Fixed-H seed0 only. Three RL seeds and 20 scenes per task are not 1040 independent samples.',
        'accounting': 'Retained 390000 training steps for all 26 models. Add archived progress as interrupted observed lower bound; unflushed steps/reset overhead unknown. Recovery reruns not claimed exact mid-run resume.',
        'time_epoch': time.time()})


def archive_partial():
    for entry in read(REC / 'inventory.json'):
        if entry['state'] != 'interrupted':
            continue
        original = OUT / entry['name']
        target = REC / 'interrupted' / entry['name']
        if target.exists():
            continue
        for rel, digest in entry['files'].items():
            assert sha(OUT / rel) == digest, rel
        assert original.resolve().parent == OUT.resolve()
        assert OUT.resolve() in target.resolve().parents
        target.parent.mkdir(parents=True, exist_ok=True)
        original.rename(target)
        log = OUT / (entry['name'] + '.log')
        if log.exists():
            log.rename(target.parent / log.name)
    write(REC / 'archive_complete.json', {'complete': True, 'epoch': time.time()})


def execute(args, label):
    no_competing_run()
    verify()
    log = REC / 'logs' / (label + '.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:
        result = subprocess.run([sys.executable, '-u'] + args, cwd=str(ROOT),
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('%s exited %s; %s' % (label, result.returncode, log))
    with MUTEX:
        state = read(REC / 'status.json')
        state.setdefault('finished_jobs', []).append(label)
        state['updated_epoch'] = time.time()
        write(REC / 'status.json', state)
    print('complete', label, flush=True)


def train(job):
    task, h = job
    folder = OUT / name(job)
    if (folder / 'completed.json').exists():
        return
    if (folder / 'manifest.json').exists():
        raise RuntimeError('New interrupted attempt requires a new recovery ledger: ' + str(folder))
    execute([str(SCRIPTS / 'run.py'), '--task', task, '--seed', '0', '--steps', '15000',
             '--out', str(folder), '--aligned', '--scaled-obs', '--batch-size', '256',
             '--buffer-size', '1000000', '--ent-coef', '1.0', '--fixed-horizon', str(h),
             '--test-bank', str(ART / 'configs' / (task + '_validation_bank.json'))], name(job))


def evaluate(job):
    folder, use_value = job
    mode = 'holdout_value' if use_value else 'holdout_no_value'
    if (folder / mode / 'completed.json').exists():
        return
    task = read(folder / 'manifest.json')['task']
    args = [str(SCRIPTS / 'evaluate_saved.py'), '--model-dir', str(folder),
            '--bank', str(ART / 'configs' / (task + '_holdout_bank.json')),
            '--out', str(folder / mode)]
    if not use_value:
        args.append('--no-value')
    execute(args, folder.name + '_' + mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare:
        print(json.dumps(read(REC / 'protocol.json'), indent=2))
        return
    with (REC / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REC / 'completed.json').exists():
            return
        try:
            archive_partial()
            write(REC / 'status.json', {'phase': 'training', 'pid': os.getpid(), 'finished_jobs': [], 'started_epoch': time.time()})
            jobs = list(JOBS)
            random.Random(260924501).shuffle(jobs)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(train, jobs))
            assert all((OUT / name(j) / 'completed.json').exists() for j in JOBS)
            write(OUT / 'training_completed.json', {'complete': True, 'models': 14, 'recovery': str(REC.relative_to(ROOT))})
            write(REC / 'status.json', {'phase': 'holdout', 'pid': os.getpid(), 'finished_jobs': [], 'started_epoch': time.time()})
            folders = [OUT / name(j) for j in JOBS]
            folders += [p for p in OLD.iterdir() if p.is_dir() and (p / 'completed.json').exists()]
            assert len(folders) == 26
            jobs = [(p, v) for p in folders for v in [True, False]]
            random.Random(260924502).shuffle(jobs)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(evaluate, jobs))
            verify()
            write(OUT / 'holdout_completed.json', {'complete': True, 'all_models': 26, 'episode_conditions': 1040})
            write(REC / 'completed.json', {'training_and_evaluation': True, 'audit_pending': True})
            write(REC / 'status.json', {'phase': 'ready_for_audit', 'epoch': time.time()})
        except BaseException as error:
            write(REC / 'status.json', {'phase': 'failed', 'error': repr(error), 'epoch': time.time()})
            raise


if __name__ == '__main__':
    main()
