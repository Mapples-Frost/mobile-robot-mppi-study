"""Sequential teacher-data stages, with at most two active MPC solvers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
from pathlib import Path
import numpy as np
from teacher_common import OUT, ROOT, SEEDS, prepare, read, write, sha, verify

LEGACY = '/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN = str(ROOT/'.venv/bin/python')
WORKER = str(Path(__file__).with_name('teacher_worker.py'))


def launch(split, arms, label=False, count=None):
    count = count or len(read(OUT/(split+'_bank.json'))['scenes'])
    jobs = [(arm, i) for arm in arms for i in range(count)]
    np.random.RandomState(26091845).shuffle(jobs)
    def run(job):
        arm, i = job
        name = '%s_%s_%s_%02d'%('label' if label else 'eval', split, arm, i)
        log = OUT/'logs'/(name+'.log')
        log.parent.mkdir(exist_ok=True)
        cmd = [LEGACY, '-u', WORKER, '--split', split, '--scene', str(i), '--arm', arm]
        if label:
            cmd.append('--label')
        with log.open('a') as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, timeout=7200)
        print('%s exit=%d'%(name, result.returncode), flush=True)
        assert result.returncode == 0, name
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, jobs))


def fixed_best():
    rows = []
    for folder in (OUT/'evaluation/validation').glob('fixed_*'):
        results = [read(p) for p in sorted(folder.glob('scene_*/completed.json'))]
        assert len(results) == 4
        rows.append({'arm': folder.name, 'cost': float(np.mean([v['adjusted_cost'] for v in results]))})
    return min(rows, key=lambda r: (r['cost'], r['arm'])), rows


def train(round_id):
    log = OUT/'logs'/('train_round%d.log'%round_id)
    with log.open('a') as handle:
        result = subprocess.run([MODERN, '-u', str(Path(__file__).with_name('teacher_train.py')),
            '--round', str(round_id)], stdout=handle, stderr=subprocess.STDOUT, timeout=1800)
    assert result.returncode == 0, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    args = ap.parse_args()
    prepare()
    if args.prepare:
        return
    launch('train', ['switch_5_30'], label=True)
    launch('validation', ['switch_5_30'], label=True)
    train(0)
    launch('aggregation', ['round0_s0'], label=True)
    train(1)
    student_arms = ['round%d_s%d'%(r, s) for r in [0, 1] for s in SEEDS]
    coarse = [1]+list(range(5, 51, 5))
    launch('validation', ['fixed_%d'%h for h in coarse]+student_arms+['switch_5_30'])
    best, rows = fixed_best()
    center = int(best['arm'].split('_')[1])
    additional = [h for h in range(max(1, center-4), min(50, center+4)+1) if h not in coarse]
    write(OUT/'refinement.json', {'coarse_best': best, 'additional': additional})
    launch('validation', ['fixed_%d'%h for h in additional])
    best, rows = fixed_best()
    write(OUT/'selection.json', {'fixed': best['arm'], 'fixed_validation': rows,
        'students': student_arms, 'student_selection': 'None; all seeds and rounds retained',
        'model_hashes': {arm: sha(OUT/'models'/(arm+'.json')) for arm in student_arms}})
    launch('holdout', sorted(set(student_arms+['switch_5_30', best['arm'], 'fixed_30'])))
    verify()
    write(OUT/'completed.json', {'status': 'complete', 'SAC_updates': 0,
        'supervised_models': 6, 'supervised_updates_per_model': 3000, 'fixed': best['arm']})


if __name__ == '__main__':
    main()
