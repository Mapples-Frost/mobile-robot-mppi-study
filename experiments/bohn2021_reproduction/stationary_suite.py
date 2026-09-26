"""Registered terminal-stationarity comparison; all final seeds are retained."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART, ROOT
import recoverable_runtime as distribution
from optimized_runtime import install_terminal
from run import write, snapshot

OUT = ART / 'results/stationary_terminal'
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
GRID = [1] + list(range(5, 51, 5))


def job(spec):
    name, script, args = spec
    with open(OUT / (name + '.log'), 'a') as log:
        result = subprocess.run([sys.executable, '-u', str(SCRIPTS / script)] + args,
                                stdout=log, stderr=subprocess.STDOUT, timeout=18000)
    assert result.returncode == 0, name
    print(json.dumps({'complete': name}), flush=True)
    return name


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = {
        'question': 'Does eliminating terminal-value drift improve horizon learning with identical author SAC?',
        'stage1': 'Freeze analytic Riccati terminal; omit only its optimizer execution, keep graph and both replay samples.',
        'learned_arms': {'frozen': '3 new models', 'joint': 'Reuse all 3 final prior_refinement models; identical initialization/hyperparameters'},
        'seeds': [0, 1, 2], 'steps_per_model': 15000, 'new_training_transitions': 45000,
        'reward_scale': .6, 'alpha': .01, 'gamma': .97, 'batch_size': 256,
        'time_limit_bootstrap': 'Retain author time_aware=True; no simultaneous task-semantics change.',
        'validation': {'seed': 26091820, 'n': 10}, 'holdout': {'seed': 26091821, 'n': 30},
        'fixed_grid': GRID, 'fixed_selection': 'Lowest mean validation total cost; ties use smaller H; no holdout selection.',
        'fixed_budget': 'Analytic frozen terminal requires zero training transitions; deterministic duplicates are not independent seeds.',
        'joint_fixed_reference': 'Previously selected H30, all 3 prior_refinement final checkpoints; no new H selection.',
        'primary': 'Mean undiscounted total physical cost, with constraint episodes reported separately per training seed.',
        'secondary': ['performance cost', 'H proxy cost', 'mean H', 'solver failures', 'discounted physical cost'],
        'unit': 'Training seed is the learned-policy replicate; scenes are paired repeated evaluations, not extra seeds.',
        'stage2': 'Frozen analytic terminal continuous versus categorical SAC H1..50; register library and matched controls before training.',
        'joint_restore_gate': 'Only consider joint discrete training if frozen discrete has zero validation constraints in every seed and improves mean validation total cost >=2% against both matched continuous and selected fixed H; decision uses validation only.',
        'claims': 'Method extensions, not exact reproduction; H cost is a compute proxy, not measured speedup.'}
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write(path, protocol)
    distribution.OUT = OUT
    install_terminal('pendulum')
    distribution.install_distribution()
    for name, seed, count in [('validation', 26091820, 10), ('holdout', 26091821, 30)]:
        path = OUT / (name + '_bank.json')
        if not path.exists():
            env = distribution.make_env('pendulum', seed)
            np.random.seed(seed)
            cases = []
            for _ in range(count):
                env.reset()
                cases.append(snapshot(env))
            write(path, {'seed': seed, 'cases': cases})
    names = ['stationary_run.py', 'stationary_evaluate.py', 'stationary_suite.py', 'run.py',
             'runtime.py', 'optimized_runtime.py', 'forecast_runtime.py', 'recoverable_runtime.py', 'riccati_terminal_probe.py']
    hashes = {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in names}
    hash_path = OUT / 'source_hashes.json'
    if hash_path.exists():
        assert json.loads(hash_path.read_text()) == hashes, 'Source changed after registration'
    else:
        write(hash_path, hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    prepare()
    if args.prepare_only:
        return
    jobs = []
    for seed in range(3):
        name = 'pendulum_rl_s%d' % seed
        jobs.append((name, 'stationary_run.py', ['--task', 'pendulum', '--seed', str(seed),
            '--steps', '15000', '--out', str(OUT / name), '--aligned', '--scaled-obs', '--batch-size', '256',
            '--buffer-size', '1000000', '--ent-coef', '0.01', '--test-bank', str(OUT / 'validation_bank.json')]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, jobs))
    validation_jobs = [('fixed_h%d_validation' % h, 'stationary_evaluate.py',
        ['--fixed-horizon', str(h), '--bank', str(OUT / 'validation_bank.json'),
         '--out', str(OUT / ('fixed_h%d_validation' % h))]) for h in GRID]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, validation_jobs))
    candidates = [(json.loads((OUT / ('fixed_h%d_validation' % h) / 'summary.json').read_text())['mean_total_cost'], h) for h in GRID]
    cost, selected = min(candidates)
    write(OUT / 'selected_fixed.json', {'h': selected, 'validation_cost': cost, 'candidates': candidates})
    jobs = []
    for group in ['frozen', 'joint', 'joint_fixed']:
        for seed in range(3):
            origin = OUT if group == 'frozen' else ART / 'results/prior_refinement'
            name = 'pendulum_fixed_h30_s%d' % seed if group == 'joint_fixed' else 'pendulum_rl_s%d' % seed
            spec = ['--model-dir', str(origin / name), '--bank', str(OUT / 'holdout_bank.json'),
                    '--out', str(OUT / ('holdout_%s_s%d' % (group, seed)))]
            if group == 'joint_fixed':
                spec += ['--fixed-horizon', '30']
            jobs.append(('holdout_%s_s%d' % (group, seed), 'stationary_evaluate.py', spec))
    jobs.append(('holdout_fixed', 'stationary_evaluate.py', ['--fixed-horizon', str(selected),
        '--bank', str(OUT / 'holdout_bank.json'), '--out', str(OUT / 'holdout_fixed')]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, jobs))
    write(OUT / 'completed.json', {'training_transitions': 45000, 'selected_fixed_h': selected,
                                 'all_final_seeds_evaluated': True})


if __name__ == '__main__':
    main()
