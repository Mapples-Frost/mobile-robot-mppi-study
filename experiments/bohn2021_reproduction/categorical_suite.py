"""Register and execute matched modern SAC arms without tuning on holdout."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from categorical_runtime import DEPS, np, torch
from runtime import ART, ROOT
from run import write

OUT = ART / 'results/categorical_frozen'
BASE = ART / 'results/stationary_terminal'
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'


def job(spec):
    mode, seed, evaluation = spec
    name = '%s_s%d' % (mode, seed)
    args = [sys.executable, '-u', str(SCRIPTS / 'categorical_run.py'), '--mode', mode,
            '--seed', str(seed), '--steps', '15000', '--out', str(OUT / name), '--bank',
            str(BASE / ('holdout_bank.json' if evaluation else 'validation_bank.json'))]
    if evaluation:
        args.append('--eval-only')
    with open(OUT / (name + ('_holdout' if evaluation else '') + '.log'), 'a') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=18000)
    assert result.returncode == 0, (name, evaluation)
    print(json.dumps({'complete': name, 'holdout': evaluation}), flush=True)


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = {
        'question': 'Does categorical SAC improve frozen-terminal horizon allocation relative to same-library continuous SAC?',
        'library': 'Tianshou 0.5.1 PyTorch CPU; SACPolicy and DiscreteSACPolicy; library core unmodified',
        'seeds': [0, 1, 2], 'steps_per_model': 15000, 'training_runs': 6, 'new_training_transitions': 90000,
        'common': {'actor_layers': [32, 32], 'critic_layers': [256, 256], 'gamma': .97, 'alpha': .01,
            'tau': .005, 'adam_lr': .0003, 'batch_size': 256, 'buffer_size': 1000000, 'updates': 14745,
            'warmup': '100 uniform integer H samples shared across arms per seed',
            'reward': 'Unchanged physical reward, divided by .6 as in author pendulum SAC',
            'observation': '56-dimensional full-reference preview', 'terminal': 'Frozen float32 Riccati prior',
            'first_transition': 'Zero terminal on first training action, then frozen prior; matches legacy callback timing',
            'time_limit': 'Bootstrap at steps termination, stop at physical constraint',
            'MPC': 'Same Python3.7 author plant/solver worker, zero-terminal H50 reset warmup'},
        'arms': {'continuous': 'tanh Gaussian, round affine H in1..50; scalar action-conditioned Q',
                 'discrete': '50 categorical logits, exact action expectation,50-output Q'},
        'entropy_caveat': 'Same numeric alpha does not equate differential and categorical entropy. This compares standard algorithm packages; it does not isolate rounding alone. No temperature selection using holdout.',
        'legacy_caveat': 'Modern twin-Q targets/minQ actor differ from author V-target/Q1 actor; use modern continuous as primary discrete comparator.',
        'evaluation': {'validation_bank': str(BASE / 'validation_bank.json'), 'holdout_bank': str(BASE / 'holdout_bank.json'),
                       'final_only': True, 'deterministic': True, 'all_seeds': True},
        'fixed_reference': 'Frozen analytic terminal H selected on stationary validation grid; zero learning budget',
        'gate': 'Before opening modern holdout: all discrete seeds zero validation constraints AND mean discrete validation cost <=.98*mean modern continuous AND <=.98*selected fixed validation cost.',
        'gate_action': 'Passing permits separately registered joint-discrete study; failing keeps terminal frozen and reports failure without tuning.',
        'unit': '3 training seeds, paired repeated scenarios; no significance claim from episode pseudoreplication',
        'fidelity': 'Explicit method extension, not exact paper reproduction; H cost is compute proxy only'}
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write(path, protocol)
    names = ['categorical_run.py', 'categorical_runtime.py', 'categorical_suite.py',
             'categorical_verify.py', 'stationary_worker.py']
    hashes = {n: hashlib.sha256((SCRIPTS / n).read_bytes()).hexdigest() for n in names}
    for name in ['sac.py', 'discrete_sac.py']:
        path = DEPS / 'tianshou/policy/modelfree' / name
        hashes['tianshou/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = OUT / 'source_hashes.json'
    if path.exists():
        assert json.loads(path.read_text()) == hashes
    else:
        write(path, hashes)
    write(OUT / 'dependencies.json', {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'isolated_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(DEPS)])}})
    assert json.loads((BASE / 'categorical_verification.json').read_text())['discrete_soft_bellman_exact']
    for mode in ['continuous', 'discrete']:
        assert json.loads((BASE / ('smoke_' + mode) / 'completed.json').read_text())['save_load_exact']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--holdout-only', action='store_true')
    args = ap.parse_args()
    prepare()
    if args.prepare_only:
        return
    jobs = [(mode, seed, False) for seed in range(3) for mode in ['continuous', 'discrete']]
    if not args.holdout_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(job, jobs))
    selected = json.loads((BASE / 'selected_fixed.json').read_text())
    data = {mode: [json.loads((OUT / ('%s_s%d' % (mode, seed)) / 'eval_validation_bank/summary.json').read_text())
                   for seed in range(3)] for mode in ['continuous', 'discrete']}
    means = {mode: float(np.mean([r['mean_total_cost'] for r in group])) for mode, group in data.items()}
    passed = all(r['constraint_episodes'] == 0 for r in data['discrete']) and means['discrete'] <= .98 * means['continuous'] and means['discrete'] <= .98 * selected['validation_cost']
    write(OUT / 'validation_gate.json', {'passed': bool(passed), 'means': means, 'selected_fixed': selected,
        'discrete_constraint_episodes': [r['constraint_episodes'] for r in data['discrete']],
        'decision': 'Eligible for separately registered joint training' if passed else 'Do not restore joint learning'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, [(m, s, True) for m, s, _ in jobs]))
    write(OUT / 'completed.json', {'new_training_transitions': 90000, 'all_final_seeds_evaluated': True,
                                 'joint_restore_gate_passed': bool(passed)})


if __name__ == '__main__':
    main()
