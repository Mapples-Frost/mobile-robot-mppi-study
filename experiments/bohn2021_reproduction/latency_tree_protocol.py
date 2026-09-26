"""Freeze the next learned method before new banks or outcome collection."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/latency_tree_2026-09-26'
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
TASKS = ('vehicle', 'pendulum')
REG = OUT / 'registration.json'
MODERN = str(ROOT / '.venv/bin/python')
LEGACY = '/home/mapples/.local/share/bohn2021-python37/bin/python'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=serial, allow_nan=False, ensure_ascii=False) + '\n')
    temporary.replace(path)


def model_dir(task, seed):
    from min_q_eval_suite import model_dir as inherited
    return inherited(task, 'fixed', seed)


def bank_name(task, split, seed=None):
    assert split in ('smoke', 'fit', 'select', 'validation', 'test')
    assert (seed is not None) == (split in ('fit', 'select'))
    return OUT / 'banks' / ('%s_%s%s_bank.json' % (task, split, '_s%d' % seed if seed is not None else ''))


def bank_specs():
    jobs = []
    for ti, task in enumerate(TASKS):
        for split, base, count in (('fit', 2609261000, 12), ('select', 2609262000, 16)):
            for seed in range(3):
                jobs.append(dict(task=task, split=split, seed=seed, rng=base + ti * 100 + seed, cases=count))
        for split, base, count in (('smoke', 2609260900, 2), ('validation', 2609263000, 64), ('test', 2609264000, 128)):
            jobs.append(dict(task=task, split=split, seed=None, rng=base + ti * 100, cases=count))
    return jobs


def spec():
    return dict(method='Categorical cross-entropy direct closed-loop depth2 horizon tree; extension, not original SAC',
        tasks=list(TASKS), seeds=[0, 1, 2], horizons=list(range(5, 51, 5)), nodes=3, leaves=4,
        generations=4, population=12, elites=3, update_weight=.6, exploration=.1,
        initial_base_action_probability=.7, threshold_quantiles=[.1, .25, .5, .75, .9],
        fit_cases=12, select_cases=16, finalists=4, select_repeats=2, bank_specs=bank_specs(),
        search_seed_base=2609265000, order_seed_base=2609266000, workers=1,
        objective='relative raw total cost +0.5*(measured net decision time ratio-1), lexicographically after safety/NI violations',
        no_pruning=True, no_certificate_teacher=True, no_validation_tuning=True, no_test_access=True,
        inherited_terminal_steps_each=15000, fixed_grid=list(range(5,51,5)),
        formal_gate='Both tasks x both nominated fixed comparators x all3 seeds. Same unchanged safety, physicalNI2%, within-episode adaptation, >=3% total-cost gain with paired95% upper delta<0 OR >=10% measured timing gain with paired95% upper ratio<1, both repeats favorable,totalNI2%. Independently sealed confirmation repeats all criteria.',
        reference_reset='Primary-terminal canonical H50 warmup for all same-seed arms, own terminal restored before scored actions',
        training_budget_max_episodes=4920, training_budget_max_steps=738000, bank_generation_resets=556,
        main_claim='Learned tree alone must pass. Hand-coded C5 and tree+C5 are separately reported diagnostic comparators.')


def register():
    if REG.exists():
        verify(); print('Existing latency-tree registration verified'); return
    assert not OUT.exists(), 'Existing partial unregistered output requires inspection'
    files = set(SCRIPTS.glob('*.py'))
    files.update((ART / 'sources').glob('*/**/*.py'))
    files.add(ROOT / 'docs/protocols/bohn2021_latency_tree_2026-09-26.md')
    for name in ('latency_tree_protocol.py', 'latency_tree_policy.py', 'latency_tree_run.py',
                 'latency_tree_audit.py', 'latency_tree_checks.py', 'latency_tree_learning_audit.py'):
        assert (SCRIPTS / name).is_file(), ('Required source missing', name)
    for task in TASKS:
        files.add(ART / 'configs' / (task + '.json'))
        for seed in range(3):
            folder = model_dir(task, seed)
            manifest, done = read(folder / 'manifest.json'), read(folder / 'completed.json')
            assert done['status'] == 'complete' and manifest['steps'] == done['steps'] == 15000
            assert manifest['seed'] == seed and manifest['fixed_horizon'] == (25 if task == 'vehicle' else 30)
            files.update(folder / n for n in ('model.zip', 'manifest.json', 'completed.json'))
    previous = ART / 'results/failure_state_probe_2026-09-26'
    assert read(previous / 'package_check.json')['passed']
    files.update(previous / n for n in ('package_manifest.json', 'package_check.json', 'timing_delivery/report.json'))
    OUT.mkdir()
    write(REG, dict(registered_utc=datetime.now(timezone.utc).isoformat(), spec=spec(),
                    hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}, test_accessed=False))
    print(json.dumps(dict(registered=True, frozen_files=len(files), banks=len(bank_specs()), cases=sum(x['cases'] for x in bank_specs())), indent=2))


def verify():
    saved = read(REG)
    assert saved['spec'] == spec()
    for name, digest in saved['hashes'].items():
        assert sha(ROOT / name) == digest, ('Frozen source/input changed', name)
    old = ART / 'results/gated_horizon_search_2026-09-25'
    assert not (old / 'confirmation_registration.json').exists() and not (old / 'evaluations/test').exists()
    return saved


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--register', action='store_true')
    if parser.parse_args().register: register()
    else: verify(); print('Latency-tree frozen inputs verified')
