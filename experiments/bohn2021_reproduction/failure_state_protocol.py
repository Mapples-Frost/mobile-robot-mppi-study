"""Machine-frozen post-hoc diagnosis on every already exposed training case."""
import argparse
import ast
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from failure_state_policy import policies, check

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OLD = ART / 'results/gated_horizon_search_2026-09-25'
OUT = ART / 'results/failure_state_probe_2026-09-26'
REG = OUT / 'registration.json'
TASKS = ('pendulum', 'vehicle')
LEGACY = '/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN = str(ROOT / '.venv/bin/python')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    def serial(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=serial, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def rel(p):
    return p.resolve().relative_to(ROOT).as_posix()


def source_model(task, seed):
    if seed:
        return ART / 'results/min_q_training_2026-09-24' / ('%s_fixed_h%d_s%d' % (task, 30 if task == 'pendulum' else 25, seed))
    return ART / ('results/paper_defaults/pendulum_fixed_h30' if task == 'pendulum' else 'results/paper_exact_grid_2026-09-23/vehicle_fixed_h25')


def bank(task, seed):
    return OLD / 'banks' / ('%s_train_s%d_bank.json' % (task, seed))


def old_folder(task, seed, policy):
    if policy == ('fixed30' if task == 'pendulum' else 'fixed25'):
        return OLD / 'train' / ('%s_s%d' % (task, seed)) / 'fixed'
    if task == 'vehicle' and policy == 'selected':
        chosen = read(OLD / 'train' / ('vehicle_s%d' % seed) / 'policy.json')
        return OLD / 'train' / ('vehicle_s%d' % seed) / chosen['id']
    return None


def jobs():
    return [(t, s, p) for t in TASKS for s in range(3) for p in policies(t)]


def sources():
    # Freeze local transitive imports so a helper change cannot silently alter runs.
    queue = [SCRIPTS / n for n in ('failure_state_protocol.py', 'failure_state_policy.py',
              'failure_state_run.py', 'failure_state_audit.py', 'failure_state_report.py', 'failure_state_pipeline.py')]
    found = set()
    while queue:
        path = queue.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                local = SCRIPTS / (name.split('.')[0] + '.py')
                if local.exists():
                    queue.append(local)
    for package in ('gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon'):
        found.update((ART / 'sources' / package).rglob('*.py'))
    found.add(ROOT / 'docs/protocols/bohn2021_failure_state_probe_2026-09-26.md')
    for task in TASKS:
        found.add(ART / 'configs' / (task + '.json'))
        for seed in range(3):
            found.add(bank(task, seed))
            found.update(source_model(task, seed) / n for n in ('model.zip', 'manifest.json', 'completed.json'))
            found.add(OLD / 'train' / ('%s_s%d' % (task, seed)) / 'policy.json')
            for policy in policies(task):
                folder = old_folder(task, seed, policy)
                if folder:
                    found.add(folder / 'completed.json')
    found.update(OLD / n for n in ('audit_train.json', 'training_angular_diagnosis/report.json',
                 'training_angular_diagnosis/all_cases.csv', 'validation_finish_status.json',
                 'effect_gate_review/report.json', 'opportunity_diagnosis/report.json'))
    return {rel(p): sha(p) for p in sorted(found)}


def register():
    assert check()['passed']
    finish = read(OLD / 'validation_finish_status.json')
    assert finish['complete'] and not finish['active'] and not finish['test_opened']
    assert not read(OLD / 'effect_gate_review/report.json')['validation_effect_passed']
    assert not (OLD / 'evaluations/test').exists()
    angular = read(OLD / 'training_angular_diagnosis/report.json')
    assert angular['passed'] and angular['training_only']
    with (OLD / 'training_angular_diagnosis/all_cases.csv').open() as f:
        rows = list(csv.DictReader(f))
    smoke = {t: {} for t in TASKS}
    for seed in range(3):
        first = min(int(r['case']) for r in rows if int(r['seed']) == seed and r['certified_at_scored_start'] == 'True')
        smoke['pendulum'][str(seed)] = sorted(set((0, first)))
        smoke['vehicle'][str(seed)] = [0]
        for task in TASKS:
            assert len(read(bank(task, seed))['cases']) == 24
            m = read(source_model(task, seed) / 'manifest.json')
            d = read(source_model(task, seed) / 'completed.json')
            assert m['seed'] == seed and d['status'] == 'complete' and d['steps'] == m['steps'] == 15000
            assert m['fixed_horizon'] == (30 if task == 'pendulum' else 25)
    timing_jobs=[]
    for repeat in range(2):
        ordered=[list(j) for j in jobs() if j[0]=='pendulum']
        random.Random(2609265100+repeat).shuffle(ordered)
        timing_jobs.extend([dict(repeat=repeat,job=j) for j in ordered])
    value = dict(post_hoc=True, diagnostic_only=True, learned_policy=False,
                 source_hashes=sources(), jobs=[list(j) for j in jobs()], smoke_cases=smoke,
                 smoke='Every policy, all3 seeds: original case0 plus first already documented start-certified pendulum case. Two exact repeats. Additional certified case exercises active branch, not efficacy selection.',
                 full='All24 original training cases per seed, all4 policies per task. Reuse audited original fixed reference and vehicle selected policy; run every remaining condition. No pruning or new training.',
                 policies={t:list(policies(t)) for t in TASKS},
                 state='Current plant theta/omega only for angular trigger, no oracle or future state. Original selected vehicle gate unchanged.',
                 shared='Same seed terminal, canonical reset, masked50 NLP, bounded original retries, dynamics/cost/limits. No early termination or failed-scene exclusion.',
                 timing='Optional next stage only after full audit. All4 pendulum policies,3 seeds,24 scenes,2 serial randomized repeats; gross/net logging,reset and retries retained. Order frozen here; timing code must be separately frozen before measuring.',
                 timing_order_seed=2609265100,timing_jobs=timing_jobs,
                 timing_screen='At least one certificate candidate must satisfy aggregate success/constraint/initial and final failure rates and physical-cost NI2% for all3 seeds and actually intervene before scheduling timing. Otherwise stop this mechanism as failed without speed claim. Not the final reproduction gate.',
                 budgets=dict(smoke_episodes=72, smoke_control_step_upper_bound=8400,
                              full_comparison_episodes=576, reused_episodes=216, new_full_episodes=360,
                              new_full_control_step_upper_bound=43200, pendulum_timing_episodes=576,
                              pendulum_timing_step_upper_bound=57600),
                 isolation='No new validation/test outcomes; original validation/test gates unchanged. Any adopted new method requires fresh train/validation/sealed-test.',
                 audit='Independent formula decisions, full costs/observations/context/termination, DOP853 or analytic dynamics at1e-7, retry/counter reconciliation and exact replay. Partial attempts persist and cannot silently resume.',
                 new_model_training_steps=0, test_access=False)
    if REG.exists():
        old = read(REG)
        assert {k:v for k,v in old.items() if k not in ('registered_utc','source_hashes')} == {k:v for k,v in value.items() if k!='source_hashes'}
        verify()
    else:
        OUT.mkdir(exist_ok=False)
        write(REG, dict(registered_utc=datetime.now(timezone.utc).isoformat(), **value))
    print(json.dumps(dict(registered=True, inputs=len(value['source_hashes']), jobs=len(jobs()), smoke_cases=smoke, policy_checks=check()), indent=2))


def verify():
    spec = read(REG)
    amendment = OUT / 'serialization_repair/amendment.json'
    replacements = {}
    if amendment.exists():
        amended = read(amendment)
        assert amended['original_registration_sha256'] == sha(REG)
        receipt = OUT / 'serialization_repair/failure_receipt.json'
        assert sha(receipt) == amended['failure_receipt_sha256']
        for name,value in read(receipt)['archived_files'].items():
            assert sha(ROOT/name)==value
        for name,item in amended['changed_sources'].items():
            assert spec['source_hashes'][name] == item['old_sha256']
            assert sha(ROOT/item['archive']) == item['old_sha256']
            replacements[name]=item['new_sha256']
    for name, value in spec['source_hashes'].items():
        assert sha(ROOT / name) == replacements.get(name,value), ('Frozen source changed', name)
    assert not (OLD / 'evaluations/test').exists()
    return spec


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:register()
    else:verify();print('Frozen sources verified')
