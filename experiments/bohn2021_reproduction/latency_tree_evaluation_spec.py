"""Immutable execution contract for the latency-tree formal evaluation.

This module adds evaluation machinery; it never modifies the training registration.
Registration is intentionally refused until all listed implementation checks exist.
"""
import argparse
import json
import math
from pathlib import Path

from latency_tree_protocol import (
    ROOT, ART, OUT, REG, SCRIPTS, TASKS, read, write, sha, verify,
    model_dir, bank_name,
)

HS = tuple(range(5, 51, 5))
BASE = {'vehicle': 25, 'pendulum': 30}
SETTINGS = dict(aligned=True, scaled_obs=True, ent_coef='1.0',
                no_online_value=False, batch_size=256, buffer_size=1000000)
EVAL_REG = OUT / 'evaluation_registration.json'
REQUIRED = (
    'latency_tree_evaluation_spec.py', 'latency_tree_evaluate.py',
    'latency_tree_baselines.py', 'latency_tree_fixed_train.py',
    'latency_tree_evaluation_audit.py', 'latency_tree_effect.py',
    'latency_tree_effect_review.py', 'latency_tree_evaluation_checks.py',
    'latency_tree_training_delivery.py', 'latency_tree_training_delivery_review.py',
)


def verify_hashes(mapping):
    for name, digest in mapping.items():
        assert sha(Path(name)) == digest, ('Changed frozen evaluation input', name)


def frozen_write(path, value):
    """Never replace an existing registration, nomination, audit, or result."""
    if path.exists():
        assert read(path) == json.loads(json.dumps(value)), ('Immutable record differs', str(path))
    else:
        write(path, value)


def normalize(task, family, seed, h):
    assert task in TASKS and seed in (0, 1, 2)
    if family in ('grid', 'matched') and h == BASE[task]:
        family = 'primary'
    if family in ('adaptive', 'certificate', 'combined'):
        assert h == 0 and (family == 'adaptive' or task == 'pendulum')
    else:
        assert family in ('primary', 'matched', 'grid') and h in HS
        assert family != 'primary' or h == BASE[task]
    return task, family, seed, h


def arm_name(family, seed, h):
    return '%s_h%d_s%d' % (family, h, seed)


def initial_jobs():
    jobs = []
    for task in TASKS:
        for seed in range(3):
            jobs += [(task, 'adaptive', seed, 0), (task, 'primary', seed, BASE[task])]
            jobs += [(task, 'matched', seed, h) for h in HS if h != BASE[task]]
            if task == 'pendulum':
                jobs += [(task, 'certificate', seed, 0), (task, 'combined', seed, 0)]
        jobs += [(task, 'grid', 0, h) for h in HS if h != BASE[task]]
    assert len(jobs) == len(set(jobs)) == 90
    return jobs


def smoke_jobs():
    # Both tasks/all seeds plus distinct independent terminals and both C5 arms.
    jobs = [(t, f, s, BASE[t] if f == 'primary' else 0)
            for t in TASKS for s in range(3) for f in ('primary', 'adaptive')]
    jobs += [('vehicle', 'grid', 0, 5), ('pendulum', 'grid', 0, 35),
             ('pendulum', 'certificate', 0, 0), ('pendulum', 'combined', 0, 0)]
    return jobs


def model_key(task, h, seed):
    return '%s_h%d_s%d' % (task, h, seed)


def check_model(folder, task, h, seed):
    manifest, done = read(folder / 'manifest.json'), read(folder / 'completed.json')
    assert manifest['adaptations'] == SETTINGS, str(folder)
    assert manifest['steps'] == done['steps'] == 15000
    assert manifest['fixed_horizon'] == h and manifest['seed'] == seed
    assert manifest['task'] == task and done['status'] == 'complete'
    assert isinstance(done['final_hash'], str) and (folder / 'model.zip').is_file()
    return done


def existing_model(task, h, seed):
    if h == BASE[task]:
        result = model_dir(task, seed)
    elif seed == 0:
        name = '%s_fixed_h%d' % (task, h)
        candidates = [ART / 'results' / group / name
                      for group in ('paper_defaults', 'paper_exact_grid_2026-09-23')]
        found = [p for p in candidates if (p / 'completed.json').exists()]
        assert len(found) == 1, (task, h, found)
        result = found[0]
    else:
        candidates = [ART / 'results' / group / 'extra_fixed' / model_key(task, h, seed)
                      for group in ('gated_horizon_search_2026-09-25',
                                    'conservative_iteration_2026-09-24')]
        found = [p for p in candidates if (p / 'completed.json').exists()]
        if not found:
            assert not any(p.exists() for p in candidates), ('Inspect inherited partial model', candidates)
            return None
        # Deterministic provenance priority, never selection by evaluation score.
        for p in found:
            check_model(p, task, h, seed)
        assert len({sha(p / 'model.zip') for p in found}) == 1, ('Ambiguous inherited terminals', found)
        result = found[0]
    check_model(result, task, h, seed)
    return result


def specification():
    return dict(
        validation_cases=64, test_cases=128, smoke_cases=2, seeds=[0, 1, 2],
        horizons=list(HS), initial_jobs=initial_jobs(), smoke_jobs=smoke_jobs(),
        workers=1, timing_repeats=2, timing_order_seed=2609268000,
        cost_bootstrap_seed=2609268100, timing_bootstrap_seed=2609268101,
        bootstrap_draws=10000,
        fixed_selection='Full independent seed0 grid; full matched-terminal grid/all3 seeds. '
                        'No aggregate safety decrease versus primary; exact integer failure-rate comparison. '
                        'Lowest mean raw total cost, exact ties smaller H.',
        common_route='Each task must use one route across both fixed comparator labels and all3 seeds.',
        timing='Serial two-repeat exact replay. Feature/context extraction needed for policy plus '
               'controller.get_action, all retries and checks; measured recovery logging deducted. '
               'Reset, simulator, audit-only context, and outer trace writes excluded and disclosed.',
        baseline_training_steps=15000, baseline_final_diagnostic_episodes_per_value_setting=64,
        confirm='Require independent positive validation effect review before any test rollout. '
                'No policy retuning or favorable scenario selection.',
        diagnostic='C5 and learned+C5 are separately retained; only adaptive tree can satisfy main gate.',
        limitations='Intervals resample paired scenes conditional on fitted models; seeds and repeats '
                    'are not independent scene replications. WSL scheduling remains a timing limitation.',
    )


def register():
    verify()
    if EVAL_REG.exists():
        return verify_evaluation()
    assert not (OUT / 'evaluations').exists() and not (OUT / 'evaluation_smoke').exists()
    paths = [SCRIPTS / n for n in REQUIRED]
    assert all(p.is_file() for p in paths), 'Complete evaluation implementation before freezing'
    checks_path = OUT / 'evaluation_checks.json'
    checks = read(checks_path)
    assert checks['passed'] and checks['simulations'] == 0 and not checks['test_accessed']
    verify_hashes(checks['hashes'])
    assert all(str(p) in checks['hashes'] for p in paths)
    delivery_checks_path = OUT / 'training_delivery_checks.json'
    delivery_checks = read(delivery_checks_path)
    assert delivery_checks['passed'] and delivery_checks['simulations'] == 0
    assert not delivery_checks['test_accessed']
    verify_hashes(delivery_checks['hashes'])
    original = (SCRIPTS / 'conservative_fixed_train.py').read_text()
    expected = original.replace(
        'from conservative_iteration import OUT,verify',
        'from latency_tree_evaluation_spec import OUT,verify_evaluation as verify'
    ).replace("(task+'_validation.json')", "(task+'_validation_bank.json')")
    assert (SCRIPTS / 'latency_tree_fixed_train.py').read_text() == expected
    paths += [REG, checks_path, delivery_checks_path, OUT / 'banks/completed.json', OUT / 'split_audit.json']
    inventory = {}
    for task in TASKS:
        for h in HS:
            for seed in range(3):
                source = existing_model(task, h, seed)
                key = model_key(task, h, seed)
                inventory[key] = str(source) if source is not None else None
                if source is not None:
                    paths += [source / n for n in ('model.zip', 'manifest.json', 'completed.json')]
    # Previously audited instrumentation is inherited byte for byte except routing.
    for task in TASKS:
        receipt = ART / 'results/conservative_iteration_2026-09-24/extra_fixed_smoke' / (
            '%s_h%d_s0' % (task, BASE[task])) / 'instrumentation_completed.json'
        saved = read(receipt)
        assert saved['passed'] and saved['smoke']
        verify_hashes(saved['hashes'])
        paths.append(receipt)
    data = dict(spec=specification(), model_inventory=inventory,
                hashes={str(p): sha(p) for p in sorted(set(paths))},
                test_accessed=False)
    frozen_write(EVAL_REG, data)
    return data


def verify_evaluation():
    verify()
    saved = read(EVAL_REG)
    assert saved['spec'] == json.loads(json.dumps(specification()))
    verify_hashes(saved['hashes'])
    return saved


def fixed_model(task, h, seed):
    saved = read(EVAL_REG)
    name = saved['model_inventory'][model_key(task, h, seed)]
    return Path(name) if name is not None else OUT / 'extra_fixed' / model_key(task, h, seed)


def source_for(task, family, seed, h):
    return fixed_model(task, h, seed) if family == 'grid' else model_dir(task, seed)


def freeze_policies():
    verify_evaluation()
    status = read(OUT / 'status.json')
    assert status['complete'] and not status['active'], 'Training must finish before evaluating'
    cmdline = Path('/proc') / str(status['pid']) / 'cmdline'
    if cmdline.exists():
        assert b'latency_tree_run.py' not in cmdline.read_bytes(), 'Training process has not exited'
    paths = [EVAL_REG, OUT / 'status.json']
    for name in ('audit_train.json', 'learning_audit.json'):
        path = OUT / name
        audit = read(path)
        assert audit['passed']
        verify_hashes(audit['hashes'])
        paths.append(path)
    policies = {}
    for task in TASKS:
        for seed in range(3):
            folder = OUT / 'train' / ('%s_s%d' % (task, seed))
            done = read(folder / 'completed.json')
            assert done['passed']
            verify_hashes(done['hashes'])
            policy, fit = read(folder / 'policy.json'), read(folder / 'fit.json')
            assert policy['task'] == task and policy['kind'] in ('constant', 'tree')
            assert not fit['validation_access'] and not fit['test_access']
            policies['%s_s%d' % (task, seed)] = policy
            paths += [folder / n for n in ('completed.json', 'policy.json', 'fit.json')]
    value = dict(policies=policies, hashes={str(p): sha(p) for p in paths},
                 validation_accessed=False, test_accessed=False)
    path = OUT / 'fitted_policy_registration.json'
    if not path.exists():
        assert not (OUT / 'evaluations').exists()
    frozen_write(path, value)
    return value


def deployed_policy(task, family, seed, h):
    from latency_tree_policy import constant
    if family in ('adaptive', 'combined'):
        learned = read(OUT / 'fitted_policy_registration.json')['policies']['%s_s%d' % (task, seed)]
        return learned if family == 'adaptive' else dict(kind='combined', task=task, learned=learned)
    return dict(kind='certificate', task=task) if family == 'certificate' else constant(task, h)


def selected_jobs():
    selection = read(OUT / 'baseline_selection.json')
    verify_hashes(selection['hashes'])
    result = []
    for task in TASKS:
        nomination = selection['nominations'][task]
        for seed in range(3):
            result.append((task, 'adaptive', seed, 0))
            for label, family in (('independent', 'grid'), ('matched', 'matched')):
                result.append(normalize(task, family, seed, nomination[label + '_h']))
            if task == 'pendulum':
                result += [(task, 'certificate', seed, 0), (task, 'combined', seed, 0)]
    return sorted(set(result))


def confirmation():
    saved = read(OUT / 'confirmation_registration.json')
    assert saved['validation_gate_passed'] and saved['independent_review_passed']
    verify_hashes(saved['hashes'])
    assert saved['jobs'] == [list(j) for j in selected_jobs()]
    return saved


def aggregate(episodes, count):
    assert len(episodes) == count and [e['case'] for e in episodes] == list(range(count))
    counters = ('steps', 'success', 'constraint', 'initial_failed_steps',
                'solver_failure_steps', 'switches', 'retries')
    for e in episodes:
        assert all(type(e[k]) in (int, bool) and e[k] >= 0 for k in counters)
        assert e['steps'] > 0 and e['success'] in (0, 1) and e['constraint'] in (0, 1)
        assert e['initial_failed_steps'] <= e['steps'] and e['solver_failure_steps'] <= e['steps']
        assert all(math.isfinite(e[k]) for k in ('total_cost', 'physical_constraint_cost'))
    value = {k: sum(int(e[k]) for e in episodes) for k in counters}
    for k in ('total_cost', 'physical_constraint_cost'):
        value[k] = math.fsum(e[k] for e in episodes) / count
    value['initial_failure_rate'] = value['initial_failed_steps'] / value['steps']
    value['final_failure_rate'] = value['solver_failure_steps'] / value['steps']
    return value


def safe(a, b):
    assert a['steps'] > 0 and b['steps'] > 0
    return (a['success'] >= b['success'] and a['constraint'] <= b['constraint']
            and all(a[k] * b['steps'] <= b[k] * a['steps']
                    for k in ('initial_failed_steps', 'solver_failure_steps')))


def result_folder(task, family, seed, h, split='validation'):
    job = normalize(task, family, seed, h)
    return OUT / 'evaluations' / split / task / arm_name(job[1], seed, h)


def load(task, family, seed, h, split='validation'):
    assert split in ('validation', 'test')
    if split == 'test':
        confirmation()
    task, family, seed, h = normalize(task, family, seed, h)
    folder = result_folder(task, family, seed, h, split)
    done = read(folder / 'completed.json')
    assert done['passed']
    verify_hashes(done['hashes'])
    summary = read(folder / 'summary.json')
    assert (summary['task'], summary['family'], summary['seed'], summary['h'],
            summary['split'], summary['repeats']) == (task, family, seed, h, split, 1)
    rows = summary['episodes']
    assert all(r['repeat'] == 0 for r in rows)
    return dict(aggregate(rows, 64 if split == 'validation' else 128),
                task=task, family=family, seed=seed, h=h, path=str(folder), episodes=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--freeze-policies', action='store_true')
    args = parser.parse_args()
    result = register() if args.register else freeze_policies() if args.freeze_policies else verify_evaluation()
    print(json.dumps(dict(passed=True, frozen_files=len(result['hashes']), test_accessed=False)))
