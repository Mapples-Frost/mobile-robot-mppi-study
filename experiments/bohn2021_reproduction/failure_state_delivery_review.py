"""Independent arithmetic and immutable-snapshot review; no simulations.

This intentionally does not import the generator's aggregation, comparison or
budget code. The existing dynamics audit covers raw-to-episode validation; this
review checks the additional episode-to-delivery and ledger transformations.
Do not run while serial diagnostic timing is active.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
DEST = OUT / 'delivery_review'
SPEC = {
    'post_hoc': True,
    'diagnostic_only': True,
    'scope': 'All24 conditions,576 episode summaries,18 comparisons,432 pairs,all6 CSV/Markdown hashes and the immutable pre-timing attempt ledger.',
    'independence': 'No imports of report aggregation, comparator or ledger functions. Raw-to-episode and dynamics validity inherit the frozen full audit; no repeated integrations.',
    'budget': 'Read exactly the stored pre-timing counter manifest; verify current bytes and independently sum each counter. Do not reclassify later timing as part of the earlier snapshot.',
    'numeric': 'Integers/booleans/identifiers exact; finite float results rel_tol=1e-11, abs_tol=1e-10. Scientific thresholds unchanged, no rounded decisions.',
    'no_simulation': True,
    'no_new_training': True,
    'test_access': False,
    'core_effect_evaluated': False,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        assert read(path) == value, ('Existing review differs', str(path))
        return
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def idle_required():
    p = OUT / 'timing_status.json'
    if p.exists():
        s = read(p)
        assert not s['active'], 'Do not review while serial timing is active'


def registration():
    return dict(SPEC, source_sha256=sha(Path(__file__)), inputs={
        str(OUT / p): sha(OUT / p) for p in (
            'registration.json', 'audit_smoke.json', 'audit_full.json',
            'delivery/report.json', 'serialization_repair/failure_receipt.json',
            'serialization_repair/amendment.json')})


def register():
    idle_required()
    DEST.mkdir(exist_ok=True)
    if (DEST / 'registration.json').exists():
        verify_registration()
    else:
        write_new(DEST / 'registration.json', registration())


def verify_registration():
    saved = read(DEST / 'registration.json')
    current = registration()
    amendment_path = DEST / 'verifier_repair/amendment.json'
    if amendment_path.exists():
        amendment = read(amendment_path)
        assert amendment['original_registration_sha256'] == sha(DEST / 'registration.json')
        assert amendment['old_source_sha256'] == saved['source_sha256']
        assert amendment['new_source_sha256'] == current['source_sha256']
        assert amendment['scientific_rules_changed'] is False
        for name, digest in amendment['archives'].items():
            assert sha(DEST / 'verifier_repair' / name) == digest
        current['source_sha256'] = saved['source_sha256']
    equal(saved, current, 'registration')


def percent_change(candidate, reference):
    # Relative change is undefined for a zero reference. NI uses the absolute
    # inequality independently; never hide a positive increase behind zero.
    return None if reference == 0 else 100 * (candidate - reference) / abs(reference)


def equal(actual, expected, name):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), ('keys', name, actual.keys(), expected.keys())
        for k, value in expected.items():
            equal(actual[k], value, name + '/' + k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), name
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, name + '/' + str(i))
    elif isinstance(expected, bool):
        assert isinstance(actual, bool) and actual == expected, (name, actual, expected)
    elif isinstance(expected, int):
        assert isinstance(actual, int) and not isinstance(actual, bool) and actual == expected, (name, actual, expected)
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), name
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), (name, actual, expected)
    else:
        assert actual == expected, (name, actual, expected)


def csv_equal(path, expected):
    with path.open(newline='', encoding='utf-8-sig') as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(expected), str(path)
    for i, (a, e) in enumerate(zip(saved, expected)):
        assert a.keys() == e.keys(), (str(path), i, 'columns')
        converted = {}
        for k, v in e.items():
            if isinstance(v, bool):
                assert a[k] in ('True', 'False')
                converted[k] = a[k] == 'True'
            elif isinstance(v, int):
                converted[k] = int(a[k])
            elif isinstance(v, float):
                converted[k] = float(a[k])
            else:
                converted[k] = a[k]
        equal(converted, e, path.name + '/' + str(i))


def totals(episodes):
    sums = {out: sum(int(e[src]) for e in episodes) for out, src in (
        ('steps', 'steps'), ('success', 'success'), ('constraints', 'constraint'),
        ('initial_failures', 'initial_failed_steps'), ('final_failures', 'solver_failure_steps'),
        ('retries', 'retries'), ('switches', 'switches'))}
    for name in ('total_cost', 'physical_constraint_cost', 'performance_cost', 'h_penalty', 'constraint_cost'):
        sums[name] = math.fsum(e[name] for e in episodes) / len(episodes)
    sums['initial_failure_rate'] = sums['initial_failures'] / sums['steps']
    sums['final_failure_rate'] = sums['final_failures'] / sums['steps']
    return sums


def comparison(task, seed, policy, reference, a, b, intervened):
    return dict(task=task, seed=seed, policy=policy, reference=reference,
        safe=(a['success'] >= b['success'] and a['constraints'] <= b['constraints']
              and a['initial_failures'] * b['steps'] <= b['initial_failures'] * a['steps']
              and a['final_failures'] * b['steps'] <= b['final_failures'] * a['steps']),
        physical_noninferior=(a['physical_constraint_cost'] - b['physical_constraint_cost']
                              <= 0.02 * abs(b['physical_constraint_cost'])),
        intervened=intervened,
        total_change_percent=percent_change(a['total_cost'], b['total_cost']),
        physical_change_percent=percent_change(a['physical_constraint_cost'], b['physical_constraint_cost']),
        success_delta=a['success'] - b['success'], constraints_delta=a['constraints'] - b['constraints'],
        initial_failure_rate_delta=a['initial_failure_rate'] - b['initial_failure_rate'],
        final_failure_rate_delta=a['final_failure_rate'] - b['final_failure_rate'],
        steps_delta=a['steps'] - b['steps'])


def self_check():
    idle_required()
    verify_registration()
    base = dict(steps=100, success=20, constraints=4, initial_failures=10, final_failures=5,
                initial_failure_rate=.1, final_failure_rate=.05,
                total_cost=100., physical_constraint_cost=100.)
    cases = []

    def evaluate(candidate, reference=None):
        return comparison('pendulum', 0, 'certificate5', 'fixed30', candidate, reference or base, True)

    def record(name, passed):
        assert passed, name
        cases.append(name)

    record('identical_reference_is_safe_and_noninferior', all(evaluate(dict(base))[k] for k in ('safe', 'physical_noninferior')))
    fewer = dict(base, steps=50, initial_failures=6, initial_failure_rate=.12)
    record('fewer_failures_but_higher_rate_is_unsafe', not evaluate(fewer)['safe'])
    same_rate = dict(base, steps=200, initial_failures=20, final_failures=10)
    record('equal_exact_rates_with_different_counts_preserved', evaluate(same_rate)['safe'])
    for field, value in (('success', 19), ('constraints', 5), ('initial_failures', 11), ('final_failures', 6)):
        record('reject_worse_' + field, not evaluate(dict(base, **{field: value}))['safe'])
    negative = dict(base, physical_constraint_cost=-100., total_cost=-110.)
    record('negative_physical_cost_boundary', evaluate(dict(negative, physical_constraint_cost=-98.), negative)['physical_noninferior'])
    record('negative_physical_cost_outside_boundary', not evaluate(dict(negative, physical_constraint_cost=-97.99), negative)['physical_noninferior'])
    zero = dict(base, physical_constraint_cost=0.)
    record('zero_reference_allows_no_positive_increase', not evaluate(dict(zero, physical_constraint_cost=1e-12), zero)['physical_noninferior'])
    record('zero_reference_percentage_is_undefined', evaluate(dict(zero), zero)['physical_change_percent'] is None)
    record('zero_total_reference_percentage_is_undefined', evaluate(dict(base, total_cost=0.), dict(base, total_cost=0.))['total_change_percent'] is None)
    record('negative_total_cost_uses_absolute_denominator', evaluate(dict(negative, total_cost=-121.), negative)['total_change_percent'] == -10.)
    rejected = False
    try:
        equal({'value': float('nan')}, {'value': 1.}, 'nonfinite')
    except AssertionError:
        rejected = True
    record('nonfinite_delivery_rejected', rejected)
    rejected = False
    try:
        equal([1, 2], [1, 2, 3], 'missing_case')
    except AssertionError:
        rejected = True
    record('missing_case_rejected', rejected)
    result = dict(passed=True, checks=cases, source_sha256=sha(Path(__file__)), simulations=0, test_accessed=False)
    write_new(DEST / 'checks.json', result)
    print(json.dumps(result, indent=2))


def review():
    idle_required()
    verify_registration()
    checks = read(DEST / 'checks.json')
    assert checks['passed'] and checks['source_sha256'] == sha(Path(__file__))
    report = read(OUT / 'delivery/report.json')
    audit = read(OUT / 'audit_full.json')
    smoke = read(OUT / 'audit_smoke.json')
    assert report['passed'] and audit['passed'] and smoke['passed']
    assert report['diagnostic_only'] and report['post_hoc'] and not report['validation_or_test']
    assert report['new_training_steps'] == 0 and not report['goal_complete']
    assert not report['timing_scheduled'] and report['serial_timing_required_before_speed_claim']
    assert not (OUT.parent / 'gated_horizon_search_2026-09-25/evaluations/test').exists()
    seen_hashes = {}

    def verify_hashes(values):
        for name, digest in values.items():
            if name in seen_hashes:
                assert seen_hashes[name] == digest
            assert sha(Path(name)) == digest, ('Hash mismatch', name)
            seen_hashes[name] = digest

    verify_hashes(report['hashes'])
    verify_hashes(audit['hashes'])
    verify_hashes({str(OUT / 'delivery' / p): h for p, h in report['delivery_hashes'].items()})
    jobs = [(t, s, p) for t, ps in (
        ('pendulum', ('fixed30', 'certificate5', 'certificate10', 'certificate15')),
        ('vehicle', ('fixed25', 'fixed30', 'fixed35', 'selected')))
        for s in range(3) for p in ps]
    assert read(OUT / 'registration.json')['jobs'] == [list(j) for j in jobs]
    rows, episodes, pairs, comparisons, lookup = [], [], [], [], {}
    reused_episodes = new_episodes = 0
    for task, seed, policy in jobs:
        folder = OUT / 'full' / ('%s_s%d_%s' % (task, seed, policy))
        done = read(folder / 'completed.json')
        verify_hashes(done['hashes'])
        summary = read(folder / 'summary.json')
        es = summary['episodes']
        assert len(es) == 24 and [e['case'] for e in es] == list(range(24))
        assert all(e['repeat'] == 0 for e in es)
        assert [summary[k] for k in ('task', 'seed', 'policy')] == [task, seed, policy]
        assert summary['reused'] == (policy in ('fixed30',) if task == 'pendulum' else policy in ('fixed25', 'selected'))
        reused_episodes += len(es) if summary['reused'] else 0
        new_episodes += 0 if summary['reused'] else len(es)
        agg = totals(es)
        checks = [g for g in audit['groups'] if (g['task'], g['seed'], g['policy']) == (task, seed, policy)]
        assert len(checks) == 1
        intervention_steps = checks[0]['intervention_steps']
        if not summary['reused']:
            assert sum(e['intervention_steps'] for e in es) == intervention_steps
        rows.append(dict(task=task, seed=seed, policy=policy, reused=summary['reused'], intervention_steps=intervention_steps, **agg))
        lookup[(task, seed, policy)] = (agg, es, intervention_steps)
        episodes.extend(dict(task=task, seed=seed, policy=policy, **{k: v for k, v in e.items() if k != 'intervention_steps'}) for e in es)
    for task, seed, policy in jobs:
        reference = 'fixed30' if task == 'pendulum' else 'fixed25'
        if policy == reference:
            continue
        a, ae, interventions = lookup[(task, seed, policy)]
        b, be, _ = lookup[(task, seed, reference)]
        comparisons.append(comparison(task, seed, policy, reference, a, b, interventions > 0))
        for x, y in zip(ae, be):
            pairs.append(dict(task=task, seed=seed, policy=policy, reference=reference, case=x['case'],
                total_delta=x['total_cost'] - y['total_cost'], physical_delta=x['physical_constraint_cost'] - y['physical_constraint_cost'],
                success_delta=int(x['success']) - int(y['success']), constraints_delta=int(x['constraint']) - int(y['constraint']),
                initial_failures_delta=x['initial_failed_steps'] - y['initial_failed_steps'],
                final_failures_delta=x['solver_failure_steps'] - y['solver_failure_steps']))
    assert (len(rows), len(episodes), len(comparisons), len(pairs), reused_episodes, new_episodes) == (24, 576, 18, 432, 216, 360)
    equal(report['comparisons'], comparisons, 'comparisons')
    for name, values in (('all_conditions.csv', rows), ('all_episodes.csv', episodes),
                         ('all_comparisons.csv', comparisons), ('all_paired_cases.csv', pairs)):
        csv_equal(OUT / 'delivery' / name, values)
    viable = [p for p in ('certificate5', 'certificate10', 'certificate15') if all(
        c['safe'] and c['physical_noninferior'] and c['intervened'] for c in comparisons if c['task'] == 'pendulum' and c['policy'] == p)]
    assert viable == report['timing_screen_candidates']
    ledger = report['budget']
    counters = ledger['counter_rows']
    assert len(counters) == ledger['counter_files'] == len({c['path'] for c in counters})
    expected_counters = set()
    for phase in ('full', 'smoke', 'serialization_repair'):
        for pattern in ('attempt_*.json', 'solver_attempts.json'):
            expected_counters.update(str(p) for p in (OUT / phase).rglob(pattern))
    assert {c['path'] for c in counters} == expected_counters, 'Missing or extra pre-timing counters'
    groups = {}
    fields = ('step_attempts', 'reset_attempts', 'solve_attempts', 'solve_completed', 'warmups', 'retries')
    for c in counters:
        path = Path(c['path'])
        verify_hashes({str(path): c['sha256']})
        phase = path.relative_to(OUT).parts[0]
        assert phase in ('full', 'smoke', 'serialization_repair')
        g = groups.setdefault(phase, {f: 0 for f in fields})
        raw = read(path)
        if c['kind'] == 'environment':
            assert c['steps'] == raw['step_calls'] and c['resets'] == raw['reset_calls']
            assert c['solver_attempts'] == c['solver_completed'] == 0
            g['step_attempts'] += raw['step_calls']
            g['reset_attempts'] += raw['reset_calls']
        else:
            assert c['kind'] == 'solver' and c['steps'] == c['resets'] == 0
            assert c['solver_attempts'] == raw['solve_attempts'] and c['solver_completed'] == raw['solve_completed']
            for f, src in (('solve_attempts', 'solve_attempts'), ('solve_completed', 'solve_completed'),
                           ('warmups', 'warmup_attempts'), ('retries', 'retry_attempts')):
                g[f] += raw[src]
    equal(groups, ledger['groups'], 'budget/groups')
    total = {f: sum(g[f] for g in groups.values()) for f in fields}
    equal(total, ledger['total'], 'budget/total')
    csv_equal(OUT / 'delivery/all_budget_counters.csv', counters)
    receipt = read(OUT / 'serialization_repair/failure_receipt.json')
    equal(report['failed_recording_attempts'], receipt['budget'], 'failed_recording')
    for k, v in receipt['budget'].items():
        assert groups['serialization_repair'][k] == v
    for phase, source in (('full', audit), ('smoke', smoke)):
        for target, origin in (('step_attempts', 'new_control_steps'), ('reset_attempts', 'new_resets'), ('solve_attempts', 'new_solver_calls')):
            assert groups[phase][target] == source[origin]
    for target, origin in (('step_attempts', 'new_control_steps'), ('reset_attempts', 'new_resets'), ('solve_attempts', 'new_solver_calls')):
        assert total[target] == audit[origin] + smoke[origin] + receipt['budget'][target]
    assert report['new_numerical_integrations'] == audit['new_numerical_integrations'] + smoke['new_numerical_integrations']
    assert sum(r['steps'] for r in rows) == audit['recorded_steps'] == 41135
    result = dict(passed=True, post_hoc=True, diagnostic_only=True, conditions=len(rows), episodes=len(episodes),
        comparisons=len(comparisons), paired_cases=len(pairs), reused_episodes=reused_episodes, new_episodes=new_episodes,
        timing_screen_candidates=viable, pre_timing_budget=total, counter_files=len(counters), verified_hashes=len(seen_hashes),
        inputs=registration()['inputs'], registration_sha256=sha(DEST / 'registration.json'),
        checks_sha256=sha(DEST / 'checks.json'),
        limitations=SPEC['independence'], simulations=0, new_training_steps=0, test_accessed=False, core_effect_evaluated=False)
    write_new(DEST / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
        print('Independent delivery review registered; no simulation')
    elif args.check:
        self_check()
    else:
        review()
