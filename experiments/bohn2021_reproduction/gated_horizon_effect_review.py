"""Independent arithmetic review of the frozen validation effect gate.

No production selection/report functions are imported. Formal review waits for
all validation/timing controllers to finish; it never opens test trajectories.
This checks existing criteria, not a new analysis or a relaxed acceptance rule.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
REG = RUN / 'effect_review_registration.json'
DEST = RUN / 'effect_gate_review'
BASE = {'vehicle': 25, 'pendulum': 30}
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
np = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name):
    p = ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)
    return (p if p.is_absolute() else ROOT / p).resolve()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def guard():
    proc = Path('/proc') if Path('/proc').exists() else Path('//wsl.localhost/Ubuntu-20.04/proc')
    for name in ('validation_finish_status.json', 'validation_status.json',
                 'posttrain_status.json', 'baseline_completion_status.json'):
        state = read(RUN / name)
        if state.get('active') or not state.get('complete'):
            raise RuntimeError('PENDING: ' + name + ' is not complete')
        if (proc / str(state['pid']) / 'cmdline').exists():
            raise RuntimeError('PENDING: controller process still exists: ' + str(state['pid']))


def compare(expected, actual, label):
    if isinstance(expected, dict):
        assert set(expected) <= set(actual), ('Missing keys', label)
        for k, v in expected.items():
            compare(v, actual[k], label + '/' + k)
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual), ('Length', label)
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, label + '/' + str(i))
    elif isinstance(expected, bool):
        assert isinstance(actual, bool) and expected == actual, ('Boolean mismatch', label, expected, actual)
    elif isinstance(expected, int):
        assert type(actual) is int and expected == actual, ('Count mismatch', label, expected, actual)
    elif isinstance(expected, float):
        assert math.isfinite(expected) and math.isfinite(actual)
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12), ('Numeric mismatch', label, expected, actual)
    else:
        assert expected == actual, ('Value mismatch', label, expected, actual)


def aggregate(rows):
    assert len(rows) == 32 and [r['case'] for r in rows] == list(range(32)), 'Missing or duplicate scenes'
    counts = {'steps': 'steps', 'success': 'success', 'constraints': 'constraint',
              'initial_failures': 'initial_failed_steps', 'final_failures': 'solver_failure_steps',
              'switches': 'switches', 'retries': 'retries'}
    for row in rows:
        assert all(isinstance(row[k], (int, bool)) and row[k] >= 0 for k in counts.values())
        assert row['steps'] > 0 and row['success'] in (0, 1) and row['constraint'] in (0, 1)
        assert row['initial_failed_steps'] <= row['steps'] and row['solver_failure_steps'] <= row['steps']
        assert all(math.isfinite(row[k]) for k in ('total_cost', 'physical_constraint_cost'))
    values = {k: int(sum(r[v] for r in rows)) for k, v in counts.items()}
    for k in ('total_cost', 'physical_constraint_cost'):
        values[k] = float(np.mean([r[k] for r in rows]))
    for name in ('initial', 'final'):
        values[name + '_failure_rate'] = values[name + '_failures'] / values['steps']
    return values


def cost_interval(differences):
    values = np.asarray(differences, dtype=float)
    assert values.ndim in (1, 2) and values.shape[-1] == 32 and np.isfinite(values).all()
    # Paired scene indices are shared across seeds. Seeds are not resampled.
    draws = np.random.RandomState(2609257100).randint(32, size=(10000, 32))
    scene_means = values if values.ndim == 1 else values.mean(axis=0)
    distribution = scene_means[draws].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.percentile(distribution, 2.5)),
                upper=float(np.percentile(distribution, 97.5)))


def time_interval(adaptive, fixed, a_rows, b_rows):
    assert len(adaptive) == len(fixed) == 2
    arrays = []
    for repeats, validation in ((adaptive, a_rows), (fixed, b_rows)):
        assert all(len(r) == 32 and [e['case'] for e in r] == list(range(32)) for r in repeats)
        step_counts = np.array([r['steps'] for r in validation], dtype=int)
        assert all(np.array_equal([e['steps'] for e in r], step_counts) for r in repeats)
        totals = np.array([[e['decision_total_s'] for e in r] for r in repeats], dtype=float)
        assert np.isfinite(totals).all() and (totals > 0).all()
        arrays.append((totals, step_counts))
    (at, an), (bt, bn) = arrays
    # The declared estimand is the ratio of total time / total control steps.
    ix = np.random.RandomState(2609257101).randint(32, size=(10000, 32))
    a_boot = at.mean(axis=0)[ix].sum(axis=1) / an[ix].sum(axis=1)
    b_boot = bt.mean(axis=0)[ix].sum(axis=1) / bn[ix].sum(axis=1)
    ratios = a_boot / b_boot
    repeats = [float((at[i].sum() / an.sum()) / (bt[i].sum() / bn.sum())) for i in range(2)]
    return dict(mean_ratio=float((at.mean(axis=0).sum() / an.sum()) / (bt.mean(axis=0).sum() / bn.sum())),
                lower=float(np.percentile(ratios, 2.5)), upper=float(np.percentile(ratios, 97.5)),
                repeat_ratios=repeats)


def flags(a, b, cost, timing):
    safety = a['success'] >= b['success'] and a['constraints'] <= b['constraints']
    safety = safety and all(a[k] <= b[k] + 1e-12 for k in ('initial_failure_rate', 'final_failure_rate'))
    physical_ok = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(b['physical_constraint_cost'])
    total_ok = a['total_cost'] <= b['total_cost'] + .02 * abs(b['total_cost'])
    relative_cost = cost['mean'] / max(abs(b['total_cost']), 1e-12)
    cost_gain = relative_cost <= -.03 and cost['upper'] < 0
    time_gain = timing is not None and timing['mean_ratio'] <= .9 and timing['upper'] < 1
    time_gain = time_gain and all(r < 1 for r in timing['repeat_ratios']) and total_ok
    return dict(cost_difference=cost, relative_cost_difference=relative_cost, safe=bool(safety),
                control_noninferior=bool(physical_ok), total_noninferior=bool(total_ok),
                adapted=a['switches'] > 0, cost_gain=bool(cost_gain), time_gain=bool(time_gain), timing=timing)


def group_gate(rows):
    assert len(rows) == 3
    result = dict(safe_all_seeds=all(r['safe'] and r['control_noninferior'] for r in rows),
                  adapted_all_seeds=all(r['adapted'] for r in rows),
                  cost_route_all_seeds=all(r['cost_gain'] for r in rows),
                  time_route_all_seeds=all(r['time_gain'] for r in rows),
                  timing_complete=all(r['timing'] is not None for r in rows))
    result['passed'] = (result['safe_all_seeds'] and result['adapted_all_seeds'] and result['timing_complete']
                        and (result['cost_route_all_seeds'] or result['time_route_all_seeds']))
    return result


def self_check():
    b = dict(success=32, constraints=0, initial_failure_rate=0., final_failure_rate=0.,
             physical_constraint_cost=100., total_cost=100., switches=0)
    a = dict(b, total_cost=96., switches=1)
    cost = dict(mean=-4., lower=-5., upper=-3.)
    tm = dict(mean_ratio=.8, lower=.7, upper=.9, repeat_ratios=[.75, .85])
    good = flags(a, b, cost, tm)
    assert group_gate([good] * 3)['passed']
    checked = ['all_seeds_pass']
    for field, value in [('success', 31), ('constraints', 1), ('initial_failure_rate', .01),
                         ('final_failure_rate', .01), ('physical_constraint_cost', 102.01), ('switches', 0)]:
        bad = flags(dict(a, **{field: value}), b, cost, tm)
        assert not group_gate([good, good, bad])['passed']
        checked.append('single_seed_' + field)
    assert not group_gate([flags(a, b, cost, None)] * 3)['passed']
    checked.append('timing_required_for_cost_route')
    mixed = [dict(good, cost_gain=True, time_gain=False), dict(good, cost_gain=False, time_gain=True), good]
    assert not group_gate(mixed)['passed']
    checked.append('different_routes_cannot_be_pooled_across_seeds')
    flat = dict(mean=0., lower=-1., upper=1.)
    bad_repeat = dict(tm, repeat_ratios=[.59, 1.01])
    assert not flags(dict(a, total_cost=100.), b, flat, bad_repeat)['time_gain']
    checked.append('both_timing_repeats_must_improve')
    assert not flags(dict(a, total_cost=102.01), b, dict(mean=2.01, lower=1., upper=3.), tm)['time_gain']
    checked.append('timing_gain_requires_total_cost_noninferiority')
    negative_b = dict(b, total_cost=-100., physical_constraint_cost=-100.)
    assert flags(dict(a, total_cost=-98., physical_constraint_cost=-98.), negative_b, flat, tm)['total_noninferior']
    assert not flags(dict(a, total_cost=-97.99, physical_constraint_cost=-97.99), negative_b, flat, tm)['control_noninferior']
    checked.append('signed_cost_margin_uses_absolute_reference')
    assert not flags(a, b, dict(cost, upper=0.), tm)['cost_gain']
    assert not flags(a, b, cost, dict(tm, upper=1.))['time_gain']
    checked.append('strict_interval_upper_bounds')
    rows = [dict(case=i, steps=2, success=True, constraint=False, initial_failed_steps=0,
                 solver_failure_steps=0, switches=1, retries=0, total_cost=1., physical_constraint_cost=1.) for i in range(32)]
    for label, invalid in [('missing', rows[:-1]), ('duplicate', rows[:-1] + [rows[0]]),
                           ('nonfinite', [dict(rows[0], total_cost=float('nan'))] + rows[1:])]:
        try:
            aggregate(invalid)
        except AssertionError:
            checked.append('reject_' + label)
        else:
            raise AssertionError('Accepted ' + label)
    repeats_a = [[dict(case=i, steps=2, decision_total_s=1.) for i in range(32)] for _ in range(2)]
    repeats_b = [[dict(case=i, steps=2, decision_total_s=2.) for i in range(32)] for _ in range(2)]
    compare(dict(mean_ratio=.5, lower=.5, upper=.5, repeat_ratios=[.5, .5]),
            time_interval(repeats_a, repeats_b, rows, rows), 'known_constant_ratio')
    compare(dict(mean=-4., lower=-4., upper=-4.), cost_interval([-4.] * 32), 'known_constant_difference')
    checked.append('known_constant_bootstrap_intervals')
    changed = copy.deepcopy(repeats_a)
    changed[1][0]['steps'] = 1
    try:
        time_interval(changed, repeats_b, rows, rows)
    except AssertionError:
        checked.append('reject_changed_repeat_step_count')
    else:
        raise AssertionError('Accepted changed timing trajectory length')
    return dict(passed=True, checks=checked, count=len(checked))


def review():
    guard()  # Run before any outcome summaries are read.
    registration = read(REG)
    hashes = {}

    def record(path, expected=None):
        key = relative(path)
        assert '/evaluations/test/' not in key and '/timing_test/' not in key, 'Test access forbidden'
        value = sha(path)
        assert expected is None or value == expected, 'Hash mismatch: ' + key
        assert key not in hashes or hashes[key] == value, 'Source changed: ' + key
        hashes[key] = value

    def graph(mapping):
        for name, value in mapping.items():
            record(resolve(name), value)

    graph(registration['source_hashes'])
    record(REG)
    graph(read(RUN / 'validation_finish_status.json')['output_hashes'])
    validation = read(RUN / 'audit_validation.json')
    timing_audit = read(RUN / 'audit_timing_validation.json')
    assert validation['passed'] and timing_audit['passed']
    graph(validation['hashes'])
    graph(timing_audit['hashes'])
    selection = read(RUN / 'baseline_selection.json')
    assert set(selection['nominations']) == set(BASE) and not selection['pending_independent_baselines']
    graph(selection['hashes'])
    record(RUN / 'baseline_selection.json')
    delivered = read(RUN / 'validation_delivery/effect_gate.json')
    graph(delivered['hashes'])
    assert delivered['timing_audited'] and delivered['extra_baselines_audited']
    assert delivered['independent_test_unlocked'] is False
    expected_keys = {(t, label, seed) for t in BASE for label in ('independent', 'matched') for seed in range(3)}
    found = {(r['task'], r['comparator'], r['seed']): r for r in delivered['comparisons']}
    assert len(found) == len(delivered['comparisons']) == 12 and set(found) == expected_keys
    cache, timed, consumed = {}, {}, set()

    def arm(task, family, seed, h):
        family = 'primary' if h == BASE[task] else family
        key = (task, family, seed, h)
        if key in cache:
            return cache[key]
        name = '%s_h%d_s%d' % (family, h, seed)
        folder = RUN / 'evaluations/validation' / task / name
        done_path = folder / 'completed.json'
        assert str(done_path) in validation['hashes'], 'Arm absent from full validation audit'
        done = read(done_path)
        assert done['passed']
        graph(done['hashes'])
        summary = read(folder / 'summary.json')
        assert (summary['task'], summary['family'], summary['seed'], summary['h']) == key
        assert summary['split'] == 'validation' and summary['episodes'] == done['result']['episodes']
        rows = summary['episodes']
        values = aggregate(rows)
        values.update(task=task, family=family, seed=seed, h=h, path=str(folder))
        repeats = []
        for repeat in range(2):
            tf = RUN / 'timing_validation' / ('r%d_%s_%s' % (repeat, task, name))
            tp = tf / 'completed.json'
            assert str(tp) in timing_audit['hashes'], 'Arm absent from timing audit'
            td = read(tp)
            assert td['passed'] and td['exact_replay']
            assert td['hashes'][str(done_path)] == sha(done_path)
            graph(td['hashes'])
            ts = read(tf / 'summary.json')
            assert (ts['task'], ts['family'], ts['seed'], ts['h']) == key
            assert ts['repeat'] == repeat and ts['split'] == 'validation' and ts['smoke'] is False
            repeats.append(ts['episodes'])
            consumed.add(str(tp))
        cache[key] = (values, rows, repeats)
        return cache[key]

    results, groups = [], {}
    for task, nomination in selection['nominations'].items():
        groups[task] = {}
        for label, family in [('independent', 'grid'), ('matched', 'matched')]:
            per_seed, deltas = [], []
            for seed in range(3):
                a, ae, at = arm(task, 'adaptive', seed, 0)
                h = nomination[label + '_h']
                b, be, bt = arm(task, family, seed, h)
                differences = [x['total_cost'] - y['total_cost'] for x, y in zip(ae, be)]
                costs = cost_interval(differences)
                timing = time_interval(at, bt, ae, be)
                row = flags(a, b, costs, timing)
                row.update(task=task, comparator=label, seed=seed, fixed_h=h, adaptive=a, fixed=b)
                compare(row, found[(task, label, seed)], '%s/%s/%s' % (task, label, seed))
                results.append(row)
                per_seed.append(row)
                deltas.append(differences)
            group = group_gate(per_seed)
            group['paired_scene_interval'] = cost_interval(deltas)
            compare(group, delivered['effects'][task][label], task + '/' + label)
            groups[task][label] = group
        groups[task]['passed'] = all(groups[task][c]['passed'] for c in ('independent', 'matched'))
    assert set(delivered['effects']) == set(BASE)
    compare(groups, delivered['effects'], 'all_effects')
    expected_timing = {str(RUN / 'timing_validation/completed.json')} | consumed
    assert set(timing_audit['hashes']) == expected_timing, 'Missing or extra timing arms'
    assert timing_audit['conditions'] == len(consumed) == 2 * len(cache)
    assert timing_audit['episodes'] == len(consumed) * 32
    passed = all(g['passed'] for g in groups.values())
    assert delivered['validation_effect_passed'] is passed, 'Final effect verdict differs'
    # Recheck inputs at the end so a concurrent rewrite cannot silently pass.
    graph(dict(hashes))
    result = dict(passed=True, validation_effect_passed=passed, comparisons_checked=len(results),
                  unique_validation_arms=len(cache), timing_conditions=len(consumed),
                  effects=groups, comparisons=results, self_checks=self_check(), hashes=hashes,
                  source_hash=sha(Path(__file__)), test_accessed=False, new_simulation_steps=0,
                  scope='Independent reaggregation and fixed-rule bootstrap/acceptance check using already audited validation and timing records. This is not an independent experimental replication or an effect success claim.',
                  numeric_comparison_tolerance=dict(relative=1e-10, absolute=1e-12,
                      scope='Arithmetic comparison only; all acceptance inequalities and boolean verdicts use the original exact thresholds.'))
    path = DEST / 'report.json'
    if path.exists():
        assert read(path) == result, 'Existing review differs; preserve it and diagnose'
    else:
        DEST.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'validation_effect_passed', 'comparisons_checked',
                      'unique_validation_arms', 'timing_conditions', 'test_accessed', 'new_simulation_steps')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'audit'), required=True)
    args = parser.parse_args()
    if args.mode == 'audit':
        guard()
    import numpy as np
    if args.mode == 'check':
        print(json.dumps(self_check(), indent=2))
    else:
        review()
