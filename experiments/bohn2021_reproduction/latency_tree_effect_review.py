"""Independent aggregation, nomination, bootstrap and gate review.

Does not import production evaluator/selection/acceptance functions. This is an
independent arithmetic implementation, not an independent experiment or agent.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from latency_tree_protocol import OUT, read, sha, write

BASE = {'vehicle': 25, 'pendulum': 30}
TASKS = ('vehicle', 'pendulum')


def equal(a, b, label='value'):
    if isinstance(a, dict):
        assert set(a) <= set(b), (label, 'missing keys')
        for key in a:
            equal(a[key], b[key], label + '/' + key)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, label + '/' + str(i))
    elif isinstance(a, bool) or a is None:
        assert type(a) is type(b) and a == b, (label, a, b)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (label, a, b)
    else:
        assert a == b, (label, a, b)


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(Path(path)) == digest, path


def summarize(rows, n):
    assert len(rows) == n and [e['case'] for e in rows] == list(range(n))
    result = {}
    for field in ('steps', 'success', 'constraint', 'initial_failed_steps',
                  'solver_failure_steps', 'switches', 'retries'):
        assert all(type(e[field]) in (int, bool) and e[field] >= 0 for e in rows)
        result[field] = sum(int(e[field]) for e in rows)
    for e in rows:
        assert e['steps'] > 0 and e['success'] in (0, 1) and e['constraint'] in (0, 1)
        assert 0 <= e['initial_failed_steps'] <= e['steps']
        assert 0 <= e['solver_failure_steps'] <= e['steps']
    for field in ('total_cost', 'physical_constraint_cost'):
        assert all(math.isfinite(e[field]) for e in rows)
        result[field] = math.fsum(e[field] for e in rows) / n
    for field, source in (('initial_failure_rate', 'initial_failed_steps'),
                          ('final_failure_rate', 'solver_failure_steps')):
        result[field] = result[source] / result['steps']
    return result


def safety(a, b):
    if a['success'] < b['success'] or a['constraint'] > b['constraint']:
        return False
    from fractions import Fraction
    return all(Fraction(a[k], a['steps']) <= Fraction(b[k], b['steps'])
               for k in ('initial_failed_steps', 'solver_failure_steps'))


def cost_ci(deltas):
    matrix = np.atleast_2d(np.array(deltas, dtype=float))
    assert matrix.shape[0] in (1, 3) and matrix.shape[1] > 1 and np.isfinite(matrix).all()
    n = matrix.shape[1]
    indices = np.random.RandomState(2609268100).randint(n, size=(10000, n))
    # Independent reduction order; all model rows share the paired scene sample.
    means = np.mean(matrix[:, indices], axis=(0, 2))
    return {'mean': float(matrix.mean()),
            'lower': float(np.percentile(means, 2.5)), 'upper': float(np.percentile(means, 97.5))}


def time_ci(a_times, b_times, a_rows, b_rows):
    n = len(a_rows)
    assert len(b_rows) == n and n > 1
    assert len(a_times) == len(b_times) == 2
    values = []
    for times, rows in ((a_times, a_rows), (b_times, b_rows)):
        assert [r['case'] for r in rows] == list(range(n))
        steps = np.array([r['steps'] for r in rows], dtype=int)
        assert (steps > 0).all()
        assert all(len(repeat) == n and [r['case'] for r in repeat] == list(range(n)) for repeat in times)
        assert all(np.array_equal(steps, [r['steps'] for r in repeat]) for repeat in times)
        totals = np.array([[r['decision_total_s'] for r in repeat] for repeat in times], dtype=float)
        assert np.isfinite(totals).all() and (totals > 0).all()
        values.append((totals, steps))
    (at, astep), (bt, bstep) = values
    indexes = np.random.RandomState(2609268101).randint(n, size=(10000, n))
    am, bm = (at[0] + at[1]) / 2, (bt[0] + bt[1]) / 2
    aa = np.sum(am[indexes], axis=1) / np.sum(astep[indexes], axis=1)
    bb = np.sum(bm[indexes], axis=1) / np.sum(bstep[indexes], axis=1)
    ratios = aa / bb
    return {'mean_ratio': float((np.sum(am) / np.sum(astep)) / (np.sum(bm) / np.sum(bstep))),
            'lower': float(np.percentile(ratios, 2.5)), 'upper': float(np.percentile(ratios, 97.5)),
            'repeat_ratios': [float((np.sum(at[r]) / np.sum(astep)) /
                                    (np.sum(bt[r]) / np.sum(bstep))) for r in range(2)]}


def comparison(a, b, a_rows, b_rows, timing):
    ci = cost_ci([x['total_cost'] - y['total_cost'] for x, y in zip(a_rows, b_rows)])
    relative = ci['mean'] / abs(b['total_cost']) if b['total_cost'] != 0 else None
    physical = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(b['physical_constraint_cost'])
    total = a['total_cost'] <= b['total_cost'] + .02 * abs(b['total_cost'])
    cost_pass = relative is not None and relative <= -.03 and ci['upper'] < 0
    time_pass = (timing is not None and timing['mean_ratio'] <= .9 and timing['upper'] < 1
                 and max(timing['repeat_ratios']) < 1 and total)
    return dict(cost_difference=ci, relative_cost_difference=relative, safe=safety(a, b),
                control_noninferior=bool(physical), total_noninferior=bool(total),
                adapted=a['switches'] > 0, cost_gain=bool(cost_pass),
                time_gain=bool(time_pass), timing=timing)


def combined(rows):
    assert len(rows) == 3
    safety_pass = all(r['safe'] and r['control_noninferior'] for r in rows)
    adaptation = all(r['adapted'] for r in rows)
    cost = all(r['cost_gain'] for r in rows)
    timing = all(r['time_gain'] for r in rows)
    measured = all(r['timing'] is not None for r in rows)
    return dict(passed=safety_pass and adaptation and measured and (cost or timing),
                safe_all_seeds=safety_pass, adapted_all_seeds=adaptation,
                cost_route_all_seeds=cost, time_route_all_seeds=timing,
                timing_complete=measured)


def review(split):
    assert split in ('validation', 'test')
    from latency_tree_protocol import verify as verify_training
    verify_training()
    registration = read(OUT / 'evaluation_registration.json')
    verify(registration['hashes'])
    n = 64 if split == 'validation' else 128
    if split == 'test':
        confirm = read(OUT / 'confirmation_registration.json')
        assert confirm['validation_gate_passed'] and confirm['independent_review_passed']
        verify(confirm['hashes'])
    gate_path = OUT / (split + '_delivery/effect_gate.json')
    gate = read(gate_path)
    assert gate['passed_calculation'] and gate['timing_audited'] and gate['raw_records_audited']
    verify(gate['hashes'])
    hashes = {str(gate_path): sha(gate_path), str(Path(__file__).resolve()): sha(Path(__file__)),
              str(OUT / 'evaluation_registration.json'): sha(OUT / 'evaluation_registration.json')}
    cache, timing_cache = {}, {}

    def canonical(task, family, seed, h):
        return task, 'primary' if family in ('grid', 'matched') and h == BASE[task] else family, seed, h

    def raw(task, family, seed, h):
        key = canonical(task, family, seed, h)
        task, family, seed, h = key
        if key not in cache:
            folder = OUT / 'evaluations' / split / task / ('%s_h%d_s%d' % (family, h, seed))
            receipt = folder / 'completed.json'
            done = read(receipt)
            assert done['passed'] and str(receipt) in gate['hashes']
            verify(done['hashes'])
            s = read(folder / 'summary.json')
            assert (s['task'], s['family'], s['seed'], s['h'], s['split'], s['repeats']) == (
                task, family, seed, h, split, 1)
            rows = s['episodes']
            cache[key] = summarize(rows, n), rows
            hashes[str(receipt)] = sha(receipt)
        return cache[key]

    def times(task, family, seed, h):
        key = canonical(task, family, seed, h)
        if key not in timing_cache:
            task, family, seed, h = key
            result = []
            for repeat in (0, 1):
                folder = OUT / ('timing_' + split) / ('r%d_%s_%s_h%d_s%d' % (
                    repeat, task, family, h, seed))
                receipt = folder / 'completed.json'
                done = read(receipt)
                assert done['passed'] and done['exact_replay'] and str(receipt) in gate['hashes']
                verify(done['hashes'])
                result.append(read(folder / 'summary.json')['episodes'])
                hashes[str(receipt)] = sha(receipt)
            timing_cache[key] = result
        return timing_cache[key]

    selected = read(OUT / 'baseline_selection.json')
    verify(selected['hashes'])
    if split == 'validation':
        for task in TASKS:
            primary = [raw(task, 'primary', s, BASE[task])[0] for s in range(3)]
            eligible = []
            for h in range(5, 51, 5):
                summary = raw(task, 'grid', 0, h)[0]
                if safety(summary, primary[0]):
                    eligible.append((summary['total_cost'], h))
            assert min(eligible)[1] == selected['nominations'][task]['independent_h']
            eligible = []
            for h in range(5, 51, 5):
                rows = [raw(task, 'matched', s, h)[0] for s in range(3)]
                if all(safety(a, b) for a, b in zip(rows, primary)):
                    eligible.append((math.fsum(a['total_cost'] for a in rows) / 3, h))
            assert min(eligible)[1] == selected['nominations'][task]['matched_h']
    expected, comparisons = {}, []
    for task in TASKS:
        labels = {}
        for label, family in (('independent', 'grid'), ('matched', 'matched')):
            rows, deltas = [], []
            h = selected['nominations'][task][label + '_h']
            for seed in range(3):
                a, ar = raw(task, 'adaptive', seed, 0)
                b, br = raw(task, family, seed, h)
                tm = time_ci(times(task, 'adaptive', seed, 0), times(task, family, seed, h), ar, br)
                row = comparison(a, b, ar, br, tm)
                saved = [r for r in gate['comparisons'] if (
                    r['task'], r['comparator'], r['seed']) == (task, label, seed)]
                assert len(saved) == 1 and saved[0]['fixed_h'] == h
                equal(row, saved[0], '%s/%s/%d' % (task, label, seed))
                equal(a, saved[0]['adaptive'])
                equal(b, saved[0]['fixed'])
                rows.append(row)
                comparisons.append(dict(row, task=task, comparator=label, seed=seed))
                deltas.append([x['total_cost'] - y['total_cost'] for x, y in zip(ar, br)])
            labels[label] = dict(combined(rows), paired_scene_interval=cost_ci(deltas))
        common_cost = all(r['cost_route_all_seeds'] for r in labels.values())
        common_time = all(r['time_route_all_seeds'] for r in labels.values())
        expected[task] = dict(labels, common_cost_route=common_cost, common_time_route=common_time,
                              passed=all(r['passed'] for r in labels.values()) and (common_cost or common_time))
    assert len(gate['comparisons']) == 12
    equal(expected, gate['effects'])
    assert gate['effect_passed'] == all(g['passed'] for g in expected.values())
    checked_diagnostics = 0
    for seed in range(3):
        for family in ('certificate', 'combined'):
            labels = ['adaptive', 'independent', 'matched'] + (['certificate'] if family == 'combined' else [])
            for label in labels:
                if label in ('independent', 'matched'):
                    bfamily = 'grid' if label == 'independent' else 'matched'
                    bh = selected['nominations']['pendulum'][label + '_h']
                else:
                    bfamily, bh = label, 0
                a, ar = raw('pendulum', family, seed, 0)
                b, br = raw('pendulum', bfamily, seed, bh)
                row = comparison(a, b, ar, br, time_ci(
                    times('pendulum', family, seed, 0), times('pendulum', bfamily, seed, bh), ar, br))
                saved = [r for r in gate['diagnostics'] if (
                    r['family'], r['seed'], r['comparator']) == (family, seed, label)]
                assert len(saved) == 1 and saved[0]['contributes_to_main_gate'] is False
                equal(row, saved[0])
                checked_diagnostics += 1
    assert checked_diagnostics == len(gate['diagnostics']) == 21
    output = dict(passed=True, effect_passed=gate['effect_passed'], effects=expected,
                  comparisons_checked=12, diagnostic_comparisons_checked=21,
                  fixed_nominations_checked=split == 'validation', split=split,
                  hashes=hashes, simulations=0, test_accessed=split == 'test',
                  scope='Independent arithmetic and selection implementation over already audited '
                        'records; not an independent run of the experiment. Negative gates stay negative.')
    target = OUT / (split + '_delivery/independent_review.json')
    if target.exists():
        equal(output, read(target))
    else:
        write(target, output)
    print(json.dumps({k: v for k, v in output.items() if k not in ('hashes', 'effects')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    review(parser.parse_args().split)

