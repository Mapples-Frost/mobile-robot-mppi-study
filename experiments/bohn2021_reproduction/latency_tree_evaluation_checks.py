"""Adversarial known-answer checks before any formal evaluation outcome."""
import argparse
import ast
import copy
import csv
import math
import tempfile
from pathlib import Path
from runtime import ROOT
import numpy as np
from latency_tree_evaluation_spec import (
    OUT, SCRIPTS, REQUIRED, read, sha, write, verify, aggregate, safe,
    initial_jobs, smoke_jobs, normalize, existing_model, specification,
)
from latency_tree_effect import assess, cost_interval, time_interval, group_gate, task_gate, csv_output
from latency_tree_effect_review import (
    summarize, safety, cost_ci, time_ci, comparison, combined, equal,
)


def run_checks():
    verify()
    checked = []

    def record(name, value):
        assert value, name
        checked.append(name)

    def rejects(name, function):
        try:
            function()
        except (AssertionError, ValueError, KeyError, TypeError):
            checked.append(name)
        else:
            raise AssertionError('Invalid input accepted: ' + name)

    def episodes(cost=100., physical=100., switches=1, n=8):
        return [dict(case=i, repeat=0, steps=10, success=True, constraint=False,
                     initial_failed_steps=0, solver_failure_steps=0, switches=switches,
                     retries=0, total_cost=cost, physical_constraint_cost=physical)
                for i in range(n)]

    def arm(rows):
        return dict(aggregate(rows, len(rows)), episodes=rows)

    def latency(rows, per_step):
        return [[dict(case=e['case'], steps=e['steps'],
                      decision_total_s=e['steps'] * per_step) for e in rows] for _ in range(2)]

    b_rows, a_rows = episodes(), episodes(cost=96.)
    a, b = arm(a_rows), arm(b_rows)
    tm = dict(mean_ratio=.8, lower=.75, upper=.9, repeat_ratios=[.8, .8])
    good = assess(a, b, tm)
    record('positive_all_seed_gate', group_gate([good] * 3)['passed'])
    equal(good, comparison(summarize(a_rows, 8), summarize(b_rows, 8), a_rows, b_rows, tm))
    record('independent_positive_calculation', True)
    for field, replacement in (
        ('success', 7), ('constraint', 1), ('initial_failed_steps', 1),
        ('solver_failure_steps', 1), ('physical_constraint_cost', 102.0001), ('switches', 0)
    ):
        changed = dict(a, **{field: replacement})
        bad = assess(changed, b, tm)
        record('one_seed_' + field + '_blocks', not group_gate([good, bad, good])['passed'])
    missing = assess(a, b, None)
    record('timing_required_even_for_cost_route', not group_gate([missing] * 3)['passed'])
    mixed = [dict(good, cost_gain=True, time_gain=False),
             dict(good, cost_gain=False, time_gain=True), good]
    record('same_route_across_seeds', not group_gate(mixed)['passed'])
    cost_group = group_gate([dict(good, time_gain=False)] * 3)
    time_group = group_gate([dict(good, cost_gain=False)] * 3)
    record('same_route_across_both_comparators',
           not task_gate(dict(independent=cost_group, matched=time_group))['passed'])
    record('same_route_positive_task',
           task_gate(dict(independent=cost_group, matched=cost_group))['passed'])
    record('cost_3percent_boundary', assess(arm(episodes(cost=97.)), b, tm)['cost_gain'])
    record('cost_under_3percent_rejected', not assess(arm(episodes(cost=97.00001)), b, tm)['cost_gain'])
    bound_a, bound_b = episodes(cost=100., n=2), episodes(cost=100., n=2)
    bound_a[0]['total_cost'] = 94.
    bound_result = assess(arm(bound_a), arm(bound_b), tm)
    record('strict_cost_interval_upper_zero', bound_result['cost_difference']['upper'] == 0.
           and not bound_result['cost_gain'])
    record('time_10percent_boundary', assess(b, b, dict(tm, mean_ratio=.9))['time_gain'])
    record('time_under_10percent_rejected', not assess(b, b, dict(tm, mean_ratio=.900001))['time_gain'])
    record('strict_time_interval', not assess(b, b, dict(tm, upper=1.))['time_gain'])
    record('both_repeats_faster', not assess(b, b, dict(tm, repeat_ratios=[.6, 1.]))['time_gain'])
    record('time_needs_total_NI', not assess(arm(episodes(cost=102.0001)), b, tm)['time_gain'])
    record('physical_NI_boundary', assess(arm(episodes(physical=102.)), b, tm)['control_noninferior'])
    zero = arm(episodes(cost=0., physical=0.))
    zero_result = assess(zero, zero, tm)
    record('zero_cost_no_invented_percentage',
           zero_result['relative_cost_difference'] is None and not zero_result['cost_gain'])
    equal(zero_result, comparison(summarize(zero['episodes'], 8), summarize(zero['episodes'], 8),
                                  zero['episodes'], zero['episodes'], tm))
    record('zero_cost_independent_agreement', True)
    negative = arm(episodes(cost=-100., physical=-100.))
    record('negative_reference_NI', assess(arm(episodes(cost=-98., physical=-98.)),
                                         negative, tm)['control_noninferior'])
    record('negative_reference_violation', not assess(arm(episodes(cost=-97.9, physical=-97.9)),
                                                     negative, tm)['total_noninferior'])
    strict_a = dict(a, steps=10**14, initial_failed_steps=1)
    strict_b = dict(b, steps=10**14, initial_failed_steps=0)
    record('tiny_failure_increase_not_tolerated', not safe(strict_a, strict_b))
    record('independent_fraction_safety', not safety(strict_a, strict_b))
    equal_rate_a = dict(a, steps=100, initial_failed_steps=2, solver_failure_steps=4)
    equal_rate_b = dict(b, steps=50, initial_failed_steps=1, solver_failure_steps=2)
    record('different_denominator_equal_rates', safe(equal_rate_a, equal_rate_b)
           and safety(equal_rate_a, equal_rate_b))
    for bad in (a_rows[:-1], a_rows[:-1] + [a_rows[0]],
                [dict(a_rows[0], total_cost=float('nan'))] + a_rows[1:],
                [dict(a_rows[0], steps=0)] + a_rows[1:],
                [dict(a_rows[0], initial_failed_steps=11)] + a_rows[1:]):
        index = len(checked)
        rejects('invalid_production_episodes_%d' % index, lambda bad=bad: aggregate(bad, 8))
        rejects('invalid_independent_episodes_%d' % index, lambda bad=bad: summarize(bad, 8))
    for n in (2, 64, 128):
        constant = cost_interval([-4.] * n)
        equal(constant, dict(mean=-4., lower=-4., upper=-4.))
        equal(constant, cost_ci([-4.] * n))
        record('known_cost_interval_n%d' % n, True)
        matrix = np.arange(3 * n, dtype=float).reshape(3, n) / 7
        equal(cost_interval(matrix), cost_ci(matrix))
        record('paired_shared_seed_indices_n%d' % n, True)
        r = episodes(n=n)
        at, bt = latency(r, .04), latency(r, .05)
        interval = time_interval(at, bt, r, r)
        equal(interval, time_ci(at, bt, r, r))
        equal(interval, dict(mean_ratio=.8, lower=.8, upper=.8, repeat_ratios=[.8, .8]))
        record('known_time_interval_n%d' % n, True)
    r = episodes(n=2)
    r[0]['steps'], r[1]['steps'] = 1, 100
    at, bt = latency(r, .04), latency(r, .05)
    at[0][0]['decision_total_s'] = at[1][0]['decision_total_s'] = .4
    weighted = time_interval(at, bt, r, r)
    equal(weighted['mean_ratio'], 4.4 / 5.05)
    equal(weighted, time_ci(at, bt, r, r))
    record('latency_weighted_by_actual_control_steps', True)
    for label, mutate in (
        ('bad_case', lambda x: x[0][0].update(case=1)),
        ('bad_steps', lambda x: x[1][0].update(steps=99)),
        ('zero_time', lambda x: x[0][0].update(decision_total_s=0)),
        ('nan_time', lambda x: x[1][0].update(decision_total_s=float('nan'))),
    ):
        bad = copy.deepcopy(at)
        mutate(bad)
        rejects(label + '_production', lambda bad=bad: time_interval(bad, bt, r, r))
        rejects(label + '_independent', lambda bad=bad: time_ci(bad, bt, r, r))
    rng = np.random.RandomState(2609268999)
    for i in range(25):
        ar, br = episodes(), episodes()
        for x, y in zip(ar, br):
            x['total_cost'] += rng.uniform(-7, 7)
            x['physical_constraint_cost'] += rng.uniform(-3, 3)
            x['success'] = bool(rng.randint(2))
            y['success'] = bool(rng.randint(2))
            x['initial_failed_steps'], y['initial_failed_steps'] = int(rng.randint(3)), int(rng.randint(3))
        aa, bb = arm(ar), arm(br)
        t = time_interval(latency(ar, .04), latency(br, .05), ar, br)
        p = assess(aa, bb, t)
        independent = comparison(summarize(ar, 8), summarize(br, 8), ar, br, t)
        equal(p, independent)
        equal(group_gate([p] * 3), combined([independent] * 3))
    record('25_random_independent_formula_crosschecks', True)
    jobs = initial_jobs()
    record('90_unique_full_grid_jobs', len(jobs) == len(set(jobs)) == 90)
    for task in TASKS:
        for seed in range(3):
            record('%s_s%d_all_matched_H' % (task, seed),
                   all(normalize(task, 'matched', seed, h) in jobs for h in range(5, 51, 5)))
        record(task + '_independent_seed0_all_H',
               all(normalize(task, 'grid', 0, h) in jobs for h in range(5, 51, 5)))
    record('smoke_covers_distinct_terminals',
           ('vehicle', 'grid', 0, 5) in smoke_jobs() and ('pendulum', 'grid', 0, 35) in smoke_jobs())
    record('C5_all_seeds_retained',
           all(('pendulum', f, s, 0) in jobs for f in ('certificate', 'combined') for s in range(3)))
    with tempfile.TemporaryDirectory(prefix='latency_tree_check_') as temp:
        path = Path(temp) / 'table.csv'
        rows = [dict(case=0, value=1.2), dict(case=1, value=-4)]
        csv_output(path, rows)
        csv_output(path, rows)
        with path.open(newline='') as stream:
            record('CSV_exact_reuse', len(list(csv.DictReader(stream))) == 2)
        rejects('CSV_overwrite_refused', lambda: csv_output(path, [dict(case=0, value=7)]))
    for name in REQUIRED:
        source = SCRIPTS / name
        ast.parse(source.read_text(), filename=str(source))
    record('all_evaluation_sources_parse', True)
    original = (SCRIPTS / 'conservative_fixed_train.py').read_text()
    expected = original.replace('from conservative_iteration import OUT,verify',
                                'from latency_tree_evaluation_spec import OUT,verify_evaluation as verify')
    expected = expected.replace("(task+'_validation.json')", "(task+'_validation_bank.json')")
    record('baseline_instrumentation_exact_reuse',
           expected == (SCRIPTS / 'latency_tree_fixed_train.py').read_text())
    available = 0
    for task in TASKS:
        for h in range(5, 51, 5):
            for seed in range(3):
                available += existing_model(task, h, seed) is not None
    record('all20_independent_seed0_models_available', available >= 24)
    paths = [SCRIPTS / n for n in REQUIRED]
    value = dict(passed=True, checks=checked, count=len(checked), simulations=0,
                 test_accessed=False, existing_models=available,
                 hashes={str(p): sha(p) for p in paths})
    target = OUT / 'evaluation_checks.json'
    if target.exists():
        assert read(target) == value
    else:
        write(target, value)
    print('Evaluation checks passed: %d; available inherited models: %d; no simulations/test outcomes'
          % (len(checked), available))


TASKS = ('vehicle', 'pendulum')

if __name__ == '__main__':
    run_checks()
