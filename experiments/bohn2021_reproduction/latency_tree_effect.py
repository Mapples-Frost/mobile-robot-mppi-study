"""All-case paired effect assessment; diagnostics cannot unlock confirmation."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from latency_tree_evaluation_spec import (
    OUT, EVAL_REG, TASKS, BASE, read, sha, verify_hashes, frozen_write, freeze_policies,
    initial_jobs, selected_jobs, normalize, arm_name, source_for, load, safe,
    confirmation, aggregate,
)

DRAWS = 10000
COST_SEED = 2609268100
TIME_SEED = 2609268101


def cost_interval(differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    assert values.ndim == 2 and values.shape[0] in (1, 3) and values.shape[1] > 1
    assert np.isfinite(values).all()
    # The same case index is drawn for every trained seed; seeds are not resampled.
    draws = np.random.RandomState(COST_SEED).randint(
        values.shape[1], size=(DRAWS, values.shape[1]))
    distribution = values.mean(axis=0)[draws].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.percentile(distribution, 2.5)),
                upper=float(np.percentile(distribution, 97.5)))


def time_interval(adaptive, fixed, a_rows, b_rows):
    n = len(a_rows)
    assert n == len(b_rows) and n > 1 and len(adaptive) == len(fixed) == 2
    arrays = []
    for repeats, expected in ((adaptive, a_rows), (fixed, b_rows)):
        assert len(expected) == n and [e['case'] for e in expected] == list(range(n))
        counts = np.array([e['steps'] for e in expected], dtype=int)
        assert (counts > 0).all()
        assert all(len(r) == n and [e['case'] for e in r] == list(range(n)) for r in repeats)
        assert all(np.array_equal([e['steps'] for e in r], counts) for r in repeats)
        totals = np.array([[e['decision_total_s'] for e in r] for r in repeats], dtype=float)
        assert np.isfinite(totals).all() and (totals > 0).all()
        arrays.append((totals, counts))
    (at, an), (bt, bn) = arrays
    ix = np.random.RandomState(TIME_SEED).randint(n, size=(DRAWS, n))
    am, bm = at.mean(axis=0), bt.mean(axis=0)
    ratios = ((am[ix].sum(axis=1) / an[ix].sum(axis=1)) /
              (bm[ix].sum(axis=1) / bn[ix].sum(axis=1)))
    return dict(
        mean_ratio=float((am.sum() / an.sum()) / (bm.sum() / bn.sum())),
        lower=float(np.percentile(ratios, 2.5)), upper=float(np.percentile(ratios, 97.5)),
        repeat_ratios=[float((at[r].sum() / an.sum()) / (bt[r].sum() / bn.sum()))
                       for r in range(2)])


def assess(a, b, timing):
    assert [e['case'] for e in a['episodes']] == [e['case'] for e in b['episodes']]
    interval = cost_interval([aa['total_cost'] - bb['total_cost']
                              for aa, bb in zip(a['episodes'], b['episodes'])])
    denominator = abs(b['total_cost'])
    relative = interval['mean'] / denominator if denominator > 0 else None
    control = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(
        b['physical_constraint_cost'])
    total = a['total_cost'] <= b['total_cost'] + .02 * abs(b['total_cost'])
    gain = relative is not None and relative <= -.03 and interval['upper'] < 0
    time_gain = (timing is not None and timing['mean_ratio'] <= .9 and timing['upper'] < 1
                 and all(r < 1 for r in timing['repeat_ratios']) and total)
    return dict(cost_difference=interval, relative_cost_difference=relative,
                safe=bool(safe(a, b)), control_noninferior=bool(control),
                total_noninferior=bool(total), adapted=a['switches'] > 0,
                cost_gain=bool(gain), time_gain=bool(time_gain), timing=timing)


def group_gate(rows):
    assert len(rows) == 3
    result = dict(safe_all_seeds=all(r['safe'] and r['control_noninferior'] for r in rows),
                  adapted_all_seeds=all(r['adapted'] for r in rows),
                  cost_route_all_seeds=all(r['cost_gain'] for r in rows),
                  time_route_all_seeds=all(r['time_gain'] for r in rows),
                  timing_complete=all(r['timing'] is not None for r in rows))
    result['passed'] = (result['safe_all_seeds'] and result['adapted_all_seeds']
                        and result['timing_complete']
                        and (result['cost_route_all_seeds'] or result['time_route_all_seeds']))
    return result


def task_gate(groups):
    assert set(groups) == {'independent', 'matched'}
    common_cost = all(g['cost_route_all_seeds'] for g in groups.values())
    common_time = all(g['time_route_all_seeds'] for g in groups.values())
    return dict(passed=all(g['passed'] for g in groups.values()) and (common_cost or common_time),
                common_cost_route=common_cost, common_time_route=common_time)


def timing_rows(a, split):
    result = []
    for repeat in range(2):
        folder = OUT / ('timing_' + split) / ('r%d_%s_%s' % (
            repeat, a['task'], arm_name(a['family'], a['seed'], a['h'])))
        done = read(folder / 'completed.json')
        assert done['passed'] and done['exact_replay']
        verify_hashes(done['hashes'])
        summary = read(folder / 'summary.json')
        assert summary['policy'] == read(Path(a['path']) / 'summary.json')['policy']
        result.append(summary['episodes'])
    return result


def lightweight(a):
    return {k: v for k, v in a.items() if k != 'episodes'}


def csv_output(path, rows):
    assert rows
    columns = sorted({k for row in rows for k in row})
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    value = stream.getvalue()
    if path.exists():
        with path.open(newline='') as existing:
            assert existing.read() == value
    else:
        with path.open('w', newline='') as target:
            target.write(value)


def report(split):
    freeze_policies()
    assert split in ('validation', 'test')
    audit_stage = 'full' if split == 'validation' else 'initial'
    audit_paths = [OUT / ('audit_evaluation_%s_%s.json' % (split, audit_stage)),
                   OUT / ('audit_evaluation_%s_timing.json' % split)]
    hashes = {str(EVAL_REG): sha(EVAL_REG)}
    for p in audit_paths:
        value = read(p)
        assert value['passed']
        verify_hashes(value['hashes'])
        hashes[str(p)] = sha(p)
    selection_path = OUT / 'baseline_selection.json'
    selection = read(selection_path)
    verify_hashes(selection['hashes'])
    hashes[str(selection_path)] = sha(selection_path)
    if split == 'test':
        jobs = [tuple(j) for j in confirmation()['jobs']]
        hashes[str(OUT / 'confirmation_registration.json')] = sha(OUT / 'confirmation_registration.json')
    else:
        completed = read(OUT / 'baseline_completion.json')
        assert completed['passed']
        verify_hashes(completed['hashes'])
        hashes[str(OUT / 'baseline_completion.json')] = sha(OUT / 'baseline_completion.json')
        jobs = sorted(set(initial_jobs() + [tuple(j) for j in selection['supplement_jobs']]))
    arms, all_episodes, pairs, comparisons, diagnostics, effects = {}, [], [], [], [], {}
    for job in jobs:
        a = load(*job, split=split)
        arms[tuple(job)] = a
        all_episodes += [dict(task=a['task'], family=a['family'], seed=a['seed'], h=a['h'], **e)
                         for e in a['episodes']]
        p = Path(a['path']) / 'completed.json'
        hashes[str(p)] = sha(p)
    time_cache = {}
    for job in selected_jobs():
        a = arms[job]
        time_cache[job] = timing_rows(a, split)
        for repeat in range(2):
            p = OUT / ('timing_' + split) / ('r%d_%s_%s' % (
                repeat, job[0], arm_name(job[1], job[2], job[3]))) / 'completed.json'
            hashes[str(p)] = sha(p)
    for task in TASKS:
        groups = {}
        for label, family in (('independent', 'grid'), ('matched', 'matched')):
            rows, differences = [], []
            for seed in range(3):
                aj = (task, 'adaptive', seed, 0)
                bj = normalize(task, family, seed, selection['nominations'][task][label + '_h'])
                a, b = arms[aj], arms[bj]
                row = assess(a, b, time_interval(time_cache[aj], time_cache[bj],
                                                a['episodes'], b['episodes']))
                row.update(task=task, comparator=label, seed=seed, fixed_h=b['h'],
                           adaptive=lightweight(a), fixed=lightweight(b))
                rows.append(row)
                comparisons.append(row)
                delta = []
                for aa, bb in zip(a['episodes'], b['episodes']):
                    delta.append(aa['total_cost'] - bb['total_cost'])
                    pairs.append(dict(task=task, comparator=label, seed=seed, case=aa['case'],
                                      adaptive_cost=aa['total_cost'], fixed_cost=bb['total_cost'],
                                      cost_difference=delta[-1], adaptive_success=aa['success'],
                                      fixed_success=bb['success'], adaptive_constraint=aa['constraint'],
                                      fixed_constraint=bb['constraint'],
                                      adaptive_initial_failures=aa['initial_failed_steps'],
                                      fixed_initial_failures=bb['initial_failed_steps'],
                                      adaptive_final_failures=aa['solver_failure_steps'],
                                      fixed_final_failures=bb['solver_failure_steps']))
                differences.append(delta)
            groups[label] = dict(group_gate(rows), paired_scene_interval=cost_interval(differences))
        effects[task] = dict(groups, **task_gate(groups))
    # These comparisons remain outside the main all-task gate.
    for seed in range(3):
        nomination = selection['nominations']['pendulum']
        references = [('adaptive', ('pendulum', 'adaptive', seed, 0))]
        references += [(label, normalize('pendulum', family, seed, nomination[label + '_h']))
                       for label, family in (('independent', 'grid'), ('matched', 'matched'))]
        for family in ('certificate', 'combined'):
            aj = ('pendulum', family, seed, 0)
            extras = [('certificate', ('pendulum', 'certificate', seed, 0))] if family == 'combined' else []
            for label, bj in references + extras:
                a, b = arms[aj], arms[bj]
                row = assess(a, b, time_interval(time_cache[aj], time_cache[bj],
                                                a['episodes'], b['episodes']))
                diagnostics.append(dict(row, task='pendulum', family=family, seed=seed,
                                        comparator=label, contributes_to_main_gate=False))
    result = dict(passed_calculation=True, effect_passed=all(e['passed'] for e in effects.values()),
                  split=split, effects=effects, comparisons=comparisons, diagnostics=diagnostics,
                  timing_audited=True, raw_records_audited=True, hashes=hashes,
                  independent_review_pending=True, test_unlocked=False,
                  scope='All trained seeds, both fixed comparator labels, one common route per task. '
                        'Intervals condition on fitted models and resample paired scenes, not seeds or repeats. '
                        'A calculation pass alone does not unlock test or establish original SAC reproduction.')
    dest = OUT / (split + '_delivery')
    dest.mkdir(exist_ok=True)
    frozen_write(dest / 'effect_gate.json', result)
    csv_output(dest / 'all_conditions.csv', [lightweight(a) for a in arms.values()])
    csv_output(dest / 'all_episodes.csv', all_episodes)
    csv_output(dest / 'paired_episodes.csv', pairs)
    flat = []
    for r in comparisons + diagnostics:
        flat.append(dict(task=r['task'], method=r.get('family', 'adaptive'), seed=r['seed'],
                         comparator=r['comparator'], safe=r['safe'], adapted=r['adapted'],
                         physical_noninferior=r['control_noninferior'],
                         total_noninferior=r['total_noninferior'],
                         relative_cost=r['relative_cost_difference'],
                         cost_delta=r['cost_difference']['mean'],
                         cost_upper=r['cost_difference']['upper'], time_ratio=r['timing']['mean_ratio'],
                         time_upper=r['timing']['upper'],
                         repeat0_ratio=r['timing']['repeat_ratios'][0],
                         repeat1_ratio=r['timing']['repeat_ratios'][1],
                         cost_gain=r['cost_gain'], time_gain=r['time_gain']))
    csv_output(dest / 'all_comparisons.csv', flat)
    print('All-case effect calculation complete; effect_passed=%s; independent review still required'
          % result['effect_passed'])


def freeze_confirmation():
    freeze_policies()
    gate_path = OUT / 'validation_delivery/effect_gate.json'
    review_path = OUT / 'validation_delivery/independent_review.json'
    gate, review = read(gate_path), read(review_path)
    assert gate['effect_passed'] and review['passed'] and review['effect_passed']
    verify_hashes(gate['hashes'])
    verify_hashes(review['hashes'])
    assert review['hashes'][str(gate_path)] == sha(gate_path)
    jobs = selected_jobs()
    paths = [gate_path, review_path, OUT / 'baseline_selection.json',
             OUT / 'fitted_policy_registration.json', EVAL_REG]
    for j in jobs:
        paths += [source_for(*j) / n for n in ('model.zip', 'manifest.json', 'completed.json')]
    value = dict(validation_gate_passed=True, independent_review_passed=True, jobs=jobs,
                 test_cases_per_task=128, hashes={str(p): sha(p) for p in sorted(set(paths))},
                 rule='Exactly the same safety, adaptation, cost/time route and two-repeat requirements. '
                      'All cases and failures retained; no test-driven retuning.')
    target = OUT / 'confirmation_registration.json'
    if not target.exists():
        assert not (OUT / 'evaluations/test').exists() and not (OUT / 'timing_test').exists()
    frozen_write(target, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--freeze-confirmation', action='store_true')
    args = parser.parse_args()
    freeze_confirmation() if args.freeze_confirmation else report(args.split)
