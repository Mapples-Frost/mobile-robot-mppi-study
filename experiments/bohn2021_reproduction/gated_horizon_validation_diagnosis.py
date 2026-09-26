"""Post-hoc descriptive decomposition of the completed failed validation.

All scenes/seeds retained. Reads audited validation and timing only. No tests,
new simulation, policy selection, or changes to the registered effect gate.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from gated_horizon_latency_delivery import RUN, ROOT, guard, read, sha, resolve, relative

DEST = RUN / 'validation_diagnosis'
REG = RUN / 'validation_diagnosis_registration.json'
BASE = {'vehicle': 25, 'pendulum': 30}


def main():
    guard()
    hashes = {}

    def record(path, expected=None):
        name, value = relative(path), sha(path)
        assert expected is None or expected == value, 'Hash mismatch: ' + name
        assert name not in hashes or hashes[name] == value
        hashes[name] = value

    for name, value in read(REG)['source_hashes'].items():
        record(resolve(name), value)
    record(REG)
    source = RUN / 'effect_gate_review/report.json'
    reviewed = read(source)
    assert reviewed['passed'] and reviewed['validation_effect_passed'] is False
    record(source)
    for name, value in reviewed['hashes'].items():
        record(resolve(name), value)
    paired, coverage, concentration, time_rows, time_summary = [], [], [], [], []
    sources = {}

    def arm(task, family, seed, h):
        key = (task, family, seed, h)
        if key not in sources:
            folder = RUN / 'evaluations/validation' / task / ('%s_h%d_s%d' % (family, h, seed))
            done = read(folder / 'completed.json')
            assert done['passed']
            record(folder / 'completed.json')
            for name, value in done['hashes'].items():
                record(resolve(name), value)
            summary = read(folder / 'summary.json')
            assert len(summary['episodes']) == 32 and [r['case'] for r in summary['episodes']] == list(range(32))
            sources[key] = (folder, summary)
        return sources[key]

    for task in BASE:
        for seed in range(3):
            folder, data = arm(task, 'adaptive', seed, 0)
            _, base = arm(task, 'primary', seed, BASE[task])
            totals = Counter()
            for a, b in zip(data['episodes'], base['episodes']):
                path = folder / ('r0_trace_%02d.json' % a['case'])
                trace = read(path)
                assert len(trace) == a['steps']
                short = sum(r['horizon'] < BASE[task] for r in trace)
                assert all(r['horizon'] <= BASE[task] for r in trace)
                assert short == sum(bool(r['gate']['use_short']) for r in trace)
                totals.update(steps=len(trace), short_steps=short, short_cases=int(short > 0),
                              failed_cases=int(not a['success']), failed_cases_with_short=int(not a['success'] and short > 0),
                              switches=a['switches'])
                paired.append(dict(task=task, seed=seed, case=a['case'], steps=a['steps'], short_steps=short,
                                   adaptive_success=a['success'], primary_success=b['success'],
                                   total_delta=a['total_cost'] - b['total_cost'],
                                   physical_delta=a['physical_constraint_cost'] - b['physical_constraint_cost'],
                                   horizon_penalty_delta=a['h_penalty'] - b['h_penalty']))
            coverage.append(dict(task=task, seed=seed, cases=32, **totals,
                                 short_step_fraction=totals['short_steps'] / totals['steps']))
    for row in reviewed['comparisons']:
        a = row['adaptive']
        b = row['fixed']
        _, ad = arm(a['task'], a['family'], a['seed'], a['h'])
        _, bd = arm(b['task'], b['family'], b['seed'], b['h'])
        differences = [x['total_cost'] - y['total_cost'] for x, y in zip(ad['episodes'], bd['episodes'])]
        assert math.isclose(sum(differences) / 32, row['cost_difference']['mean'], rel_tol=1e-12, abs_tol=1e-12)
        ordered = sorted(enumerate(differences), key=lambda p: (-abs(p[1]), p[0]))
        absolute = sum(abs(d) for d in differences)
        positive = sum(max(d, 0) for d in differences)
        negative = sum(min(d, 0) for d in differences)
        concentration.append(dict(task=row['task'], seed=row['seed'], comparator=row['comparator'],
                                  mean_delta=sum(differences) / 32, harmed_cases=sum(d > 1e-10 for d in differences),
                                  improved_cases=sum(d < -1e-10 for d in differences),
                                  near_equal_cases=sum(abs(d) <= 1e-10 for d in differences),
                                  total_positive_deltas=positive, total_negative_deltas=negative,
                                  largest_absolute_case=ordered[0][0], largest_absolute_delta=ordered[0][1],
                                  largest_absolute_fraction=abs(ordered[0][1]) / absolute if absolute else 0.,
                                  second_largest_absolute_case=ordered[1][0], second_largest_absolute_delta=ordered[1][1]))
    for task in BASE:
        for seed in range(3):
            categories = {k: Counter() for k in ('all', 'first_attempt_failed', 'final_failed', 'short_h')}
            for repeat in (0, 1):
                folder = RUN / 'timing_validation' / ('r%d_%s_adaptive_h0_s%d' % (repeat, task, seed))
                done = read(folder / 'completed.json')
                assert done['passed'] and done['exact_replay']
                record(folder / 'completed.json')
                for name, value in done['hashes'].items():
                    record(resolve(name), value)
                for case in range(32):
                    rows = read(folder / ('r0_trace_%02d.json' % case))
                    for step, row in enumerate(rows):
                        tm = row['timing']
                        attempts = row['recovery']['attempts']
                        solver_s = sum(x['solver_s'] for x in attempts)
                        assert solver_s <= tm['controller_s'] + 1e-12
                        values = dict(steps=1, decision_s=tm['decision_s'], solver_s=solver_s,
                                      controller_other_s=tm['controller_s'] - solver_s,
                                      selection_s=tm['selection_s'], retries=len(attempts) - 1)
                        flags = dict(first_attempt_failed=not attempts[0]['success'],
                                     final_failed=not row['recovery']['final_success'], short_h=row['horizon'] < BASE[task])
                        time_rows.append(dict(task=task, seed=seed, repeat=repeat, case=case, step=step,
                                              horizon=row['horizon'], **flags, **values))
                        categories['all'].update(values)
                        for label, enabled in flags.items():
                            if enabled:
                                categories[label].update(values)
            for label, values in categories.items():
                time_summary.append(dict(task=task, seed=seed, category=label, **values,
                                         step_fraction=values['steps'] / categories['all']['steps'],
                                         decision_time_fraction=values['decision_s'] / categories['all']['decision_s']))
    assert len(paired) == 192 and len(concentration) == 12 and len(coverage) == 6 and len(time_summary) == 24
    for name, value in hashes.items():
        assert sha(resolve(name)) == value
    report = dict(passed=True, post_hoc=True, validation_effect_passed=False, cases=192,
                  coverage=coverage, concentration=concentration, timing=time_summary,
                  measured_adaptive_steps=len(time_rows), hashes=hashes, source_hash=sha(Path(__file__)),
                  new_simulation_steps=0, test_accessed=False,
                  caveats='Descriptive post-hoc decomposition, not causal evidence or a new acceptance test. Timing categories overlap. Solver intervals include all attempts; controller_other includes feasibility/recovery and internal bookkeeping. Concentrated differences do not justify excluding scenes.')
    if (DEST / 'report.json').exists():
        old = read(DEST / 'report.json')
        assert {k: v for k, v in old.items() if k != 'delivery_hashes'} == report
        for name, value in old['delivery_hashes'].items():
            assert sha(DEST / name) == value
    else:
        assert not DEST.exists(), 'Inspect partial diagnosis before retrying'
        DEST.mkdir()
        for name, rows in [('all_adaptive_scenes.csv', paired), ('horizon_coverage.csv', coverage),
                           ('cost_concentration.csv', concentration), ('adaptive_timing_steps.csv', time_rows),
                           ('adaptive_timing_categories.csv', time_summary)]:
            with (DEST / name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        report['delivery_hashes'] = {p.name: sha(p) for p in sorted(DEST.glob('*.csv'))}
        (DEST / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'coverage', 'concentration', 'timing')}, indent=2))


if __name__ == '__main__':
    main()
