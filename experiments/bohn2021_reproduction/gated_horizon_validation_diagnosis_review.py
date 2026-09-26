"""Independent, read-only raw-record review of post-hoc validation diagnosis.

The original diagnosis and its registration stay byte-identical. A supplemental
table explicitly writes zeros for empty categories; no data are imputed into
non-empty categories. Does not simulate, select policies, or read test outcomes.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
SOURCE = RUN / 'validation_diagnosis'
DEST = RUN / 'validation_diagnosis_review'
METRICS = ('steps', 'decision_s', 'solver_s', 'controller_other_s', 'selection_s', 'retries')
BASE = {'vehicle': 25, 'pendulum': 30}
PREFIX = '/home/mapples/projects/mobile-robot-mppi-study/'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name):
    p = Path(name[len(PREFIX):]) if name.startswith(PREFIX) else Path(name)
    return p if p.is_absolute() else ROOT / p


def relative(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def close(a, b, label):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (label, a, b)


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    hashes = {}

    def record(path, expected=None):
        value = sha(path)
        assert expected is None or value == expected, ('Hash mismatch', str(path))
        hashes[relative(path)] = value

    record(Path(__file__))
    for name in ('validation_finish_status.json', 'validation_status.json', 'posttrain_status.json', 'baseline_completion_status.json'):
        state = read(RUN / name)
        assert state.get('complete') and not state.get('active')
        assert not (Path('/proc') / str(state['pid']) / 'cmdline').exists()
        record(RUN / name)
    assert not (RUN / 'evaluations/test').exists()
    assert not (RUN / 'confirmation_registration.json').exists()
    diagnosis = read(SOURCE / 'report.json')
    assert diagnosis['passed'] and diagnosis['post_hoc'] and not diagnosis['validation_effect_passed']
    record(SOURCE / 'report.json')
    for name, value in diagnosis['hashes'].items():
        record(resolve(name), value)
    for name, value in diagnosis['delivery_hashes'].items():
        record(SOURCE / name, value)
    latency_path = RUN / 'validation_latency_delivery/pooled_arms.csv'
    manifest = read(RUN / 'validation_latency_delivery/manifest.json')
    record(RUN / 'validation_latency_delivery/manifest.json')
    # Its file hash is also checked below against the recorded delivery manifest.
    for name, value in manifest.get('hashes', {}).items():
        record(resolve(name), value)
    for name, value in manifest.get('delivery_hashes', {}).items():
        p = RUN / 'validation_latency_delivery' / name
        record(p, value)
    record(latency_path)
    pooled = {(r['task'], int(r['seed'])): r for r in table(latency_path) if r['family'] == 'adaptive'}
    stored_steps = table(SOURCE / 'adaptive_timing_steps.csv')
    lookup = {(r['task'], int(r['seed']), int(r['repeat']), int(r['case']), int(r['step'])): r for r in stored_steps}
    assert len(lookup) == len(stored_steps) == 25578
    expected = {(r['task'], r['seed'], r['category']): r for r in diagnosis['timing']}
    assert len(expected) == 24
    coverage = {(r['task'], r['seed']): r for r in diagnosis['coverage']}
    rows, missing_zero_fields, checked_steps = [], [], 0
    for task, base_h in BASE.items():
        for seed in range(3):
            cats = {k: {m: 0 for m in METRICS} for k in ('all', 'first_attempt_failed', 'final_failed', 'short_h')}
            for repeat in (0, 1):
                folder = RUN / 'timing_validation' / ('r%d_%s_adaptive_h0_s%d' % (repeat, task, seed))
                done = read(folder / 'completed.json')
                assert done['passed'] and done['exact_replay']
                record(folder / 'completed.json')
                for name, value in done['hashes'].items():
                    record(resolve(name), value)
                for case in range(32):
                    trace = read(folder / ('r0_trace_%02d.json' % case))
                    for step, r in enumerate(trace):
                        saved = lookup[(task, seed, repeat, case, step)]
                        assert int(saved['horizon']) == r['horizon']
                        attempts = r['recovery']['attempts']
                        flags = {'first_attempt_failed': not attempts[0]['success'],
                                 'final_failed': not r['recovery']['final_success'], 'short_h': r['horizon'] < base_h}
                        assert all(saved[k] == str(v) for k, v in flags.items())
                        solver = math.fsum(a['solver_s'] for a in attempts)
                        values = dict(steps=1, decision_s=r['timing']['decision_s'], solver_s=solver,
                                      controller_other_s=r['timing']['controller_s'] - solver,
                                      selection_s=r['timing']['selection_s'], retries=len(attempts) - 1)
                        close(values['decision_s'], values['solver_s'] + values['controller_other_s'] + values['selection_s'], 'time identity')
                        for k, v in values.items():
                            close(float(saved[k]), v, k)
                            cats['all'][k] += v
                            for label, enabled in flags.items():
                                if enabled:
                                    cats[label][k] += v
                        checked_steps += 1
            all_row = cats['all']
            assert all_row['steps'] == int(pooled[(task, seed)]['decision_steps']) == 2 * coverage[(task, seed)]['steps']
            close(all_row['decision_s'], float(pooled[(task, seed)]['decision_total_s']), 'pooled decision total')
            close(all_row['selection_s'], float(pooled[(task, seed)]['selection_total_s']), 'pooled selection total')
            assert cats['short_h']['steps'] == 2 * coverage[(task, seed)]['short_steps']
            summary_path = RUN / 'evaluations/validation' / task / ('adaptive_h0_s%d' % seed) / 'summary.json'
            summary = read(summary_path)
            record(summary_path)
            assert cats['first_attempt_failed']['steps'] == 2 * sum(e['initial_failed_steps'] for e in summary['episodes'])
            assert cats['final_failed']['steps'] == 2 * sum(e['solver_failure_steps'] for e in summary['episodes'])
            assert all_row['retries'] == 2 * sum(e['retries'] for e in summary['episodes'])
            failed_short = 0
            for e in summary['episodes']:
                if not e['success']:
                    trace_path = summary_path.parent / ('r0_trace_%02d.json' % e['case'])
                    record(trace_path)
                    failed_short += int(any(r['horizon'] < base_h for r in read(trace_path)))
            assert failed_short == coverage[(task, seed)]['failed_cases_with_short']
            for label, values in cats.items():
                original = expected[(task, seed, label)]
                for metric, value in values.items():
                    if metric not in original:
                        assert values['steps'] == 0 and value == 0
                        missing_zero_fields.append(dict(task=task, seed=seed, category=label, field=metric))
                    else:
                        close(value, original[metric], metric)
                step_fraction = values['steps'] / all_row['steps']
                time_fraction = values['decision_s'] / all_row['decision_s']
                close(step_fraction, original['step_fraction'], 'step fraction')
                close(time_fraction, original['decision_time_fraction'], 'decision fraction')
                rows.append(dict(task=task, seed=seed, category=label, **values,
                                 step_fraction=step_fraction, decision_time_fraction=time_fraction))
    assert checked_steps == len(lookup)
    for name, value in hashes.items():
        assert sha(resolve(name)) == value
    result = dict(passed=True, post_hoc=True, raw_timing_steps_checked=checked_steps,
                  adaptive_arms_checked=6, timing_categories_checked=24, empty_categories=5,
                  explicit_zero_fields=missing_zero_fields, input_hashes=hashes,
                  original_diagnosis_unchanged=True, new_simulation_steps=0, test_accessed=False,
                  validation_effect_passed=False,
                  scope='Independent raw-record accounting and zero-schema clarification only; no new effect test or causal attribution.')
    assert len(missing_zero_fields) == 30
    DEST.mkdir(exist_ok=True)
    output = DEST / 'adaptive_timing_categories_explicit_zero.csv'
    if output.exists():
        old = table(output)
        assert len(old) == len(rows)
        for a, b in zip(old, rows):
            for k, v in b.items():
                assert a[k] == str(v), (k, a[k], v)
    else:
        with output.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result['delivery_hashes'] = {output.name: sha(output)}
    path = DEST / 'report.json'
    if path.exists():
        assert read(path) == result
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'raw_timing_steps_checked', 'adaptive_arms_checked', 'timing_categories_checked', 'empty_categories', 'original_diagnosis_unchanged', 'test_accessed')}, indent=2))


if __name__ == '__main__':
    main()
