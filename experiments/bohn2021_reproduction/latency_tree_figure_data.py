"""Audited, all-seed inputs for post-campaign figures; never selects a policy.

Only run after BOTH native controllers finish. No production training/effect
module is imported. Confidence intervals are copied from independently reviewed
effects; plotted normalization is explicitly conditional on observed fixed cost.
"""
import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26'
REG = OUT / 'figure_source_registration.json'
TASKS = ('vehicle', 'pendulum')
BASE = {'vehicle': 25, 'pendulum': 30}
SOURCES = ('latency_tree_figure_data.py', 'latency_tree_figures.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def frozen_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            assert stream.read() == value, ('Different existing output; inspect', str(path))
    else:
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(value)


def frozen_json(path, value):
    frozen_text(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + '\n')


def csv_write(path, rows):
    assert rows
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader()
    writer.writerows(rows)
    frozen_text(path, buffer.getvalue())


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, ('Changed evidence', path)


def sources():
    return {str(Path(__file__).parent / name): sha(Path(__file__).parent / name) for name in SOURCES}


def idle(verify_sources=True):
    # Read status before loading numpy, pandas or matplotlib. Never wait in Python.
    for name in ('status.json', 'native_pipeline_v2_status.json', 'budget_finish_status.json'):
        value = read(OUT / name)
        assert value['complete'] and not value['active'], ('Campaign not finished', name)
    pipeline = read(OUT / 'native_pipeline_v2_status.json')
    assert pipeline['stage'] in ('negative_validation_test_sealed',
                                  'confirmation_review_complete_delivery_pending')
    assert read(OUT / 'budget_finish_status.json')['stage'] == 'budget_complete_final_delivery_pending'
    controllers = [read(OUT / name) for name in ('native_pipeline_v2_status.json', 'budget_finish_status.json')]
    pids = [value['pid'] for value in controllers]
    assert all(type(pid) is int and pid > 0 for pid in pids)
    ps = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
    assert ps.is_file(), 'Cannot verify native controller exit'
    query = ('$ErrorActionPreference="Stop"; Get-CimInstance Win32_Process -Filter "ProcessId=%d OR ProcessId=%d" | '
             'Where-Object { $_.CommandLine -match "latency_tree_(native_pipeline|budget_finish)\\.ps1" } | '
             'ForEach-Object { $_.ProcessId }') % tuple(pids)
    result = subprocess.run([str(ps), '-NoProfile', '-Command', query], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 0 and not result.stdout.strip(), 'Native controller has not exited or exit check failed'
    budget = read(OUT / 'final_budget/independent_review.json')
    assert budget['passed']
    verify(budget['hashes'])
    assert Path('/proc').is_dir(), 'Use the registered WSL audit interpreter'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        assert not any(Path(a).name in ('latency_tree_run.py', 'latency_tree_evaluate.py',
                                       'latency_tree_fixed_train.py') for a in args), args
    if verify_sources:
        registration = read(REG)
        assert registration['source_hashes'] == {
            str(Path(p).relative_to(ROOT)): digest for p, digest in sources().items()}
    return budget


def register():
    idle(verify_sources=False)
    frozen_json(REG, dict(
        source_hashes={str(Path(p).relative_to(ROOT)): digest for p, digest in sources().items()},
        timing_controllers_finished=True, changes_acceptance=False, unlocks_tests=False,
        checks_pending=True, actual_visual_review_pending=True, goal_complete=False,
        scope='Post-campaign all-case figures. Any source repair requires a preserved registration amendment.'))
    return dict(registered=True, sha256=sha(REG), python_checks_pending=True)


def canonical(task, family, seed, h):
    return task, ('primary' if family in ('grid', 'matched') and h == BASE[task] else family), seed, h


def reference(task, label, seed, nominations):
    if label in ('independent', 'matched'):
        h = nominations[task][label + '_h']
        return canonical(task, 'grid' if label == 'independent' else 'matched', seed, h)
    assert label in ('adaptive', 'certificate')
    return task, label, seed, 0


def expected_jobs(split, nominations):
    jobs = set()
    for task in TASKS:
        for seed in range(3):
            jobs.add((task, 'adaptive', seed, 0))
            for label in ('independent', 'matched'):
                jobs.add(reference(task, label, seed, nominations))
            if task == 'pendulum':
                jobs.update((task, family, seed, 0) for family in ('certificate', 'combined'))
            if split == 'validation':
                jobs.update(canonical(task, 'matched', seed, h) for h in range(5, 51, 5))
        if split == 'validation':
            jobs.update(canonical(task, 'grid', 0, h) for h in range(5, 51, 5))
    return jobs


def interval(value, percent=False):
    fields = ('mean_ratio', 'lower', 'upper') if percent else ('mean', 'lower', 'upper')
    result = [float(value[k]) for k in fields]
    assert all(math.isfinite(v) for v in result) and result[1] <= result[2]
    # Percentile intervals need not contain the point estimate.
    return [(v - 1.) * 100 for v in result] if percent else result


def relative_cost(effect, fixed_mean):
    values = interval(effect['cost_difference'])
    if fixed_mean == 0:
        assert effect['relative_cost_difference'] is None
        return [None, None, None]
    scale = 100. / abs(fixed_mean)
    expected = values[0] / abs(fixed_mean)
    assert math.isclose(expected, effect['relative_cost_difference'], rel_tol=1e-10, abs_tol=1e-10)
    return [v * scale for v in values]


def summarize(rows, count):
    assert len(rows) == count and [r['case'] for r in rows] == list(range(count))
    counters = ('steps', 'success', 'constraint', 'initial_failed_steps',
                'solver_failure_steps', 'switches', 'retries')
    for row in rows:
        assert all(type(row[k]) in (int, bool) and row[k] >= 0 for k in counters)
        assert row['steps'] > 0 and row['success'] in (0, 1) and row['constraint'] in (0, 1)
        assert row['initial_failed_steps'] <= row['steps'] and row['solver_failure_steps'] <= row['steps']
        assert row['switches'] <= max(0, row['steps'] - 1) and 5 <= row['mean_horizon'] <= 50
        assert all(math.isfinite(row[k]) for k in ('total_cost', 'physical_constraint_cost', 'mean_horizon'))
    result = {key: sum(int(row[key]) for row in rows) for key in counters}
    result.update({key: math.fsum(row[key] for row in rows) / count
                   for key in ('total_cost', 'physical_constraint_cost')})
    return result


def build_row(effect, adaptive, fixed, n, method):
    a, b = summarize(adaptive, n), summarize(fixed, n)
    delta = math.fsum(x['total_cost'] - y['total_cost'] for x, y in zip(adaptive, fixed)) / n
    assert math.isclose(delta, effect['cost_difference']['mean'], rel_tol=1e-10, abs_tol=1e-10)
    cost = relative_cost(effect, b['total_cost'])
    tm = effect['timing']
    time = interval(tm, percent=True)
    assert len(tm['repeat_ratios']) == 2 and all(math.isfinite(v) and v > 0 for v in tm['repeat_ratios'])
    row = dict(task=effect['task'], comparator=effect['comparator'], seed=effect['seed'],
               method=method, n_cases=n, main_gate=method == 'adaptive',
               cost_mean_delta=delta, fixed_mean_cost=b['total_cost'],
               cost_mean_pct=cost[0], cost_lower_pct=cost[1], cost_upper_pct=cost[2],
               time_mean_pct=time[0], time_lower_pct=time[1], time_upper_pct=time[2],
               time_repeat0_pct=100 * (tm['repeat_ratios'][0] - 1),
               time_repeat1_pct=100 * (tm['repeat_ratios'][1] - 1),
               success_loss_pp=100 * (b['success'] - a['success']) / n,
               constraint_increase_pp=100 * (a['constraint'] - b['constraint']) / n,
               initial_failure_increase_pp=100 * (a['initial_failed_steps'] / a['steps'] -
                                                 b['initial_failed_steps'] / b['steps']),
               final_failure_increase_pp=100 * (a['solver_failure_steps'] / a['steps'] -
                                               b['solver_failure_steps'] / b['steps']))
    for key in ('safe', 'control_noninferior', 'total_noninferior', 'adapted', 'cost_gain', 'time_gain'):
        assert type(effect[key]) is bool
        row[key] = effect[key]
    row.update({'adaptive_' + k: v for k, v in a.items()})
    row.update({'fixed_' + k: v for k, v in b.items()})
    denominator = abs(b['physical_constraint_cost'])
    row['physical_cost_change_pct'] = (100 * (a['physical_constraint_cost'] - b['physical_constraint_cost']) /
                                       denominator if denominator else None)
    return row


def collect(split):
    assert split in ('validation', 'test')
    idle()
    hashes = {str(REG): sha(REG), **sources()}

    def record(path):
        path = Path(path)
        value = read(path)
        hashes[str(path)] = sha(path)
        return value

    for name in ('status.json', 'native_pipeline_v2_status.json', 'budget_finish_status.json',
                 'final_budget/independent_review.json'):
        record(OUT / name)
    validation_review = record(OUT / 'validation_delivery/independent_review.json')
    assert validation_review['passed']
    if split == 'test':
        assert validation_review['effect_passed']
        confirmation = record(OUT / 'confirmation_registration.json')
        assert confirmation['validation_gate_passed'] and confirmation['independent_review_passed']
        verify(confirmation['hashes'])
    elif not validation_review['effect_passed']:
        assert not (OUT / 'confirmation_registration.json').exists()
        assert not (OUT / 'evaluations/test').exists() and not (OUT / 'timing_test').exists()
    dest = OUT / (split + '_delivery')
    gate_path, review_path = dest / 'effect_gate.json', dest / 'independent_review.json'
    gate, review = record(gate_path), record(review_path)
    assert gate['passed_calculation'] and gate['timing_audited'] and gate['raw_records_audited']
    assert review['passed'] and review['split'] == gate['split'] == split
    assert review['effect_passed'] == gate['effect_passed']
    assert review['hashes'][str(gate_path)] == sha(gate_path)
    verify(review['hashes'])
    verify(gate['hashes'])
    assert review['comparisons_checked'] == 12 and review['diagnostic_comparisons_checked'] == 21
    assert len(gate['comparisons']) == 12 and len(gate['diagnostics']) == 21
    nominations = record(OUT / 'baseline_selection.json')['nominations']
    n = 64 if split == 'validation' else 128
    arms, episodes = {}, []
    for job in sorted(expected_jobs(split, nominations)):
        task, family, seed, h = job
        folder = OUT / 'evaluations' / split / task / ('%s_h%d_s%d' % (family, h, seed))
        receipt_path = folder / 'completed.json'
        assert gate['hashes'][str(receipt_path)] == sha(receipt_path)
        receipt = record(receipt_path)
        assert receipt['passed']
        verify(receipt['hashes'])
        assert receipt['hashes'][str(folder / 'summary.json')] == sha(folder / 'summary.json')
        summary = record(folder / 'summary.json')
        assert (summary['task'], summary['family'], summary['seed'], summary['h'], summary['split']) == (*job, split)
        assert summary['repeats'] == 1 and all(r['repeat'] == 0 for r in summary['episodes'])
        summarize(summary['episodes'], n)
        arms[job] = summary['episodes']
        episodes.extend(dict(task=task, method=family, seed=seed, h=h, **r) for r in summary['episodes'])
    comparisons, pairs, timing = [], [], []
    expected_main = {(t, c, s) for t in TASKS for c in ('independent', 'matched') for s in range(3)}
    assert {(e['task'], e['comparator'], e['seed']) for e in gate['comparisons']} == expected_main
    expected_diag = {(m, c, s) for m in ('certificate', 'combined') for s in range(3)
                     for c in (('adaptive', 'independent', 'matched') + (('certificate',) if m == 'combined' else ()))}
    assert {(e['family'], e['comparator'], e['seed']) for e in gate['diagnostics']} == expected_diag
    used_jobs = set()
    for effect in gate['comparisons'] + gate['diagnostics']:
        method = effect.get('family', 'adaptive')
        task, seed, label = effect['task'], effect['seed'], effect['comparator']
        aj, bj = (task, method, seed, 0), reference(task, label, seed, nominations)
        if method != 'adaptive':
            assert effect['contributes_to_main_gate'] is False
        comparisons.append(build_row(effect, arms[aj], arms[bj], n, method))
        used_jobs.update((aj, bj))
        if method == 'adaptive':
            for a, b in zip(arms[aj], arms[bj]):
                pairs.append(dict(task=task, comparator=label, seed=seed, case=a['case'],
                                  cost_difference=a['total_cost'] - b['total_cost'],
                                  adaptive_cost=a['total_cost'], fixed_cost=b['total_cost']))
    for task, family, seed, h in sorted(used_jobs):
        for repeat in range(2):
            folder = OUT / ('timing_' + split) / ('r%d_%s_%s_h%d_s%d' % (repeat, task, family, h, seed))
            p = folder / 'completed.json'
            assert gate['hashes'][str(p)] == sha(p)
            receipt = record(p)
            assert receipt['passed'] and receipt['exact_replay']
            verify(receipt['hashes'])
            sp = folder / 'summary.json'
            assert receipt['hashes'][str(sp)] == sha(sp)
            timed = record(sp)['episodes']
            scored = arms[(task, family, seed, h)]
            assert len(timed) == n and [r['case'] for r in timed] == list(range(n))
            assert [r['steps'] for r in timed] == [r['steps'] for r in scored]
            timing.extend(dict(task=task, method=family, seed=seed, h=h, timing_repeat=repeat,
                               case=r['case'], steps=r['steps'], decision_total_s=r['decision_total_s']) for r in timed)
    for row in comparisons:
        aj = row['task'], row['method'], row['seed'], 0
        bj = reference(row['task'], row['comparator'], row['seed'], nominations)
        averages = []
        for key in (aj, bj):
            repeats = []
            for repeat in range(2):
                selected = [r for r in timing if (r['task'], r['method'], r['seed'], r['h']) == key
                            and r['timing_repeat'] == repeat]
                assert len(selected) == n
                assert all(math.isfinite(r['decision_total_s']) and r['decision_total_s'] > 0 for r in selected)
                repeats.append(math.fsum(r['decision_total_s'] for r in selected) / sum(r['steps'] for r in selected))
            averages.append(repeats)
        for repeat in range(2):
            observed = 100 * (averages[0][repeat] / averages[1][repeat] - 1)
            assert math.isclose(observed, row['time_repeat%d_pct' % repeat], rel_tol=1e-9, abs_tol=1e-9)
        observed = 100 * (math.fsum(averages[0]) / math.fsum(averages[1]) - 1)
        assert math.isclose(observed, row['time_mean_pct'], rel_tol=1e-9, abs_tol=1e-9)
    assert len(comparisons) == 33 and len(pairs) == 12 * n
    return dict(split=split, n_cases=n, effect_passed=gate['effect_passed'], effects=gate['effects'], nominations=nominations,
                comparisons=comparisons, pairs=pairs, episodes=episodes, timing=timing, hashes=hashes)


def collect_training():
    idle()
    path = OUT / 'training_delivery'
    manifest, review = read(path / 'manifest.json'), read(path / 'independent_review.json')
    assert manifest['passed'] and manifest['training_only'] and review['passed']
    assert review['candidates'] == manifest['candidate_rows'] == 288
    verify(review['hashes'])
    assert review['hashes'][str(path / 'manifest.json')] == sha(path / 'manifest.json')
    verify(manifest['hashes'])
    with (path / 'all_candidates.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(t, s, 'g%d_c%02d' % (g, c)) for t in TASKS for s in range(3)
                for g in range(4) for c in range(12)}
    assert len(rows) == 288 and {(r['task'], int(r['seed']), r['candidate']) for r in rows} == expected
    values = []
    for row in rows:
        value = dict(task=row['task'], seed=int(row['seed']), candidate=row['candidate'],
                     generation=int(row['generation']), eligible=row['eligible'] == 'True',
                     selected=row['selected_final'] == 'True', cost_change=float(row['cost_change']),
                     time_ratio=float(row['time_ratio']))
        assert row['eligible'] in ('True', 'False') and row['selected_final'] in ('True', 'False')
        assert math.isfinite(value['cost_change']) and math.isfinite(value['time_ratio'])
        values.append(value)
    policies = read(path / 'selected_policies.json')['policies']
    assert len(policies) == 6
    files = ('manifest.json', 'independent_review.json', 'all_candidates.csv', 'selected_policies.json')
    return dict(candidates=values, policies=policies,
                hashes={str(path / name): sha(path / name) for name in files})


def check():
    """Pure known answers; run only after measured work, alongside rendering checks."""
    idle()
    counts = []
    for vh in range(5, 51, 5):
        for ph in range(5, 51, 5):
            nominations = {'vehicle': {'independent_h': vh, 'matched_h': 25},
                           'pendulum': {'independent_h': ph, 'matched_h': 30}}
            jobs = expected_jobs('validation', nominations)
            assert len(jobs) == 90 + 2 * (vh != 25) + 2 * (ph != 30)
            assert expected_jobs('test', nominations) <= jobs
            counts.append(len(jobs))
    effect = dict(cost_difference=dict(mean=-2., lower=-3., upper=-1.), relative_cost_difference=-.02)
    assert relative_cost(effect, 100) == [-2., -3., -1.]
    assert relative_cost(effect, -100) == [-2., -3., -1.]
    assert relative_cost(dict(effect, relative_cost_difference=None), 0) == [None] * 3
    assert interval(dict(mean_ratio=.8, lower=.7, upper=.9), True)[0] < -19.99
    # Do not force percentile intervals to enclose their point estimate.
    assert interval(dict(mean=0., lower=1., upper=2.)) == [0., 1., 2.]
    rejected = 0
    for v in (dict(mean=0., lower=2., upper=1.), dict(mean=float('nan'), lower=0., upper=1.)):
        try:
            interval(v)
        except AssertionError:
            rejected += 1
    assert rejected == 2
    return dict(passed=True, grid_combinations=len(counts), zero_denominator_checked=True,
                percentile_outside_estimate_checked=True, simulations=0, goal_complete=False)
