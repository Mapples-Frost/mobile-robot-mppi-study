"""Post-campaign Chinese report and requirement matrix, never a goal completer.

Run only when experiment and budget controllers have finished and exited.
Environment capture describes the installed post-run state, not an independently
recreated environment. Final prose/package review is deliberately still required.
"""
import argparse
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from latency_tree_figure_data import ROOT, OUT, TASKS, idle, read, sha, verify, frozen_json, frozen_text, csv_write

ART = OUT.parents[1]
SCRIPTS = Path(__file__).resolve().parent
REG = OUT / 'final_report_registration.json'
SOURCES = ('latency_tree_final_report.py', 'latency_tree_final_report_review.py', 'latency_tree_figure_data.py')
BASE = {'vehicle': 25, 'pendulum': 30}


def source_hashes():
    return {str(SCRIPTS / name): sha(SCRIPTS / name) for name in SOURCES}


def register():
    idle(verify_sources=False)
    frozen_json(REG, dict(source_hashes=source_hashes(), simulations=0,
                          effects_redefined=False, unlocks_tests=False,
                          checks_pending=True, final_review_pending=True, goal_complete=False))
    return dict(registered=True, sha256=sha(REG), checks_pending=True)


def ready():
    idle(verify_sources=False)
    assert read(REG)['source_hashes'] == source_hashes()


def safe(a, b):
    assert a['steps'] > 0 and b['steps'] > 0
    return (a['success'] >= b['success'] and a['constraint'] <= b['constraint'] and
            all(Fraction(a[k], a['steps']) <= Fraction(b[k], b['steps'])
                for k in ('initial_failed_steps', 'solver_failure_steps')))


def logic(row):
    a, b, timing = row['adaptive'], row['fixed'], row['timing']
    for item in (a, b):
        assert all(type(item[k]) is int and item[k] >= 0 for k in (
            'steps', 'success', 'constraint', 'initial_failed_steps', 'solver_failure_steps', 'switches'))
        assert all(math.isfinite(item[k]) for k in ('total_cost', 'physical_constraint_cost'))
    assert all(math.isfinite(row['cost_difference'][k]) for k in ('mean', 'lower', 'upper'))
    assert row['cost_difference']['lower'] <= row['cost_difference']['upper']
    assert math.isclose(row['cost_difference']['mean'], a['total_cost'] - b['total_cost'], rel_tol=1e-10, abs_tol=1e-10)
    relative = (row['cost_difference']['mean'] / abs(b['total_cost'])) if b['total_cost'] else None
    if relative is None:
        assert row['relative_cost_difference'] is None
    else:
        assert math.isclose(relative, row['relative_cost_difference'], rel_tol=1e-10, abs_tol=1e-12)
    assert timing is not None and len(timing['repeat_ratios']) == 2
    assert all(math.isfinite(timing[k]) and timing[k] > 0 for k in ('mean_ratio', 'lower', 'upper'))
    assert all(math.isfinite(v) and v > 0 for v in timing['repeat_ratios'])
    assert timing['lower'] <= timing['upper']
    physical = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(b['physical_constraint_cost'])
    total = a['total_cost'] <= b['total_cost'] + .02 * abs(b['total_cost'])
    return dict(safe=safe(a, b), control_noninferior=physical, total_noninferior=total,
                adapted=a['switches'] > 0,
                cost_gain=relative is not None and relative <= -.03 and row['cost_difference']['upper'] < 0,
                time_gain=timing['mean_ratio'] <= .9 and timing['upper'] < 1 and
                          all(v < 1 for v in timing['repeat_ratios']) and total)


def check():
    ready()
    b = dict(steps=100, success=10, constraint=0, initial_failed_steps=2,
             solver_failure_steps=1, switches=0, total_cost=100., physical_constraint_cost=100.)
    row = dict(adaptive=dict(b, total_cost=97., switches=1), fixed=b,
               cost_difference=dict(mean=-3., lower=-4., upper=-2.), relative_cost_difference=-.03,
               timing=dict(mean_ratio=.9, lower=.85, upper=.95, repeat_ratios=[.89, .91]))
    answer = logic(row)
    assert all(answer.values())
    cases = 1
    for field, value, flag in [('success', 9, 'safe'), ('constraint', 1, 'safe'),
                               ('initial_failed_steps', 3, 'safe'), ('solver_failure_steps', 2, 'safe'),
                               ('switches', 0, 'adapted'), ('physical_constraint_cost', 103., 'control_noninferior')]:
        assert not logic(dict(row, adaptive=dict(row['adaptive'], **{field: value})))[flag]
        cases += 1
    equal_rate = dict(b, steps=200, initial_failed_steps=4, solver_failure_steps=2)
    assert safe(equal_rate, b)
    assert not safe(dict(equal_rate, initial_failed_steps=5), b)
    cases += 2
    zero = dict(row, fixed=dict(b, total_cost=0.), adaptive=dict(row['adaptive'], total_cost=0.),
                cost_difference=dict(mean=0., lower=0., upper=0.), relative_cost_difference=None)
    assert logic(zero)['time_gain'] and not logic(zero)['cost_gain']
    assert not logic(dict(row, timing=dict(row['timing'], upper=1.)))['time_gain']
    assert not logic(dict(row, timing=dict(row['timing'], repeat_ratios=[.8, 1.])))['time_gain']
    cases += 3
    result = dict(passed=True, checks=cases, scope='Independent Boolean logic and exact safety rates; no bootstrap rerun.',
                  source_hashes=source_hashes(), simulations=0, test_accessed=False, goal_complete=False)
    frozen_json(OUT / 'final_report_checks.json', result)
    return result


CAPTURE_CODE = '''import hashlib,json,platform,subprocess,sys
from pathlib import Path
exe=Path(sys.executable).resolve()
packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],universal_newlines=True))
print(json.dumps(dict(executable=sys.executable,resolved_executable=str(exe),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),python=sys.version,platform=platform.platform(),packages=packages)))
'''


def environment():
    ready()
    dest = OUT / 'delivery_environment'
    if dest.exists():
        record = read(dest / 'snapshot.json')
        verify(record['hashes'])
        assert record['source_hashes'] == source_hashes()
        return dict(reused=True, fresh_capture=False, captured_after_experiment=True)
    dest.mkdir()
    captures, hashes = {}, {}
    for label, executable in (('legacy', '/home/mapples/.local/share/bohn2021-python37/bin/python'),
                              ('audit', str(ROOT / '.venv/bin/python'))):
        result = subprocess.run([executable, '-c', CAPTURE_CODE], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding='utf-8', errors='replace', timeout=120)
        frozen_text(dest / (label + '_stdout.txt'), result.stdout)
        frozen_text(dest / (label + '_stderr.txt'), result.stderr)
        assert result.returncode == 0, ('Environment capture failed; preserve logs', label, result.returncode)
        captured = json.loads(result.stdout)
        assert captured['packages'] and captured['python']
        frozen_json(dest / (label + '.json'), captured)
        captures[label] = captured
        for suffix in ('.json', '_stdout.txt', '_stderr.txt'):
            p = dest / (label + suffix)
            hashes[str(p)] = sha(p)
    revisions = {}
    for name in ('gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon'):
        folder = ART / 'sources' / name
        commands = {'head': ['rev-parse', 'HEAD'], 'tracked_status': ['status', '--porcelain', '--untracked-files=no']}
        revisions[name] = {}
        for field, args in commands.items():
            result = subprocess.run(['git', '-C', str(folder)] + args, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, encoding='utf-8', timeout=60)
            assert result.returncode == 0, (name, field, result.stderr)
            revisions[name][field] = result.stdout.strip()
    pinned = {}
    requirement_path = SCRIPTS / 'requirements-legacy.txt'
    installed = {p['name'].lower().replace('_', '-'): p['version'] for p in captures['legacy']['packages']}
    for line in requirement_path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, version = line.split('==')
            pinned[name] = dict(expected=version, installed=installed.get(name.lower().replace('_', '-')))
    files = [requirement_path, SCRIPTS / 'setup.sh', ART / 'sources/runtime-freeze.txt',
             ART / 'sources/archives/manifest.json', ART / 'sources/provenance.json']
    for p in files:
        hashes[str(p)] = sha(p)
    for name, source in (('cpuinfo.txt', '/proc/cpuinfo'), ('os_release.txt', '/etc/os-release')):
        frozen_text(dest / name, Path(source).read_text())
        hashes[str(dest / name)] = sha(dest / name)
    archive_commits = {r['repo']: r['commit'] for r in read(ART / 'sources/archives/manifest.json')}
    source_mismatches = [name for name, value in revisions.items()
                         if value['head'] != archive_commits[name] or value['tracked_status']]
    pin_mismatches = [name for name, value in pinned.items() if value['expected'] != value['installed']]
    record = dict(captured_after_experiment=True, reconstruction_tested=False,
                  author_sources=revisions, legacy_pin_comparison=pinned,
                  legacy_pins_match=not pin_mismatches, legacy_pin_mismatches=pin_mismatches,
                  author_archive_commits_match=not source_mismatches,
                  author_source_mismatches=source_mismatches,
                  source_hashes=source_hashes(), hashes=hashes,
                  scope='Post-run package and author-checkout inventory. Per-condition environment records remain '
                        'the contemporaneous evidence. Historical pip freeze contains local build URLs; not a portable lockfile.',
                  simulations=0, test_accessed=False, goal_complete=False)
    frozen_json(dest / 'snapshot.json', record)
    return {k: v for k, v in record.items() if k not in ('hashes', 'source_hashes', 'author_sources', 'legacy_pin_comparison')}


def verify_figures(split, folder, remember):
    assert folder.parent.resolve() == (OUT / (split + '_figures')).resolve()
    manifest = remember(folder / 'export_manifest.json')
    assert manifest['split'] == split and manifest['visual_review_passed']
    for name in ('source_hashes', 'skill_hashes', 'hashes'):
        verify(manifest[name])
    expected = {'cost_intervals', 'latency_intervals', 'physical_cost', 'vehicle_paired_costs',
                'pendulum_paired_costs', 'vehicle_safety', 'pendulum_safety', 'horizon_switches',
                'mean_horizon', 'c5_diagnostics', 'combined_diagnostics', 'training_candidates'}
    if split == 'validation':
        expected.update(t + '_fixed_grid' for t in TASKS)
    assert set(manifest['figures']) == expected
    root = folder.parent
    assert manifest['data_manifest_sha256'] == sha(root / 'data_manifest.json')
    assert manifest['profile_review_sha256'] == sha(root / 'profile_review.json')
    data = remember(root / 'data_manifest.json')
    profile = remember(root / 'profile_review.json')
    assert profile['passed'] and profile['all_cases_and_seeds_retained'] and profile['no_outlier_exclusions']
    assert profile['data_manifest_sha256'] == sha(root / 'data_manifest.json')
    for field in ('source_hashes', 'skill_hashes', 'input_hashes', 'output_hashes'):
        verify(data[field])
    preview = root / ('preview_%02d' % manifest['revision'])
    visual = remember(preview / 'visual_review.json')
    assert manifest['visual_review_sha256'] == sha(preview / 'visual_review.json')
    assert visual['passed'] and visual['preview_manifest_sha256'] == sha(preview / 'preview_manifest.json')
    preview_manifest = remember(preview / 'preview_manifest.json')
    verify(preview_manifest['hashes'])
    assert preview_manifest['data_manifest_sha256'] == sha(root / 'data_manifest.json')
    assert set(visual['figures']) == expected
    for row in visual['figures'].values():
        assert all(row[k] is True for k in ('glyphs', 'clipping', 'overlap', 'panel_alignment',
                                           'panel_spacing', 'color_and_grayscale', 'all_data_visible', 'cross_panel_consistency'))
    for entry in manifest['figures'].values():
        assert {Path(p).suffix for p in entry['files']} == {'.png', '.pdf', '.svg'}
        assert len(entry['files']) == 4
    return manifest


def number(value):
    return '未定义/未知' if value is None else ('%.6f' % value)


def build(snapshot, validation_figures, test_figures):
    ready()
    assert snapshot > 0
    dest = OUT / ('final_delivery_%02d' % snapshot)
    assert not dest.exists(), 'Preserve existing delivery; use a new snapshot after inspecting any failure'
    hashes = {str(REG): sha(REG), **source_hashes()}

    def remember(path, passed=False):
        path = Path(path)
        value = read(path)
        if passed:
            assert value['passed'], str(path)
        hashes[str(path)] = sha(path)
        return value

    checks = remember(OUT / 'final_report_checks.json', True)
    assert checks['source_hashes'] == source_hashes()
    review_checks = remember(OUT / 'final_report_review_checks.json', True)
    assert review_checks['source_hashes'] == source_hashes()
    env = remember(OUT / 'delivery_environment/snapshot.json')
    verify(env['hashes'])
    assert env['source_hashes'] == source_hashes()
    training = remember(OUT / 'training_delivery/manifest.json', True)
    training_review = remember(OUT / 'training_delivery/independent_review.json', True)
    verify(training['hashes']); verify(training_review['hashes'])
    assert training['candidate_rows'] == training_review['candidates'] == 288
    policies = remember(OUT / 'training_delivery/selected_policies.json')['policies']
    assert {(p['task'], p['seed']) for p in policies} == {(t, s) for t in TASKS for s in range(3)}
    assert len(policies) == 6
    baseline = remember(OUT / 'baseline_completion.json', True)
    selection = remember(OUT / 'baseline_selection.json')
    verify(baseline['hashes']); verify(selection['hashes'])
    evaluation = remember(OUT / 'evaluation_registration.json')
    verify(evaluation['hashes'])
    models = []
    for task in TASKS:
        nomination = selection['nominations'][task]
        assert sorted(r['h'] for r in nomination['independent_grid']) == list(range(5, 51, 5))
        assert sorted(map(int, nomination['matched_grid'])) == list(range(5, 51, 5))
        for h, entries in nomination['matched_grid'].items():
            assert len(entries) == 3 and sorted(r['seed'] for r in entries) == [0, 1, 2]
            assert all(r['h'] == int(h) and r['task'] == task for r in entries)
        for label in ('independent', 'matched'):
            for seed in range(3):
                evaluation_h = nomination[label + '_h']
                trained_h = evaluation_h if label == 'independent' else BASE[task]
                key = '%s_h%d_s%d' % (task, trained_h, seed)
                folder = Path(evaluation['model_inventory'][key]) if evaluation['model_inventory'][key] else OUT / 'extra_fixed' / key
                model_manifest, completed = remember(folder / 'manifest.json'), remember(folder / 'completed.json')
                assert model_manifest['steps'] == completed['steps'] == 15000 and completed['status'] == 'complete'
                assert model_manifest['task'] == task and model_manifest['seed'] == seed
                assert model_manifest['fixed_horizon'] == trained_h
                model_path = folder / 'model.zip'
                hashes[str(model_path)] = sha(model_path)
                models.append(dict(task=task, comparator=label, seed=seed, evaluation_h=evaluation_h,
                                   trained_h=trained_h, training_steps=15000, folder=str(folder),
                                   model_sha256=hashes[str(model_path)], new_model=OUT in folder.parents))
    ledger = remember(OUT / 'final_budget/ledger.json')
    budget_review = remember(OUT / 'final_budget/independent_review.json', True)
    verify(ledger['hashes']); verify(ledger['source_hashes']); verify(budget_review['hashes'])
    assert ledger['passed_accounting'] and ledger['complete_budget'] == budget_review['complete_budget']
    assert budget_review['hashes'][str(OUT / 'final_budget/ledger.json')] == sha(OUT / 'final_budget/ledger.json')
    split_audit = remember(OUT / 'split_audit.json', True)
    assert split_audit['current_cases'] == 556 and not split_audit['overlaps'] and not split_audit['unsupported']
    for name in ('status.json', 'native_pipeline_v2_status.json', 'budget_finish_status.json',
                 'registration.json', 'evaluation_registration.json', 'fitted_policy_registration.json',
                 'synthetic_checks.json', 'learning_checks.json', 'audit_smoke.json', 'audit_train.json',
                 'learning_audit.json', 'evaluation_checks.json', 'audit_evaluation_smoke.json'):
        remember(OUT / name)
    assessments, rows, diagnostics = {}, [], []

    def effect(split):
        gate_path = OUT / (split + '_delivery/effect_gate.json')
        review_path = OUT / (split + '_delivery/independent_review.json')
        gate, review = remember(gate_path), remember(review_path, True)
        assert gate['split'] == review['split'] == split
        assert gate['passed_calculation'] and gate['timing_audited'] and gate['raw_records_audited']
        assert review['hashes'][str(gate_path)] == sha(gate_path)
        verify(gate['hashes']); verify(review['hashes'])
        assert review['comparisons_checked'] == 12 and review['diagnostic_comparisons_checked'] == 21
        assert len(gate['comparisons']) == 12 and len(gate['diagnostics']) == 21
        expected_diagnostics = {(method, label, seed) for method in ('certificate', 'combined')
            for seed in range(3) for label in (('adaptive', 'independent', 'matched') +
                                              (('certificate',) if method == 'combined' else ()))}
        assert {(r['family'], r['comparator'], r['seed']) for r in gate['diagnostics']} == expected_diagnostics
        for row in gate['diagnostics']:
            assert row['task'] == 'pendulum' and row['contributes_to_main_gate'] is False
            diagnostics.append(dict(split=split, method=row['family'], comparator=row['comparator'],
                seed=row['seed'], contributes_to_main_gate=False,
                cost_percent=None if row['relative_cost_difference'] is None else 100 * row['relative_cost_difference'],
                cost_delta_lower=row['cost_difference']['lower'], cost_delta_upper=row['cost_difference']['upper'],
                latency_percent=100 * (row['timing']['mean_ratio'] - 1),
                latency_ratio_lower=row['timing']['lower'], latency_ratio_upper=row['timing']['upper'],
                repeat0_ratio=row['timing']['repeat_ratios'][0], repeat1_ratio=row['timing']['repeat_ratios'][1],
                **{k: row[k] for k in ('safe', 'control_noninferior', 'total_noninferior', 'adapted', 'cost_gain', 'time_gain')}))
        expected = {(t, label, s) for t in TASKS for label in ('independent', 'matched') for s in range(3)}
        assert {(r['task'], r['comparator'], r['seed']) for r in gate['comparisons']} == expected
        for row in gate['comparisons']:
            computed = logic(row)
            assert all(type(row[k]) is bool and row[k] == v for k, v in computed.items())
            failures = [k for k in ('safe', 'control_noninferior', 'adapted') if not computed[k]]
            if not (computed['cost_gain'] or computed['time_gain']):
                failures.append('neither_cost_nor_time_route')
            rows.append(dict(split=split, task=row['task'], comparator=row['comparator'], seed=row['seed'],
                             fixed_h=row['fixed_h'], failures=';'.join(failures),
                             cost_percent=None if row['relative_cost_difference'] is None else 100 * row['relative_cost_difference'],
                             cost_delta_upper=row['cost_difference']['upper'],
                             latency_percent=100 * (row['timing']['mean_ratio'] - 1),
                             latency_ratio_upper=row['timing']['upper'],
                             repeat0_ratio=row['timing']['repeat_ratios'][0], repeat1_ratio=row['timing']['repeat_ratios'][1],
                             **computed))
        tasks = {}
        for task in TASKS:
            selected = [r for r in gate['comparisons'] if r['task'] == task]
            common_cost = all(r['cost_gain'] for r in selected)
            common_time = all(r['time_gain'] for r in selected)
            passed = all(r['safe'] and r['control_noninferior'] and r['adapted'] for r in selected) and (common_cost or common_time)
            assert gate['effects'][task]['passed'] == review['effects'][task]['passed'] == passed
            assert gate['effects'][task]['common_cost_route'] == common_cost
            assert gate['effects'][task]['common_time_route'] == common_time
            tasks[task] = dict(passed=passed, common_cost_route=common_cost, common_time_route=common_time)
        passed = all(t['passed'] for t in tasks.values())
        assert gate['effect_passed'] == review['effect_passed'] == passed
        assessments[split] = dict(passed=passed, tasks=tasks, gate=gate, review=review)
        return passed

    positive_validation = effect('validation')
    figure_results = {'validation': verify_figures('validation', validation_figures, remember)}
    positive_confirmation = False
    if positive_validation:
        confirmation = remember(OUT / 'confirmation_registration.json')
        assert confirmation['validation_gate_passed'] and confirmation['independent_review_passed']
        verify(confirmation['hashes'])
        positive_confirmation = effect('test')
        assert test_figures is not None, 'Completed confirmation needs its full figures'
        figure_results['test'] = verify_figures('test', test_figures, remember)
    else:
        assert test_figures is None
        assert not any((OUT / p).exists() for p in ('confirmation_registration.json', 'evaluations/test', 'timing_test', 'test_delivery'))
    assert ledger['test_phase_included'] == positive_validation
    for split, value in figure_results.items():
        assert value['effect_passed'] == assessments[split]['passed']
    archive_manifest = ART / 'sources/archives/manifest.json'
    for archive in remember(archive_manifest):
        path = archive_manifest.parent / archive['file']
        assert sha(path) == archive['sha256']
        hashes[str(path)] = archive['sha256']
    historical = [ART / 'report/paper_exact_grid_2026-09-23/report.md',
                  ART / 'sources/provenance.json', SCRIPTS / 'SOURCE_MAP.md', SCRIPTS / 'README.md',
                  ROOT / 'docs/reports/bohn2021_evidence_map_2026-09-24.md',
                  ROOT / 'docs/reports/bohn2021_gated_validation_failure_2026-09-26.md',
                  ROOT / 'docs/reports/bohn2021_failure_state_probe_2026-09-26.md',
                  ROOT / 'docs/reports/bohn2021_latency_tree_requirements_2026-09-26.md',
                  ROOT / 'docs/protocols/bohn2021_latency_tree_2026-09-26.md']
    for path in historical:
        assert path.is_file()
        hashes[str(path)] = sha(path)
    old_gate = remember(ART / 'results/gated_horizon_search_2026-09-25/validation_delivery/effect_gate.json')
    assert old_gate['validation_effect_passed'] is False
    diagnostic = remember(ART / 'results/failure_state_probe_2026-09-26/package_check.json', True)
    diagnostic_manifest = ART / 'results/failure_state_probe_2026-09-26/package_manifest.json'
    assert diagnostic['diagnostic_only'] and diagnostic['goal_complete'] is False
    assert diagnostic['manifest_sha256'] == sha(diagnostic_manifest)
    hashes[str(diagnostic_manifest)] = sha(diagnostic_manifest)
    requirements = [
        dict(id='all_tasks_seeds', status='verified_for_this_stage', evidence='training_delivery/manifest.json; independent_review.json', limitation='No inference to a population of training seeds.'),
        dict(id='original_vs_extension', status='documented_with_limits', evidence='Original paper-grid report; current protocol', limitation='Original missing experiment configuration not recovered; historical audit not rerun here.'),
        dict(id='strong_fixed_baselines', status='verified_for_this_stage', evidence='baseline_selection.json; baseline_completion.json; independent effect review', limitation='Independent full grid uses seed0; nominated independent H evaluated at all3 seeds.'),
        dict(id='split_integrity', status='verified_with_limits', evidence='split_audit.json; confirmation registration or absence', limitation='Exact input deduplication is not statistical-independence proof.'),
        dict(id='actual_timing', status='verified_for_this_stage', evidence='Two-repeat timing audit and independent effect review', limitation='WSL scheduling, measured logging subtraction, and excluded reset/simulator time disclosed.'),
        dict(id='complete_recorded_budget', status='verified_with_limits' if ledger['complete_budget'] else 'incomplete', evidence='final_budget/ledger.json; independent_review.json', limitation='Uninstrumented constructor/fixed-training NLP calls remain unknown.'),
        dict(id='per_condition_wall_cpu', status='not_fully_implemented', evidence='training_delivery/budget.json; final_budget/ledger.json', limitation='Protocol requested per-condition wall/CPU; only whole-phase wall and local decision times recorded.'),
        dict(id='validation_effect', status='passed' if positive_validation else 'failed', evidence='validation_delivery/effect_gate.json; independent_review.json', limitation='Validation selects baselines and cannot replace independent confirmation.'),
        dict(id='independent_confirmation', status=('passed' if positive_confirmation else 'failed') if positive_validation else 'not_run_sealed', evidence='test_delivery or verified absent', limitation='Same acceptance criteria; negative validation cannot unlock tests.'),
        dict(id='all_figures', status='verified_artifacts_visual_review_recorded', evidence='Explicit final figure manifests', limitation='This report verifies recorded visual checks; final report reviewer must actually inspect artifacts.'),
        dict(id='environment_reconstruction', status='inventory_captured_recreation_unverified', evidence='delivery_environment/snapshot.json; pinned author archives', limitation='Post-run capture is not proof of a fresh-environment reproduction.'),
        dict(id='final_report_package_review', status='pending', evidence='This delivery snapshot', limitation='Independent row-to-source report and package audit required before final acceptance.'),
    ]
    dest.mkdir()
    csv_write(dest / 'main_comparisons.csv', rows)
    csv_write(dest / 'diagnostic_comparisons.csv', diagnostics)
    csv_write(dest / 'fixed_models.csv', models)
    csv_write(dest / 'requirements.csv', requirements)
    frozen_json(dest / 'assessment.json', dict(validation_passed=positive_validation,
        confirmation_run=positive_validation, confirmation_passed=positive_confirmation,
        learned_method_core_effect_supported=positive_validation and positive_confirmation,
        original_sac_reproduced=False, requirements=requirements,
        final_report_review_pending=True, package_review_pending=True, goal_complete=False,
        hashes=hashes, source_hashes=source_hashes()))
    verdict = ('本改进方法在登记验证和独立确认中通过核心门槛；最终报告及包级审查仍待完成。'
               if positive_confirmation else '独立确认未通过，本改进方法尚未建立所要求的稳定核心效果。'
               if positive_validation else '完整验证未通过，独立确认保持封存，本改进方法尚未达到核心目标。')
    lines = ['# Bøhn 2021浅树改进：完整阶段结果', '', verdict, '',
        '这不是原SAC逐数字复现，也不是整个研究目标的自动完成声明。原SAC完整固定网格重建的历史报告未建立两任务跨种子稳定优势。'
        '浅树使用分类交叉熵优化三内部节点、四叶子的每步H选择策略；继承独立固定H终端，原50步掩码NLP、动力学与成本保持，'
        '有界求解重试及统一reset作为已披露适配。旧实验成本来自不同场景库，不与本轮绝对值拼表排名。', '',
        '## 全部主比较', '',
        '以下展示全部种子和两比较器。显示值取六位小数；通过/失败按未舍入原值判定。成本上界为绝对成本差的配对95%上界，'
        '延迟上界为实际决策时间比值95%上界。10,000次配对场景bootstrap；条件化于已拟合模型，种子和两次计时不是新增独立场景。', '',
        '|划分|任务|比较器|种子|固定H|成本变化%|成本差上界|延迟变化%|延迟比上界|安全|物理NI|回合内变化|成本路线|延迟路线|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for r in rows:
        lines.append('|%s|%s|%s|%d|%d|%s|%s|%s|%s|%s|%s|%s|%s|%s|' % (
            r['split'], r['task'], r['comparator'], r['seed'], r['fixed_h'], number(r['cost_percent']),
            number(r['cost_delta_upper']), number(r['latency_percent']), number(r['latency_ratio_upper']),
            r['safe'], r['control_noninferior'], r['adapted'], r['cost_gain'], r['time_gain']))
    lines += ['', '任务级还要求全部三种子、两比较器共同采用一条路线；单行通过不能抵消另一行失败。', '',
              '## 安全、失败、C5与模型', '',
              '安全比较同时检查成功、约束回合、初始和最终失败步率；失败率分母为实际控制步。车辆success指到达终点，倒立摆指运行至时限，'
              '不能解释为精确调节。原始次数、分母、物理与约束成本、每场景H和全部失败在各split_delivery及split_figures的CSV。', '',
              'C5是手工规则，组合是冻结学习策略加C5。每划分9项C5和12项组合比较均经独立效果复核并完整成图，全部排除于主学习门槛。', '',
              '|任务|种子|训练内选择|选择树|独特候选|重复候选执行|', '|---|---:|---|---|---:|---:|']
    for p in policies:
        lines.append('|%s|%d|%s|%s|%d|%d|' % (p['task'],p['seed'],p['selected'],p['learned_tree_selected'],p['unique_candidates'],p['repeated_candidate_evaluations']))
    lines += ['', '## 搜索、训练与仿真预算', '',
              '所有288候选（含重复）、固定参考、训练复选、冒烟、正式验证、双重复计时和获准的确认分别入账；继承模型与新增补训分列，不声称等预算优越。', '',
              '|记录项|数值|', '|---|---:|']
    for key, value in ledger['totals'].items():
        lines.append('|%s|%s|' % (key, '未知' if value is None else value))
    lines += ['', '继承终端训练步：%d；本轮新增固定终端训练步：%d；新增终端最终诊断步：%d；已保留审计离线积分：%d。' % (
        ledger['inherited_terminal_training_steps'], ledger['new_fixed_terminal_training_steps'],
        ledger['new_fixed_final_diagnostic_steps'], ledger['known_offline_integration_total']),
        '预算完整标记仅指已登记计数对账：%s。构造器内部和固定终端训练逐次NLP未仪表化；全研究求解总数与CPU未知。'
        '协议要求的每条件完整墙时/CPU未完整实现，不能由局部耗时补推。未保存凭据的旧审计重试或未刷盘历史工作不补为零。' % ledger['complete_budget'], '',
        '## 复运行和文件定位', '',
        '运行与分析解释器、实际包清单、历史安装入口、作者固定提交和归档哈希见delivery_environment/snapshot.json。'
        '这是运行后采集，非干净环境重建验证；各条件environment.json为当时的Python/平台/线程记录。历史runtime-freeze含本地构建URL，不直接当可移植锁文件。', '',
        '旧依赖清单与实际包一致：%s；不一致项：%s。作者归档提交/干净工作树一致：%s；不一致项：%s。'
        '任何不一致都需要解释，不能以已保存包清单替代环境可复现性。' % (
            env['legacy_pins_match'], ', '.join(env['legacy_pin_mismatches']) or '无',
            env['author_archive_commits_match'], ', '.join(env['author_source_mismatches']) or '无'), '',
        '本轮执行命令和退出码见native_pipeline_v2_status.json的completed_stages及各日志；训练入口为latency_tree_run.py --mode suite。'
        '已完成训练只核验后复用，禁止在现有输出目录盲目重启或运行会改写配置的旧setup/configure流程。', '',
        '训练产物：train/{task}_s{seed}/及training_delivery/。固定提名与新训练：baseline_selection.json、baseline_completion.json、extra_fixed/。'
        '原始评估：evaluations/{split}/{task}/；原始计时：timing_{split}/；全部轨迹、完成哈希与审计凭据保留。', '',
        '## 用户目标逐项核对', '', '|事项|状态|证据|限制|', '|---|---|---|---|']
    for r in requirements:
        lines.append('|%s|%s|%s|%s|' % (r['id'], r['status'], r['evidence'], r['limitation']))
    lines += ['', '## 结论边界与后续审查', '',
        '本报告重新核对主比较的精确安全率和布尔门槛，bootstrap数值依赖已通过的独立效果实现；不声称再次独立积分或另一次实验。'
        '历史原SAC和规则诊断仅核对引用文件及已有包摘要，没有重新执行其全部旧审计或打开旧封存测试。', '',
        '有限候选空间、单台WSL主机、模型条件化区间、当前场景分布和原实验配置缺失均限制推广。'
        '全部负结果保留；失败不能证明所有学习式自适应H无效，C5诊断收益不能证明学习贡献。', '',
        '独立报告与包级审查仍须逐行核对表格、全部C5结论、图表、模型来源、可复运行环境和用户要求。'
        '核心效果字段即使为真，也不自动将整个目标设为完成。若效果为负，继续依据完整证据诊断；是否已充分尝试并收尾由用户决定。', '']
    frozen_text(dest / 'report_CN.md', '\n'.join(lines))
    outputs = {str(p): sha(p) for p in dest.iterdir() if p.is_file()}
    frozen_json(dest / 'manifest.json', dict(generated=True, report_review_pending=True,
        package_review_pending=True, goal_complete=False, validation_passed=positive_validation,
        confirmation_run=positive_validation, confirmation_passed=positive_confirmation,
        input_hashes=hashes, output_hashes=outputs, source_hashes=source_hashes(),
        simulations=0, test_summary_accessed=positive_validation, raw_test_trajectories_read=False))
    return dict(generated=True, folder=str(dest), validation_passed=positive_validation,
                confirmation_passed=positive_confirmation, review_pending=True, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('register', 'check', 'environment', 'build'), required=True)
    parser.add_argument('--snapshot', type=int, default=1)
    parser.add_argument('--validation-figures', type=Path)
    parser.add_argument('--test-figures', type=Path)
    args = parser.parse_args()
    if args.mode == 'build':
        assert args.validation_figures is not None
        result = build(args.snapshot, args.validation_figures, args.test_figures)
    else:
        result = {'register': register, 'check': check, 'environment': environment}[args.mode]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
