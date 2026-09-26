"""Independent delivery arithmetic and artifact review; no new simulations.

Does not import the report generator, acceptance implementation, or plotting
functions. Numeric evidence and visual-review provenance are checked separately.
This limited review cannot decide that the user's entire research goal is done.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from xml.etree import ElementTree
from latency_tree_figure_data import idle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26'
SCRIPTS = Path(__file__).resolve().parent
NAMES = ('latency_tree_final_report.py', 'latency_tree_final_report_review.py', 'latency_tree_figure_data.py')
TASKS = ('vehicle', 'pendulum')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def hashes():
    return {str(SCRIPTS / name): digest(SCRIPTS / name) for name in NAMES}


def verify(mapping):
    for p, expected in mapping.items():
        assert digest(p) == expected, p


def write(path, record):
    if path.exists():
        assert read(path) == record, ('Existing differing review', str(path))
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')


def ready():
    idle(verify_sources=False)
    assert read(OUT / 'final_report_registration.json')['source_hashes'] == hashes()


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def exact_csv(saved, expected):
    assert set(saved) == set(expected)
    for key, value in expected.items():
        item = saved[key]
        if value is None:
            assert item == '', key
        elif type(value) is bool:
            assert item == str(value), key
        elif type(value) is int:
            assert item == str(value), key
        elif type(value) is float:
            close(float(item), value)
        else:
            assert item == value, key


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def flags(row):
    a, b, t = row['adaptive'], row['fixed'], row['timing']
    assert a['steps'] > 0 and b['steps'] > 0
    safety = (a['success'] >= b['success'] and a['constraint'] <= b['constraint'] and
              a['initial_failed_steps'] * b['steps'] <= b['initial_failed_steps'] * a['steps'] and
              a['solver_failure_steps'] * b['steps'] <= b['solver_failure_steps'] * a['steps'])
    physical = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(b['physical_constraint_cost'])
    total = a['total_cost'] <= b['total_cost'] + .02 * abs(b['total_cost'])
    rel = row['relative_cost_difference']
    return dict(safe=safety, control_noninferior=physical, total_noninferior=total,
                adapted=a['switches'] != 0,
                cost_gain=rel is not None and rel <= -.03 and row['cost_difference']['upper'] < 0,
                time_gain=t['mean_ratio'] <= .9 and t['upper'] < 1 and
                          t['repeat_ratios'][0] < 1 and t['repeat_ratios'][1] < 1 and total)


def main_record(row, split):
    values = flags(row)
    assert all(row[k] is v for k, v in values.items())
    reasons = [k for k in ('safe', 'control_noninferior', 'adapted') if values[k] is False]
    if not values['cost_gain'] and not values['time_gain']:
        reasons.append('neither_cost_nor_time_route')
    result = dict(split=split, task=row['task'], comparator=row['comparator'], seed=row['seed'],
                  fixed_h=row['fixed_h'], failures=';'.join(reasons),
                  cost_percent=None if row['relative_cost_difference'] is None else 100 * row['relative_cost_difference'],
                  cost_delta_upper=row['cost_difference']['upper'],
                  latency_percent=100 * (row['timing']['mean_ratio'] - 1),
                  latency_ratio_upper=row['timing']['upper'],
                  repeat0_ratio=row['timing']['repeat_ratios'][0], repeat1_ratio=row['timing']['repeat_ratios'][1])
    result.update(values)
    return result


def png_metadata(path):
    with path.open('rb') as stream:
        assert stream.read(8) == b'\x89PNG\r\n\x1a\n'
        found = {}
        while True:
            header = stream.read(8)
            assert len(header) == 8
            length, kind = struct.unpack('>I4s', header)
            if kind in (b'IHDR', b'pHYs'):
                payload = stream.read(length)
                if kind == b'IHDR':
                    width, height = struct.unpack('>II', payload[:8])
                    found.update(width=width, height=height, color_type=payload[9])
                else:
                    x, y, unit = struct.unpack('>IIB', payload)
                    found.update(dpi_x=x * .0254 if unit == 1 else None,
                                 dpi_y=y * .0254 if unit == 1 else None)
            else:
                stream.seek(length, 1)
            assert len(stream.read(4)) == 4
            if kind == b'IEND':
                break
        assert all(k in found for k in ('width', 'height', 'color_type', 'dpi_x', 'dpi_y'))
        return found


def check():
    ready()
    sample = dict(flag=True, count=12, value=.5, undefined=None, label='seed0')
    saved = dict(flag='True', count='12', value='0.5', undefined='', label='seed0')
    exact_csv(saved, sample)
    rejected = []
    for key, value in [('flag', 'False'), ('count', '13'), ('value', 'nan'),
                       ('undefined', '0'), ('label', 'seed1')]:
        try:
            exact_csv(dict(saved, **{key: value}), sample)
        except AssertionError:
            rejected.append(key)
    assert len(rejected) == 5
    b = dict(steps=100, success=10, constraint=0, initial_failed_steps=2,
             solver_failure_steps=1, switches=0, total_cost=100., physical_constraint_cost=100.)
    a = dict(b, steps=200, initial_failed_steps=4, solver_failure_steps=2, total_cost=97., switches=1)
    r = dict(adaptive=a, fixed=b, relative_cost_difference=-.03,
             cost_difference=dict(upper=-1.), timing=dict(mean_ratio=.9, upper=.99, repeat_ratios=[.8, .95]))
    assert all(flags(r).values())
    assert not flags(dict(r, adaptive=dict(a, initial_failed_steps=5)))['safe']
    assert not flags(dict(r, relative_cost_difference=None))['cost_gain']
    assert not flags(dict(r, timing=dict(r['timing'], upper=1.)))['time_gain']
    record = dict(passed=True, checks=10, exact_csv_mutations_rejected=rejected, source_hashes=hashes(),
                  simulations=0, test_accessed=False, goal_complete=False)
    write(OUT / 'final_report_review_checks.json', record)
    return record


def review(folder):
    ready()
    assert folder.parent.resolve() == OUT.resolve() and folder.name.startswith('final_delivery_')
    checks = read(OUT / 'final_report_review_checks.json')
    assert checks['passed'] and checks['source_hashes'] == hashes()
    manifest = read(folder / 'manifest.json')
    assert manifest['generated'] and manifest['goal_complete'] is False
    for field in ('input_hashes', 'output_hashes', 'source_hashes'):
        verify(manifest[field])
    assert manifest['source_hashes'] == hashes()
    assessment = read(folder / 'assessment.json')
    assert assessment['goal_complete'] is False and assessment['original_sac_reproduced'] is False
    assert assessment['hashes'] == manifest['input_hashes']
    rows = csv_rows(folder / 'main_comparisons.csv')
    diag = csv_rows(folder / 'diagnostic_comparisons.csv')
    expected_main, expected_diag = [], []
    validation = read(OUT / 'validation_delivery/effect_gate.json')
    splits = ('validation', 'test') if validation['effect_passed'] else ('validation',)
    observed = {}
    for split in splits:
        gate_path = OUT / (split + '_delivery/effect_gate.json')
        gate = read(gate_path)
        independent = read(OUT / (split + '_delivery/independent_review.json'))
        assert independent['passed'] and independent['effect_passed'] == gate['effect_passed']
        assert independent['hashes'][str(gate_path)] == digest(gate_path)
        verify(independent['hashes']); verify(gate['hashes'])
        assert len(gate['comparisons']) == 12 and len(gate['diagnostics']) == 21
        assert {(r['task'], r['comparator'], r['seed']) for r in gate['comparisons']} == {
            (t, c, s) for t in TASKS for c in ('independent', 'matched') for s in range(3)}
        expected_main.extend(main_record(r, split) for r in gate['comparisons'])
        task_gates = []
        for task in TASKS:
            selected = [flags(r) for r in gate['comparisons'] if r['task'] == task]
            value = (all(r['safe'] and r['control_noninferior'] and r['adapted'] for r in selected) and
                     (all(r['cost_gain'] for r in selected) or all(r['time_gain'] for r in selected)))
            assert gate['effects'][task]['passed'] == independent['effects'][task]['passed'] == value
            task_gates.append(value)
        observed[split] = all(task_gates)
        assert observed[split] == gate['effect_passed']
        for r in gate['diagnostics']:
            assert r['contributes_to_main_gate'] is False
            expected_diag.append(dict(split=split, method=r['family'], comparator=r['comparator'], seed=r['seed'],
                contributes_to_main_gate=False,
                cost_percent=None if r['relative_cost_difference'] is None else 100 * r['relative_cost_difference'],
                cost_delta_lower=r['cost_difference']['lower'], cost_delta_upper=r['cost_difference']['upper'],
                latency_percent=100 * (r['timing']['mean_ratio'] - 1),
                latency_ratio_lower=r['timing']['lower'], latency_ratio_upper=r['timing']['upper'],
                repeat0_ratio=r['timing']['repeat_ratios'][0], repeat1_ratio=r['timing']['repeat_ratios'][1],
                **{k: r[k] for k in ('safe', 'control_noninferior', 'total_noninferior', 'adapted', 'cost_gain', 'time_gain')}))
    assert len(rows) == len(expected_main) and len(diag) == len(expected_diag)
    for actual, expected in zip(rows, expected_main):
        exact_csv(actual, expected)
    for actual, expected in zip(diag, expected_diag):
        exact_csv(actual, expected)
    assert assessment['validation_passed'] == observed['validation']
    assert assessment['confirmation_run'] == ('test' in splits)
    assert assessment['confirmation_passed'] == observed.get('test', False)
    assert assessment['learned_method_core_effect_supported'] == (observed['validation'] and observed.get('test', False))
    if 'test' not in splits:
        assert not any((OUT / p).exists() for p in ('confirmation_registration.json', 'evaluations/test', 'timing_test', 'test_delivery'))
    else:
        confirmation = read(OUT / 'confirmation_registration.json')
        assert confirmation['validation_gate_passed'] and confirmation['independent_review_passed']
        verify(confirmation['hashes'])
    model_rows = csv_rows(folder / 'fixed_models.csv')
    selection = read(OUT / 'baseline_selection.json')['nominations']
    inventory = read(OUT / 'evaluation_registration.json')['model_inventory']
    expected_models = []
    for task in TASKS:
        for comparator in ('independent', 'matched'):
            for seed in range(3):
                h = selection[task][comparator + '_h']
                trained = h if comparator == 'independent' else (25 if task == 'vehicle' else 30)
                key = '%s_h%d_s%d' % (task, trained, seed)
                p = Path(inventory[key]) if inventory[key] else OUT / 'extra_fixed' / key
                saved, completed = read(p / 'manifest.json'), read(p / 'completed.json')
                assert saved['task'] == task and saved['seed'] == seed and saved['fixed_horizon'] == trained
                assert saved['steps'] == completed['steps'] == 15000 and completed['status'] == 'complete'
                expected_models.append(dict(task=task, comparator=comparator, seed=seed, evaluation_h=h,
                    trained_h=trained, training_steps=15000, folder=str(p), model_sha256=digest(p / 'model.zip'),
                    new_model=OUT in p.parents))
    assert len(model_rows) == len(expected_models) == 12
    for actual, expected in zip(model_rows, expected_models):
        exact_csv(actual, expected)
    requirements = csv_rows(folder / 'requirements.csv')
    assert len(requirements) == len(assessment['requirements']) == 12
    for actual, expected in zip(requirements, assessment['requirements']):
        exact_csv(actual, expected)
    requirements_by_id = {r['id']: r for r in requirements}
    assert len(requirements_by_id) == 12
    assert requirements_by_id['per_condition_wall_cpu']['status'] == 'not_fully_implemented'
    assert requirements_by_id['environment_reconstruction']['status'] == 'inventory_captured_recreation_unverified'
    assert requirements_by_id['final_report_package_review']['status'] == 'pending'
    text = (folder / 'report_CN.md').read_text(encoding='utf-8')
    shown = [line for line in text.splitlines() if line.startswith('|validation|') or line.startswith('|test|')]
    assert len(shown) == len(expected_main)
    for line, expected in zip(shown, expected_main):
        cells = line.strip('|').split('|')
        assert len(cells) == 14
        assert cells[:5] == [expected['split'], expected['task'], expected['comparator'], str(expected['seed']), str(expected['fixed_h'])]
        for cell, key in zip(cells[5:9], ('cost_percent', 'cost_delta_upper', 'latency_percent', 'latency_ratio_upper')):
            wanted = '未定义/未知' if expected[key] is None else format(expected[key], '.6f')
            assert cell == wanted, (key, cell, wanted)
        for cell, key in zip(cells[9:], ('safe', 'control_noninferior', 'adapted', 'cost_gain', 'time_gain')):
            assert cell == str(expected[key])
    ledger = read(OUT / 'final_budget/ledger.json')
    for key, value in ledger['totals'].items():
        assert '|%s|%s|' % (key, '未知' if value is None else value) in text
    env = read(OUT / 'delivery_environment/snapshot.json')
    verify(env['hashes'])
    assert env['legacy_pins_match'] == (not env['legacy_pin_mismatches'])
    assert env['author_archive_commits_match'] == (not env['author_source_mismatches'])
    assert ('旧依赖清单与实际包一致：%s' % env['legacy_pins_match']) in text
    assert ('作者归档提交/干净工作树一致：%s' % env['author_archive_commits_match']) in text
    for statement in ('不是原SAC逐数字复现', '每条件完整墙时/CPU未完整实现', '不能解释为精确调节',
                      '不是新增独立场景', '不声称等预算优越', '是否已充分尝试并收尾由用户决定'):
        assert statement in text, statement
    assert ('独立确认保持封存' in text) == ('test' not in splits)
    assert ledger['test_phase_included'] == ('test' in splits)
    figure_manifests = [Path(p) for p in manifest['input_hashes'] if Path(p).name == 'export_manifest.json']
    assert len(figure_manifests) == len(splits)
    verified_figures, verified_pngs = 0, 0
    figure_splits = set()
    for path in figure_manifests:
        saved = read(path)
        figure_splits.add(saved['split'])
        assert saved['visual_review_passed'] and saved['goal_complete'] is False
        verify(saved['hashes'])
        for row in saved['figures'].values():
            colors = []
            for name in row['files']:
                p = Path(name)
                if p.suffix == '.png':
                    metadata = png_metadata(p)
                    assert metadata['dpi_x'] is not None and metadata['dpi_y'] is not None
                    assert abs(metadata['dpi_x'] - 300) < .1 and abs(metadata['dpi_y'] - 300) < .1
                    assert abs(metadata['width'] / metadata['dpi_x'] - row['size_inches'][0]) < .02
                    assert abs(metadata['height'] / metadata['dpi_y'] - row['size_inches'][1]) < .02
                    colors.append(metadata['color_type'])
                    verified_pngs += 1
                elif p.suffix == '.svg':
                    root = ElementTree.parse(str(p)).getroot()
                    assert not any(element.tag.rsplit('}', 1)[-1] == 'image' for element in root.iter())
                elif p.suffix == '.pdf':
                    with p.open('rb') as stream:
                        assert stream.read(5) == b'%PDF-'
                else:
                    raise AssertionError(p)
            assert len(colors) == 2 and 0 in colors, ('Missing grayscale PNG', row)
            verified_figures += 1
    assert figure_splits == set(splits)
    assert verified_figures == 14 + (12 if 'test' in splits else 0)
    record = dict(passed_report_tables=True, main_comparisons=len(rows), diagnostic_comparisons=len(diag),
                  fixed_model_rows=len(model_rows), requirements=len(requirements), figures=verified_figures,
                  pngs=verified_pngs, validation_passed=observed['validation'],
                  confirmation_run='test' in splits, confirmation_passed=observed.get('test', False),
                  prose_semantic_review_complete=False, actual_image_review_in_this_step=False,
                  full_transitive_package_audit_complete=False, fresh_environment_reproduction_complete=False,
                  goal_complete=False, simulations=0, raw_test_trajectories_read=False,
                  test_summary_accessed='test' in splits, source_hashes=hashes(),
                  hashes={str(folder / 'manifest.json'): digest(folder / 'manifest.json'),
                          str(OUT / 'final_report_review_checks.json'): digest(OUT / 'final_report_review_checks.json')},
                  scope='Independent CSV/Markdown arithmetic and fixed model row mapping, hash verification, '
                        'PNG dimensions/DPI/grayscale and SVG raster exclusion. No new bootstrap or integration; '
                        'does not replace scientific interpretation, actual image viewing or full package audit.')
    write(folder / 'independent_table_artifact_review.json', record)
    return {k: v for k, v in record.items() if k not in ('hashes', 'source_hashes')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if args.check:
        result = check()
    else:
        assert args.folder is not None
        result = review(args.folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
