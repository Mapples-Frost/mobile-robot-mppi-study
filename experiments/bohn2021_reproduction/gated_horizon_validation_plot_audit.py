"""Check validation figure data, complete markers/intervals, fonts and exports.

Run after manual color/grayscale QA using the bundled document Python with
Pillow/pypdf. No control simulation, model choice, or sealed-test reads.
"""
import csv
import hashlib
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
DEST = RUN / 'validation_figures'
REG = RUN / 'validation_plot_audit_registration.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def main():
    from gated_horizon_latency_delivery import guard, resolve, relative
    guard()
    from gated_horizon_training_plot_check import check_pdf, check_svg
    from PIL import Image
    import pypdf
    hashes = {}

    def record(path, expected=None):
        key, value = relative(path), sha(path)
        assert expected is None or expected == value, 'Hash mismatch: ' + key
        assert key not in hashes or hashes[key] == value, 'Input changed: ' + key
        hashes[key] = value

    amendment_path = RUN / 'validation_plot_audit_amendment.json'
    registration = read(REG)
    if amendment_path.exists():
        amendment = read(amendment_path)
        record(REG, amendment['original_registration_hash'])
        for name, value in amendment['preserved_source_hashes'].items():
            record(resolve(name), value)
        registration = amendment
        record(amendment_path)
    for name, value in registration['source_hashes'].items():
        record(resolve(name), value)
    record(REG)
    prepared = read(DEST / 'input_audit.json')
    review = read(DEST / 'visual_review.json')
    layout = read(DEST / 'layout_audit.json')
    assert prepared['split'] == 'validation' and prepared['cases'] == 32
    assert prepared['all_seeds_retained'] and review['passed'] and not layout['issues']
    source = resolve(prepared['source'])
    assert source == RUN / 'validation_delivery/effect_gate.json'
    record(source, prepared['source_hash'])
    record(DEST / 'all_seed_comparisons.csv', prepared['csv_hash'])
    record(DEST / 'cost_latency_preview.png', review['preview_hash'])
    record(DEST / 'cost_latency_grayscale.png', review['grayscale_hash'])
    record(ROOT / 'experiments/bohn2021_reproduction/gated_horizon_plot.py', layout['source_hash'])
    assert layout['preview_hash'] == review['preview_hash']
    effect = read(source)
    assert effect['timing_audited'] and effect['extra_baselines_audited']
    assert prepared['registered_effect_passed'] == effect['validation_effect_passed']
    for name, value in effect['hashes'].items():
        record(resolve(name), value)
    independent = read(RUN / 'effect_gate_review/report.json')
    assert independent['passed'] and independent['validation_effect_passed'] == effect['validation_effect_passed']
    assert independent['comparisons_checked'] == 12 and independent['test_accessed'] is False
    records = effect['comparisons']
    expected_keys = {(t, s, c) for t in ('vehicle', 'pendulum') for s in range(3) for c in ('independent', 'matched')}
    source_map = {(r['task'], r['seed'], r['comparator']): r for r in records}
    assert set(source_map) == expected_keys and len(records) == 12
    with (DEST / 'all_seed_comparisons.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == prepared['rows'] == layout['rows']
    seen = set()
    for row in rows:
        labels = row['comparator'].split('+')
        assert len(set(labels)) == len(labels)
        matched = []
        for label in labels:
            key = (row['task'], int(row['seed'][1:]), label)
            assert key in source_map and key not in seen
            seen.add(key)
            matched.append(source_map[key])
        origin = matched[0]
        for other in matched[1:]:
            assert {k: v for k, v in origin.items() if k != 'comparator'} == {
                k: v for k, v in other.items() if k != 'comparator'}, 'Distinct arms merged'
        assert int(row['fixed_h']) == origin['fixed_h']
        scale = max(abs(origin['fixed']['total_cost']), 1e-12)
        close(row['cost_change_percent'], 100 * origin['relative_cost_difference'])
        close(row['time_change_percent'], 100 * (origin['timing']['mean_ratio'] - 1))
        for endpoint in ('lower', 'upper'):
            close(row['cost_' + endpoint + '_percent'], 100 * origin['cost_difference'][endpoint] / scale)
            close(row['time_' + endpoint + '_percent'], 100 * (origin['timing'][endpoint] - 1))
        for repeat in (0, 1):
            close(row['time_repeat%d_percent' % repeat], 100 * (origin['timing']['repeat_ratios'][repeat] - 1))
        assert row['safety_control_adaptation'] == str(bool(origin['safe'] and origin['control_noninferior'] and origin['adapted']))
        for arm in ('adaptive', 'fixed'):
            for metric in ('success', 'constraints', 'initial_failures', 'final_failures'):
                assert int(row[arm + '_' + metric]) == origin[arm][metric]
    assert seen == expected_keys
    svg_path = DEST / 'cost_latency.svg'
    svg = check_svg(svg_path, 4 * len(rows))
    ns = {'s': 'http://www.w3.org/2000/svg'}
    tree = ET.parse(svg_path).getroot()
    clips = {node.attrib['id']: tuple(float(node.find('s:rect', ns).attrib[k]) for k in ('x', 'y', 'width', 'height'))
             for node in tree.findall('.//s:clipPath', ns) if node.find('s:rect', ns) is not None}
    intervals = 0
    number = r'[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?'
    for group in tree.findall('.//s:g', ns):
        if not group.attrib.get('id', '').startswith('LineCollection_'):
            continue
        for path in group.findall('.//s:path', ns):
            data = path.attrib['d']
            assert set(re.sub(number, '', data).split()) <= {'M', 'L'}
            coords = [float(x) for x in re.findall(number, data)]
            assert len(coords) == 4
            clip = path.attrib.get('clip-path', group.attrib.get('clip-path'))
            assert clip and clip.startswith('url(#') and clip.endswith(')')
            left, top, width, height = clips[clip[5:-1]]
            assert all(left < x < left + width for x in coords[::2])
            assert all(top < y < top + height for y in coords[1::2])
            intervals += 1
    assert intervals == 2 * len(rows), ('Missing cost/latency intervals', intervals)
    assert len([g for g in tree.findall('.//s:g', ns) if re.fullmatch(r'axes_\d+', g.attrib.get('id', ''))]) == 4
    with Image.open(DEST / 'cost_latency.png') as image:
        assert image.size == (2160, 1950)
        assert all(abs(dpi - 300) < .01 for dpi in image.info['dpi'])
        png = dict(pixels=list(image.size), dpi=list(image.info['dpi']))
    pdf = check_pdf(DEST / 'cost_latency.pdf', 7.2, 6.5)
    for name in ('input_audit.json', 'visual_review.json', 'layout_audit.json', 'data_profile.md',
                 'cost_latency.png', 'cost_latency.pdf', 'cost_latency.svg'):
        record(DEST / name)
    record(RUN / 'effect_gate_review/report.json')
    for name, value in hashes.items():
        assert sha(resolve(name)) == value
    result = dict(passed=True, comparisons=12, figure_rows=len(rows), data_markers=4 * len(rows),
                  intervals=intervals, all_seed_labels_retained=True, test_accessed=False,
                  validation_effect_passed=effect['validation_effect_passed'], svg=svg, png=png, pdf=pdf,
                  hashes=hashes, python=platform.python_version(), pypdf_version=pypdf.__version__,
                  executable=sys.executable, source_hash=sha(Path(__file__)),
                  scope='Figure data and artifact integrity only. No new statistical test or efficacy claim; visual review is separately required.')
    path = DEST / 'deliverable_check.json'
    if path.exists():
        assert read(path) == result, 'Previous delivery check differs; preserve and inspect'
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'comparisons', 'figure_rows', 'data_markers', 'intervals', 'test_accessed')}, indent=2))


if __name__ == '__main__':
    main()
