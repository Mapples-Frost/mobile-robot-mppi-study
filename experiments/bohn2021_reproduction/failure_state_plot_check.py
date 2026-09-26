"""Read-only data/figure verification; write only an idempotent audit receipt.

Use a Python with Pillow and pypdf. No simulator or scientific report generator
is imported. The general SVG/PDF inspectors are inherited from the prior audit.
"""
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from PIL import Image
import pypdf
from gated_horizon_training_plot_check import check_pdf, check_svg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
DEST = OUT / 'figures'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-10), (a, b)


def main():
    spec, prepared, layout, visual = (read(DEST / n) for n in
        ('registration.json', 'input_audit.json', 'layout_audit.json', 'visual_review.json'))
    plot = ROOT / 'experiments/bohn2021_reproduction/failure_state_plot.py'
    assert spec['source_sha256'] == layout['source_sha256'] == visual['source_sha256'] == sha(plot)
    assert spec['diagnostic_only'] and prepared['passed'] and layout['exported'] and visual['passed']
    assert prepared['registration_sha256'] == sha(DEST / 'registration.json')
    assert prepared['inputs'] == spec['inputs']
    for name, digest in spec['inputs'].items():
        assert sha(OUT / name) == digest, name
    full = read(OUT / 'delivery/report.json')
    timing = read(OUT / 'timing_delivery/report.json')
    assert full['passed'] and timing['passed'] and timing['diagnostic_only']
    assert not full['goal_complete'] and not timing['test_accessed']
    assert read(OUT / 'delivery_review/report.json')['passed']
    assert not (OUT.parent / 'gated_horizon_search_2026-09-25/evaluations/test').exists()
    full_lookup = {(r['task'], r['seed'], r['policy']): r for r in full['comparisons']}
    timing_lookup = {(r['seed'], r['policy']): r for r in timing['comparisons']}
    assert len(full_lookup) == 18 and len(timing_lookup) == 9
    artifacts, checked_files = {}, [Path(__file__), plot,
        ROOT / 'experiments/bohn2021_reproduction/gated_horizon_training_plot_check.py']
    for task, policies, markers in (
        ('pendulum', ('certificate5', 'certificate10', 'certificate15'), 45),
        ('vehicle', ('fixed30', 'fixed35', 'selected'), 18)):
        assert sha(DEST / (task + '.csv')) == prepared['task_csv_hashes'][task]
        with (DEST / (task + '.csv')).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        expected = [(task, seed, policy) for seed in range(3) for policy in policies]
        keys = [(task, int(r['seed'][1:]), r['policy']) for r in rows]
        assert keys == expected and len(set(keys)) == 9
        for r, key in zip(rows, keys):
            source = full_lookup[key]
            for actual, origin in (('total_cost_change_percent', 'total_change_percent'),
                                   ('physical_cost_change_percent', 'physical_change_percent')):
                close(r[actual], source[origin])
            for field in ('safe', 'physical_noninferior'):
                assert r[field] == str(source[field])
            if task == 'pendulum':
                source = timing_lookup[key[1:]]
                for actual, origin in (('time_change_percent', 'mean_time_change_percent'),
                    ('repeat0_percent', 'repeat0_change_percent'), ('repeat1_percent', 'repeat1_change_percent')):
                    close(r[actual], source[origin])
        assert not layout['tasks'][task]['issues']
        assert layout['tasks'][task]['point_markers'] == markers
        assert layout['tasks'][task]['plotted_rows'] == 9
        assert sha(DEST / (task + '_preview.png')) == visual['previews'][task] == layout['tasks'][task]['preview_sha256']
        assert sha(DEST / (task + '_grayscale.png')) == visual['grayscale'][task] == layout['tasks'][task]['grayscale_sha256']
        with Image.open(DEST / (task + '.png')) as im:
            assert im.size == (2160, 1620)
            assert all(abs(dpi - 300) < .01 for dpi in im.info['dpi'])
            png = dict(pixels=list(im.size), dpi=list(im.info['dpi']))
        artifacts[task] = dict(rows=9, png=png,
            svg=check_svg(DEST / (task + '.svg'), markers),
            pdf=check_pdf(DEST / (task + '.pdf'), 7.2, 5.4))
        checked_files += [DEST / (task + suffix) for suffix in
            ('.csv', '_profile.md', '_preview.png', '_grayscale.png', '.png', '.pdf', '.svg')]
    checked_files += [DEST / name for name in ('registration.json', 'input_audit.json',
                                              'visual_review.json', 'layout_audit.json', 'render_environment_note.json')]
    result = dict(passed=True, diagnostic_only=True, rows=18, artifacts=artifacts,
        simulations=0, test_accessed=False, core_effect_evaluated=False,
        python=platform.python_version(), pypdf_version=pypdf.__version__,
        font_note='Follow Type0 DescendantFonts; require nonempty embedded font programs and reject Type3.',
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in checked_files})
    target = DEST / 'artifact_check.json'
    if target.exists():
        assert read(target) == result, 'Audit differs; preserve prior receipt before a new environment or source revision'
    else:
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write(chr(10))
    print(json.dumps(dict(passed=True, rows=18, artifacts=artifacts), indent=2))


if __name__ == '__main__':
    main()
