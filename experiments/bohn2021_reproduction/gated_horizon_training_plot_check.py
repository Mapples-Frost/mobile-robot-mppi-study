"""Verify training-only data coverage, rendered markers, sizes and embedded fonts.

Run with Pillow and pypdf; supports WSL paths and the Windows UNC mount.
Does not import the simulator or read validation/test outcomes.
"""
import csv
import hashlib
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import pypdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
DEST = OUT / 'training_figures'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
NS = {'s': 'http://www.w3.org/2000/svg'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(name):
    return ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11), (a, b)


def check_svg(path, expected):
    tree = ET.parse(path).getroot()
    assert not tree.findall('.//s:image', NS), 'Raster image in vector figure'
    clips, points = {}, []
    for group in tree.findall('.//s:clipPath', NS):
        rect = group.find('s:rect', NS)
        if rect is not None:
            clips[group.attrib['id']] = tuple(float(rect.attrib[k]) for k in ('x', 'y', 'width', 'height'))

    def inspect(node, active_clip=None):
        clip = node.attrib.get('clip-path', active_clip)
        if node.tag.endswith('}use'):
            assert clip and clip.startswith('url(#') and clip.endswith(')'), clip
            left, top, width, height = clips[clip[5:-1]]
            x, y = float(node.attrib['x']), float(node.attrib['y'])
            assert left < x < left + width and top < y < top + height, (path.name, x, y)
            points.append((x, y))
        elif node.tag.endswith('}path') and clip is not None:
            # Matplotlib emits a lone hollow marker as an inline absolute path.
            # Verify that path's control points instead of dropping the marker.
            assert clip.startswith('url(#') and clip.endswith(')') and 'transform' not in node.attrib
            left, top, width, height = clips[clip[5:-1]]
            data = node.attrib['d']
            number = r'[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?'
            assert set(re.sub(number, '', data).split()) <= {'M', 'L', 'C', 'Z', 'z'}
            values = [float(value) for value in re.findall(number, data)]
            assert values and len(values) % 2 == 0
            xs, ys = values[::2], values[1::2]
            assert all(left < x < left + width for x in xs) and all(top < y < top + height for y in ys)
            points.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
        for child in node:
            inspect(child, clip)

    for group in tree.findall('.//s:g', NS):
        if group.attrib.get('id', '').startswith('PathCollection_'):
            inspect(group)
    assert len(points) == expected, (path.name, len(points), expected)
    return dict(marker_count=len(points), marker_centers_inside_clip_bounds=True, embedded_images=0)


def check_pdf(path, width, height):
    reader = pypdf.PdfReader(path)
    assert len(reader.pages) == 1
    page, fonts = reader.pages[0], []
    close(float(page.mediabox.width) / 72, width)
    close(float(page.mediabox.height) / 72, height)

    def inspect(font_ref):
        font = font_ref.get_object()
        assert font.get('/Subtype') != '/Type3', 'Type3 font is not acceptable'
        if '/DescendantFonts' in font:
            for child in font['/DescendantFonts']:
                inspect(child)
            return
        descriptor = font['/FontDescriptor'].get_object()
        programs = [descriptor[k].get_object().get_data() for k in ('/FontFile', '/FontFile2', '/FontFile3') if k in descriptor]
        assert programs and all(len(p) > 0 for p in programs), 'Missing embedded font program'
        fonts.append(dict(name=str(font['/BaseFont']), subtype=str(font['/Subtype']), embedded_bytes=sum(map(len, programs))))

    for font_ref in page['/Resources']['/Font'].get_object().values():
        inspect(font_ref)
    assert fonts
    return dict(pages=1, dimensions_inches=[width, height], fonts=fonts, all_fonts_embedded=True, type3_fonts=0)


def main():
    prepared, exported, review = (read(DEST / n) for n in ('input_audit.json', 'exports.json', 'visual_review.json'))
    assert prepared['passed'] and exported['training_only'] and review['passed']
    for record in (prepared, exported):
        for name, value in record['hashes'].items():
            assert digest(source_path(name)) == value, name
    for name, value in prepared['csv_hashes'].items():
        assert digest(DEST / name) == value, name
    for name, value in review['preview_hashes'].items():
        assert digest(DEST / (name + '_preview.png')) == value
        assert digest(DEST / (name + '_grayscale.png')) == review['grayscale_hashes'][name]
        assert not read(DEST / (name + '_layout.json'))['issues']
    plot_source = ROOT / 'experiments/bohn2021_reproduction/gated_horizon_training_plot.py'
    assert digest(plot_source) == exported['source_hash'] == review['source_hash'] == digest(DEST / 'plot_source.py')
    cases, models = rows(DEST / 'all_cases.csv'), rows(DEST / 'all_models.csv')
    original = rows(OUT / 'selected_training_diagnostics/all_paired_training_episodes.csv')
    original_map = {(r['task'], int(r['seed']), int(r['case'])): r for r in original}
    expected = {(t, s, c) for t in ('vehicle', 'pendulum') for s in range(3) for c in range(24)}
    keys = [(r['task'], int(r['seed'][1:]), int(r['case'][1:])) for r in cases]
    assert len(cases) == len(original) == len(original_map) == 144 and set(keys) == set(original_map) == expected
    assert len(models) == 6 and {(r['task'], r['seed']) for r in models} == {(t, 'S%d' % s) for t in ('vehicle', 'pendulum') for s in range(3)}
    for row, key in zip(cases, keys):
        origin = original_map[key]
        for column, source in [('raw_cost_delta', 'delta_total_cost'), ('physical_cost_delta', 'delta_physical_constraint_cost'), ('h_penalty_delta', 'delta_h_penalty'), ('short_steps', 'short_steps'), ('steps', 'steps')]:
            close(row[column], origin[source])
        close(row['short_step_percent'], 100 * int(row['short_steps']) / int(row['steps']))
        mode = 'failed' if origin['fixed_success'] == 'False' or origin['selected_success'] == 'False' else 'short' if int(origin['short_steps']) else 'unchanged_h'
        assert row['mode'] == mode
    for model in models:
        group = [r for r in cases if (r['task'], r['seed']) == (model['task'], model['seed'])]
        assert len(group) == int(model['cases']) == 24
        for key in ('raw_cost_delta', 'physical_cost_delta', 'h_penalty_delta'):
            close(model[key], sum(float(r[key]) for r in group) / 24)
        close(model['raw_cost_delta'], float(model['physical_cost_delta']) + float(model['h_penalty_delta']))
        close(model['short_step_percent'], 100 * sum(int(r['short_steps']) for r in group) / sum(int(r['steps']) for r in group))
        assert int(model['short_cases']) == sum(int(r['short_steps']) > 0 for r in group)
        assert int(model['failed_cases']) == sum(r['mode'] == 'failed' for r in group)
    artifacts = {}
    for name, height, count in [('case_differences', 6.5, 144), ('cost_components', 4.5, 24)]:
        with Image.open(DEST / (name + '.png')) as im:
            assert im.size == (2160, round(300 * height))
            assert all(abs(dpi - 300) < .01 for dpi in im.info['dpi'])
            png = dict(pixels=list(im.size), dpi=list(im.info['dpi']))
        artifacts[name] = dict(png=png, svg=check_svg(DEST / (name + '.svg'), count), pdf=check_pdf(DEST / (name + '.pdf'), 7.2, height))
    result = dict(passed=True, scope='Training-only descriptive artifacts; not validation efficacy or reproduction success.', checked_at=datetime.now(timezone.utc).isoformat(),
        cases=144, models=6, cases_per_model=24, failed_cases_retained=sum(r['mode'] == 'failed' for r in cases), mode_counts=dict(Counter(r['mode'] for r in cases)), artifacts=artifacts,
        pdf_checker_note='Generic checker warned about missing descriptors on Type0 wrappers. This check follows DescendantFonts and verifies actual nonempty embedded font streams.',
        python=platform.python_version(), executable=sys.executable, pypdf_version=pypdf.__version__, checker_source_hash=digest(Path(__file__)),
        hashes={n:digest(DEST / n) for n in ['input_audit.json','visual_review.json','exports.json','all_cases.csv','all_models.csv'] + [n + '.' + e for n in ('case_differences','cost_components') for e in ('pdf','svg','png')]})
    (DEST / 'deliverable_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, cases=144, models=6, failed_cases_retained=result['failed_cases_retained'], artifacts=artifacts), indent=2))


if __name__ == '__main__':
    main()
