"""Verify a bounded diagnostic delivery snapshot and transitive raw hashes.

No simulations, plotting, source repair or scientific report rewrites. Use
--register once after delivery; ordinary execution verifies and writes an
idempotent receipt. Changed artifacts fail closed and require a new snapshot.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
MANIFEST = OUT / 'package_manifest.json'
RECEIPT = OUT / 'package_check.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        assert read(path) == value, ('Existing snapshot differs', str(path))
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write(chr(10))


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-10), (a, b)


def register():
    paths = list(OUT.glob('*.json'))
    paths = [p for p in paths if p not in (MANIFEST, RECEIPT)]
    paths += [OUT / 'README_CN.md', OUT / 'timing/environment.json', OUT / 'timing/completed.json']
    for name in ('delivery', 'timing_delivery', 'delivery_review', 'timing_controller_repair', 'figures', 'report_history'):
        paths += [p for p in (OUT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += list((ROOT / 'experiments/bohn2021_reproduction').glob('failure_state*.py'))
    paths += [ROOT / 'docs/reports/bohn2021_failure_state_probe_2026-09-26.md',
              ROOT / 'docs/reports/bohn2021_goal_requirements_audit_2026-09-24.md',
              ROOT / 'docs/protocols/bohn2021_failure_state_probe_2026-09-26.md']
    value = dict(diagnostic_delivery_complete=True, goal_complete=False,
        diagnostic_only=True, test_accessed=False, new_training_steps=0,
        scope='Full diagnosis and measured timing delivery, transitive source/trace hashes, independent arithmetic ledger and figures. Not independent efficacy.',
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))})
    write_new(MANIFEST, value)


def main():
    manifest = read(MANIFEST)
    assert manifest['diagnostic_delivery_complete'] and not manifest['goal_complete']
    seen = {}

    def verify(values, base=None):
        for name, digest in values.items():
            path = Path(name) if base is None else base / name
            if str(path) in seen:
                assert seen[str(path)] == digest
            assert sha(path) == digest, str(path)
            seen[str(path)] = digest

    verify(manifest['hashes'], ROOT)
    for name in ('status.json', 'finish_status.json', 'timing_status.json'):
        status = read(OUT / name)
        assert status['complete'] and not status['active']
        assert not (Path('/proc') / str(status['pid']) / 'cmdline').exists()
        verify(status['output_hashes'])
    for phase in ('full', 'timing'):
        done_paths = list((OUT / phase).glob('*/completed.json'))
        assert len(done_paths) == 24
        for path in done_paths:
            verify(read(path)['hashes'])
    original = {'registration.json': '4e47d7155a98be9d19626487c1ca2fe63bfecc3f906f54eefad9f415d5c16886',
        'timing_registration.json': '22bf6354ad3991f6b140d6f4df0ed5f233bbc7371cb58502b003dcbe6d8a0456',
        'finish_registration.json': '24e6d8d585f0ccad9a326669295e47adc362be2ad9be458e2fc03ef48744f834',
        'delivery/report.json': 'cc2c728134c919ee83e46a507d3a978f73c2b8cdec472b8396bbbdf00643da57',
        'timing_delivery/report.json': '88139d6f32ef9539ae35846d2e4f563950a0d9933567c085e0e23e0341ab70ee'}
    verify(original, OUT)
    for folder in ('delivery', 'timing_delivery'):
        report = read(OUT / folder / 'report.json')
        assert report['passed'] and report['diagnostic_only']
        verify(report['hashes'])
        verify(report['delivery_hashes'], OUT / folder)
    review = read(OUT / 'delivery_review/report.json')
    assert review['passed'] and review['simulations'] == 0
    verify(review['inputs'])
    timing = read(OUT / 'timing_delivery/report.json')
    assert (timing['conditions'], timing['episodes'], timing['steps']) == (24, 576, 37568)
    assert timing['core_effect_not_evaluated'] and not timing['test_accessed']
    with (OUT / 'timing_delivery/all_conditions.csv').open(newline='') as f:
        conditions = list(csv.DictReader(f))
    lookup = {(int(r['seed']), r['policy'], int(r['repeat'])): r for r in conditions}
    assert len(lookup) == len(conditions) == 24
    assert len(timing['comparisons']) == 9
    for row in timing['comparisons']:
        ratios = [float(lookup[(row['seed'], row['policy'], rep)]['decision_mean_s']) /
                  float(lookup[(row['seed'], 'fixed30', rep)]['decision_mean_s']) for rep in range(2)]
        close(row['mean_time_change_percent'], 100 * (sum(ratios) / 2 - 1))
        for rep in range(2):
            close(row['repeat%d_change_percent' % rep], 100 * (ratios[rep] - 1))
        assert row['both_repeats_faster'] == all(r < 1 for r in ratios)
    fields = ('step_attempts', 'reset_attempts', 'solve_attempts', 'solve_completed', 'warmups', 'retries')
    groups = {}; counters = timing['budget']['counter_rows']
    assert len(counters) == timing['budget']['counter_files'] == 174
    assert len({r['path'] for r in counters}) == len(counters)
    for row in counters:
        path = Path(row['path']); verify({str(path): row['sha256']})
        phase = path.relative_to(OUT).parts[0]
        assert phase in ('full', 'smoke', 'serialization_repair', 'timing')
        group = groups.setdefault(phase, {field: 0 for field in fields})
        raw = read(path)
        if row['kind'] == 'environment':
            group['step_attempts'] += raw['step_calls']
            group['reset_attempts'] += raw['reset_calls']
        else:
            assert row['kind'] == 'solver'
            for target, source in (('solve_attempts', 'solve_attempts'), ('solve_completed', 'solve_completed'),
                                   ('warmups', 'warmup_attempts'), ('retries', 'retry_attempts')):
                group[target] += raw[source]
    total = {field: sum(g[field] for g in groups.values()) for field in fields}
    assert groups == timing['budget']['groups'] and total == timing['budget']['total']
    assert tuple(total[f] for f in fields) == (69570, 1032, 78655, 78655, 1032, 8053)
    artifacts = read(OUT / 'figures/artifact_check.json')
    assert artifacts['passed'] and artifacts['rows'] == 18
    verify(artifacts['hashes'], ROOT)
    previous = OUT.parent / 'gated_horizon_search_2026-09-25'
    assert not (previous / 'evaluations/test').exists()
    assert not (previous / 'confirmation_registration.json').exists()
    record = dict(passed=True, diagnostic_only=True, goal_complete=False, test_accessed=False,
        manifest_sha256=sha(MANIFEST), snapshot_files=len(manifest['hashes']), verified_hashes=len(seen),
        timing_comparisons=9, counter_files=174, budget=total, simulations=0, new_training_steps=0)
    write_new(RECEIPT, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true')
    if parser.parse_args().register:
        register()
    main()
