"""Post-campaign integrity closure and complete local inventory; not goal acceptance.

No production training, plotting, or effect modules are imported. Existing hash
edges in this campaign are checked recursively. Historical/external JSON files
are hashed leaves, not an implicit rerun of their historical audits. A newly
frozen file without an earlier digest is labelled as such, never called evidence
that the file was unchanged during the experiment. No files are deleted/copied
out of the campaign, and no sealed historical test outcomes are opened.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26'
SOURCE = Path(__file__).resolve()
REG = OUT / 'package_audit_registration.json'
SKILL = Path('/mnt/c/Users/lenovo/.codex/skills/scipilot-figure-skill/scripts')
MAP_FIELDS = frozenset(('hashes', 'input_hashes', 'output_hashes', 'source_hashes',
    'raw_completion_hashes', 'skill_hashes', 'source_and_requirement_hashes', 'dependency_hashes'))
DIGEST = re.compile(r'^[0-9a-f]{64}$')
TEST_PARTS = (('evaluations', 'test'), ('timing_test',), ('test_delivery',), ('test_figures',))
RAW_ROOTS = ('bank_generation', 'smoke', 'train', 'evaluation_smoke', 'evaluations',
             'timing_validation', 'timing_test', 'extra_fixed')
NO_EDGE_JSON = re.compile(r'^(?:r\d+_(?:trace|reset)_\d+|attempt_\d+_\d+)\.json$')
SUPERSEDED = frozenset(('native_pipeline_registration.json', 'native_pipeline_launch.json',
                        'native_pipeline_status.json'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def below(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def frozen(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, ('Existing artifact differs', str(path))
    else:
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(text)


def test_outcome_path(path):
    parts = path.parts
    return any(any(parts[i:i + len(token)] == token for i in range(len(parts))) for token in TEST_PARTS)


def inventory_only(path, campaign):
    relative = path.relative_to(campaign)
    return (any('preparation' in part or 'amendment' in part for part in relative.parts)
            or len(relative.parts) == 1 and path.name in SUPERSEDED
            or path.name.endswith('_template.json'))


class Closure:
    def __init__(self, root, campaign, allowed_test=False, external_roots=()):
        self.root = root.resolve()
        self.campaign = campaign.resolve()
        self.allowed_test = allowed_test
        self.external_roots = tuple(p.resolve() for p in external_roots)
        self.expected, self.verified, self.edges = {}, {}, []
        self.parsed, self.leaves = set(), set()
        self.raw_test_bytes_hashed = False

    def path(self, name):
        assert isinstance(name, str) and name and '\x00' not in name
        p = Path(name)
        p = (p if p.is_absolute() else self.root / p).resolve()
        assert below(p, self.root) or any(below(p, r) for r in self.external_roots), ('Outside allowed roots', str(p))
        if test_outcome_path(p):
            assert self.allowed_test and below(p, self.campaign), ('Sealed test outcome reference', str(p))
        return p

    def hash(self, path):
        path = self.path(str(path))
        assert path.is_file(), ('Missing file', str(path))
        if path not in self.verified:
            before = path.stat()
            value = digest(path)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), ('File changed while hashing', str(path))
            self.verified[path] = (value, after.st_size, after.st_mtime_ns)
            if test_outcome_path(path):
                self.raw_test_bytes_hashed = True
        return self.verified[path][0]

    def edge(self, owner, field, name, wanted):
        assert isinstance(wanted, str) and DIGEST.fullmatch(wanted), ('Invalid SHA256', str(owner), field, name)
        target = self.path(name)
        if target in self.expected:
            assert self.expected[target] == wanted, ('Conflicting historical digests', str(target))
        self.expected[target] = wanted
        assert self.hash(target) == wanted, ('Digest mismatch', str(target), str(owner), field)
        self.edges.append(dict(owner=str(owner), field=field, target=str(target), sha256=wanted))
        self.visit(target)

    def visit(self, path):
        path = self.path(str(path))
        self.hash(path)
        if path.suffix != '.json' or not below(path, self.campaign):
            self.leaves.add(path)
            return
        if path in self.parsed:
            return
        self.parsed.add(path)
        # Banks and raw data carry no file-reference edges; do not deserialize them.
        if (inventory_only(path, self.campaign)
                or 'banks' in path.relative_to(self.campaign).parts and path.name != 'completed.json'
                or NO_EDGE_JSON.fullmatch(path.name)):
            self.leaves.add(path)
            return
        value = read(path)
        if not isinstance(value, dict):
            self.leaves.add(path)
            return
        for field in MAP_FIELDS:
            if field not in value:
                continue
            mapping = value[field]
            assert isinstance(mapping, dict), ('Unrecognized reference map', str(path), field)
            for name, wanted in mapping.items():
                self.edge(path, field, name, wanted)
        # Scalar links in current post-processing manifests have fixed locations.
        scalar = {}
        if path.name in ('export_manifest.json', 'preview_manifest.json'):
            figure_root = path.parent.parent
            scalar.update(data_manifest_sha256=figure_root / 'data_manifest.json',
                          profile_review_sha256=figure_root / 'profile_review.json')
            if path.name == 'export_manifest.json':
                scalar['visual_review_sha256'] = figure_root / ('preview_%02d' % value['revision']) / 'visual_review.json'
        if path.name == 'visual_review.json':
            scalar['preview_manifest_sha256'] = path.parent / 'preview_manifest.json'
        if path.name == 'profile_review.json':
            scalar['data_manifest_sha256'] = path.parent / 'data_manifest.json'
        for field, target in scalar.items():
            assert field in value, ('Missing scalar reference', str(path), field)
            self.edge(path, field, str(target), value[field])

    def assert_unchanged(self):
        for path, (_, size, modified) in self.verified.items():
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (size, modified), ('Changed during audit', str(path))


def idle():
    states = [read(OUT / n) for n in ('status.json', 'native_pipeline_v2_status.json', 'budget_finish_status.json')]
    assert all(s['complete'] is True and s['active'] is False for s in states), 'Campaign still running/incomplete'
    assert states[1]['stage'] in ('negative_validation_test_sealed', 'confirmation_review_complete_delivery_pending')
    assert states[2]['stage'] == 'budget_complete_final_delivery_pending'
    assert Path('/proc').is_dir(), 'Use the WSL audit interpreter'
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            args = p.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        assert not (args and 'python' in Path(args[0]).name and any(
            'experiments/' in a and a.endswith('.py') for a in args[1:])), ('Concurrent experiment/audit Python', args)
    pids = [states[1]['pid'], states[2]['pid']]
    assert all(type(p) is int and p > 0 for p in pids)
    command = ('$ErrorActionPreference="Stop"; Get-CimInstance Win32_Process -Filter "ProcessId=%d OR ProcessId=%d" | '
               'Where-Object { $_.CommandLine -match "latency_tree_(native_pipeline|budget_finish)\\.ps1" } | '
               'ForEach-Object { $_.ProcessId }') % tuple(pids)
    result = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-Command', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 0 and not result.stdout.strip(), 'Cannot confirm native controller exit'
    for state, filename in ((states[1], 'native_pipeline_v2_registration.json'),
                            (states[2], 'budget_finish_registration.json')):
        assert state['registration_sha256'] == digest(OUT / filename)
    budget = read(OUT / 'final_budget/independent_review.json')
    assert budget['passed'] is True
    return states


def register():
    idle()
    value = dict(source_hashes={str(SOURCE): digest(SOURCE)}, checks_pending=True,
        existing_scientific_registrations_modified=False, simulations=0, test_unlock=False,
        full_historical_audit_rerun=False, goal_complete=False)
    frozen(REG, value)
    return dict(registered=True, registration_sha256=digest(REG), checks_pending=True)


def ready():
    idle()
    assert read(REG)['source_hashes'] == {str(SOURCE): digest(SOURCE)}


def check():
    ready()
    cases = []
    def reject(label, function):
        try:
            function()
        except (AssertionError, FileNotFoundError):
            cases.append(label)
        else:
            raise AssertionError(('Did not reject', label))
    with tempfile.TemporaryDirectory(prefix='latency_package_') as directory:
        root = Path(directory).resolve()
        campaign = root / 'campaign'
        campaign.mkdir()
        leaf = campaign / 'raw.bin'
        leaf.write_bytes(b'complete raw evidence')
        child = campaign / 'child.json'
        frozen(child, dict(hashes={str(leaf): digest(leaf)}))
        parent = campaign / 'parent.json'
        frozen(parent, dict(input_hashes={'campaign/child.json': digest(child)}))
        graph = Closure(root, campaign)
        graph.visit(parent)
        assert set(graph.expected) == {child, leaf}
        graph.assert_unchanged()
        cases.append('recursive_relative_and_absolute_edges')
        graph.edge(parent, 'duplicate', 'campaign/raw.bin', digest(leaf))
        cases.append('duplicate_alias_same_digest')
        reject('conflicting_digest', lambda: graph.edge(parent, 'bad', str(leaf), '0' * 64))
        reject('invalid_digest', lambda: Closure(root, campaign).edge(parent, 'bad', str(leaf), 'not a digest'))
        reject('missing_file', lambda: Closure(root, campaign).edge(parent, 'bad', 'missing.bin', '0' * 64))
        reject('path_escape', lambda: graph.path('../outside.bin'))
        reject('current_sealed_test', lambda: graph.path(str(campaign / 'evaluations/test/raw.json')))
        allowed = Closure(root, campaign, allowed_test=True)
        assert allowed.path(str(campaign / 'evaluations/test/raw.json')).parent.name == 'test'
        cases.append('explicit_current_test_authorization')
        reject('historical_test_still_sealed', lambda: allowed.path(str(root / 'old/evaluations/test/raw.json')))
        external = root / 'historical.json'
        frozen(external, dict(hashes={'missing_historical_file': '0' * 64}))
        allowed.visit(external)
        assert external in allowed.leaves and external not in allowed.parsed
        cases.append('historical_boundary_explicit')
        reject('immutable_write', lambda: frozen(child, dict(changed=True)))
        leaf.write_bytes(b'corrupted')
        reject('raw_corruption', lambda: Closure(root, campaign).visit(parent))
        reject('mutation_during_audit', graph.assert_unchanged)
        unknown = campaign / 'unknown.json'
        frozen(unknown, dict(hashes=['not a map']))
        reject('unrecognized_map', lambda: Closure(root, campaign).visit(unknown))
    record = dict(passed=True, cases=cases, count=len(cases), simulations=0,
                  source_hashes={str(SOURCE): digest(SOURCE)}, test_outcomes_read=False, goal_complete=False)
    frozen(OUT / 'package_audit_checks.json', record)
    return record


def verify_superseded_controller(graph):
    """Validate archived v1 sources against v1, without comparing them with v2."""
    archive = OUT / 'delivery_order_amendment'
    amendment_path = archive / 'amendment.json'
    amendment = read(amendment_path)
    registration_path = archive / 'native_pipeline_registration.json'
    old = read(registration_path)
    expected = amendment['old_registration_sha256']
    assert digest(registration_path) == digest(OUT / 'native_pipeline_registration.json') == expected
    stopped = read(archive / 'stop_receipt.json')
    assert stopped['old_registration_sha256'] == expected
    assert stopped['completed_stages'] == stopped['worker_children'] == 0 and stopped['training_untouched']
    retired = read(OUT / 'native_pipeline_status.json')
    assert retired['active'] is False and retired['stage'] == 'superseded_before_any_child_stage'
    assert retired['registration_sha256'] == expected and not retired['completed_stages']
    graph.edge(amendment_path, 'old_registration_sha256', str(registration_path), expected)
    graph.edge(amendment_path, 'old_registration_sha256', str(OUT / 'native_pipeline_registration.json'), expected)
    for name, wanted in old['source_hashes'].items():
        source = archive / Path(name).name if Path(name).name in amendment['changed_sources'] else ROOT / name
        graph.edge(registration_path, 'preserved_v1_source_hashes', str(source), wanted)
    return dict(preserved_v1_sources=len(old['source_hashes']), stopped_before_any_child_stage=True,
                historical_registration_sha256=expected)


def audit(folder, snapshot):
    ready()
    folder = folder.resolve()
    assert folder.parent == OUT and re.fullmatch(r'final_delivery_[0-9]+', folder.name)
    assert snapshot > 0
    checks = read(OUT / 'package_audit_checks.json')
    assert checks['passed'] is True and checks['source_hashes'] == {str(SOURCE): digest(SOURCE)}
    table_review = read(folder / 'independent_table_artifact_review.json')
    assert table_review['passed_report_tables'] is True
    report = read(folder / 'manifest.json')
    assert report['generated'] is True and report['goal_complete'] is False
    gate_path = OUT / 'validation_delivery/effect_gate.json'
    gate = read(gate_path)
    review = read(OUT / 'validation_delivery/independent_review.json')
    assert review['passed'] is True and review['effect_passed'] == gate['effect_passed']
    assert review['hashes'][str(gate_path)] == digest(gate_path)
    test_open = gate['effect_passed'] is True
    assert report['confirmation_run'] == table_review['confirmation_run'] == test_open
    if test_open:
        confirm = read(OUT / 'confirmation_registration.json')
        assert confirm['validation_gate_passed'] and confirm['independent_review_passed']
        test_review = read(OUT / 'test_delivery/independent_review.json')
        assert test_review['passed'] is True
    else:
        assert not any((OUT / name).exists() for name in (
            'confirmation_registration.json', 'evaluations/test', 'timing_test', 'test_delivery', 'test_figures'))
    dest = OUT / ('package_integrity_%02d' % snapshot)
    assert not dest.exists(), 'Preserve prior package audits; use a new snapshot'
    graph = Closure(ROOT, OUT, allowed_test=test_open, external_roots=(SKILL,))
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and not any(part.startswith('package_integrity_') for part in p.relative_to(OUT).parts)
                   and p.name != 'run.lock' and p.suffix != '.pyc')
    assert not any(p.suffix == '.tmp' for p in paths), 'Unfinished temporary write requires inspection'
    # The selected final delivery is the root of the auditable scientific evidence.
    for p in (REG, OUT / 'package_audit_checks.json', folder / 'manifest.json',
              folder / 'independent_table_artifact_review.json', OUT / 'registration.json',
              OUT / 'native_pipeline_v2_registration.json', OUT / 'budget_finish_registration.json'):
        graph.visit(p)
    amendment = verify_superseded_controller(graph)
    # Current metadata is inspected for registered references; old preparation or
    # amendment snapshots are inventory-only because they describe superseded files.
    for p in paths:
        if not inventory_only(p, OUT):
            graph.visit(p)
        else:
            graph.hash(p)
    raw_roots = [OUT / n for n in RAW_ROOTS if (OUT / n).exists()]
    raw_paths = [p for p in paths if any(below(p, r) for r in raw_roots)
                 and (p.suffix == '.jsonl' or NO_EDGE_JSON.fullmatch(p.name)
                      or p.name in ('model.zip', 'summary.json', 'solver_attempts.json'))]
    assert raw_paths and all(p in graph.expected for p in raw_paths), (
        'Raw files missing prior recorded hashes', [str(p) for p in raw_paths if p not in graph.expected])
    # Source archive manifests use a list schema, handled explicitly.
    archives = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/sources/archives/manifest.json'
    for record in read(archives):
        graph.edge(archives, 'archive', str(archives.parent / record['file']), record['sha256'])
    graph.assert_unchanged()
    rows = []
    for p in paths:
        value, size, _ = graph.verified[p]
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=size, sha256=value,
                         matched_existing_digest=p in graph.expected,
                         provenance='prior_hash_verified' if p in graph.expected else 'first_frozen_at_package_audit'))
    edges = sorted(graph.edges, key=lambda row: (row['owner'], row['field'], row['target']))
    external = sorted(p for p in graph.verified if not below(p, OUT))
    dest.mkdir()
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with (dest / 'campaign_inventory.csv').open('x', encoding='utf-8', newline='') as stream:
        stream.write(buffer.getvalue())
    frozen(dest / 'reference_edges.json', edges)
    frozen(dest / 'external_leaves.json', [dict(path=str(p), sha256=graph.verified[p][0],
        bytes=graph.verified[p][1], historical_semantics_reaudited=False) for p in external])
    result = dict(passed_current_campaign_integrity=True, selected_delivery=str(folder),
        campaign_files=len(rows), prior_digest_verified_files=sum(r['matched_existing_digest'] for r in rows),
        newly_frozen_files=sum(not r['matched_existing_digest'] for r in rows),
        registered_reference_edges=len(edges), verified_raw_files=len(raw_paths), external_leaves=len(external),
        campaign_bytes=sum(r['bytes'] for r in rows), json_metadata_parsed=len(graph.parsed),
        test_confirmation_run=test_open, current_test_bytes_hashed=graph.raw_test_bytes_hashed,
        raw_test_trajectories_deserialized=False, historical_test_outcomes_accessed=False,
        excluded_inventory=['run.lock', '__pycache__', '*.pyc', 'previous package_integrity_* audit outputs'],
        historical_receipts_are_hashed_leaves=True, complete_historical_transitive_audit=False,
        external_dependencies_bundled=False, fresh_environment_reproduced=False,
        scientific_semantic_review_complete=False, new_visual_review_performed=False,
        superseded_controller_preservation=amendment,
        goal_complete=False, simulations=0, source_hashes={str(SOURCE): digest(SOURCE)},
        input_hashes={str(folder / 'manifest.json'): digest(folder / 'manifest.json'),
                     str(folder / 'independent_table_artifact_review.json'): digest(folder / 'independent_table_artifact_review.json'),
                     str(REG): digest(REG), str(OUT / 'package_audit_checks.json'): digest(OUT / 'package_audit_checks.json')},
        output_hashes={str(p): digest(p) for p in dest.iterdir() if p.is_file()},
        scope='Current campaign inventory and registered file-reference integrity only. '
              'No recalculation of scientific effects or historical audit rerun; no fresh environment or portable bundle claim. '
              'Newly frozen files have no earlier integrity evidence and are labelled explicitly.')
    frozen(dest / 'report.json', result)
    return {k: v for k, v in result.items() if not k.endswith('hashes')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('register', 'check', 'audit'), required=True)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--snapshot', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'audit':
        assert args.folder is not None
        answer = audit(args.folder, args.snapshot)
    else:
        answer = register() if args.mode == 'register' else check()
    print(json.dumps(answer, ensure_ascii=False, indent=2))
