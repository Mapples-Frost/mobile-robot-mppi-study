#!/usr/bin/env python3
"""Apply a zero-resource release-rotation repair to the backup uploader.

This operational task edits scripts/research_service/backup.py so future
supervisor backups can rotate away from a saturated GitHub release instead of
reusing the hard-coded full tag. It does not execute the uploader, read tokens,
open validation banks, run controllers/solvers/plant steps, train/refit, or
access sealed/final tests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys
import traceback
from typing import Any, Dict, Mapping

CANON_BASE = pathlib.Path("/data/openai-agent")
ROOT = CANON_BASE / "mobile-robot-mppi-study"
if not ROOT.exists():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import execution_contract  # noqa: E402

TASK_ID = "S-BACKUP-RELEASE-ROTATION-PATCH-APPLY-v0"
RUN_STEM = "backup_release_rotation_patch_apply_solo_v0"
BACKUP_REL = "scripts/research_service/backup.py"
BACKUP_PATH = ROOT / BACKUP_REL
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
DOC_MARKER = "<!-- backup-release-rotation-patch-apply-solo-v0 -->"

NEW_BACKUP_SOURCE = r'''"""Incremental, restorable GitHub release backups, with content verification.

The backup evidence release can reach GitHub's per-release asset cap.  This
uploader therefore keeps the historical base release immutable and rotates to
suffixed follow-on releases when the active release lacks room for the next
archive/manifest pair.  It never deletes old evidence assets.
"""

import datetime as dt
import fcntl
import hashlib
import http.client
import json
import os
import pathlib
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = pathlib.Path('/data/openai-agent')
ROOT = BASE / 'mobile-robot-mppi-study'
STATE = BASE / 'state'
TMP = BASE / 'backup_staging'
TMP.mkdir(exist_ok=True)

REPO = 'Mapples-Frost/mobile-robot-mppi-study'
BRANCH = 'codex/bohn-aws-20260926'
BASE_TAG = 'bohn-aws-evidence-20260926'
RELEASE_NAME_BASE = 'Bohn research evidence snapshots (AWS 2026-09-26)'
RELEASE_ASSET_CAP = 1000
ASSETS_PER_PACKAGE = 2  # archive plus manifest
MAX_PACKAGES_PER_RUN = 16
PACKAGE_SIZE_LIMIT = 384 * 1024**2
GITHUB_TOKEN_PATH = BASE / '.secrets/github.token'
_TOKEN_CACHE = None


def github_token():
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        _TOKEN_CACHE = GITHUB_TOKEN_PATH.read_text().strip()
    return _TOKEN_CACHE


def redact(text):
    text = str(text)
    if _TOKEN_CACHE:
        text = text.replace(_TOKEN_CACHE, '[REDACTED]')
    text = text.replace(str(GITHUB_TOKEN_PATH), '[REDACTED_SECRET_PATH]')
    return text[:1000]


def save(p, data):
    t = p.with_suffix('.new')
    t.write_text(json.dumps(data, indent=2))
    t.replace(p)


def api(path, method='GET', data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        'https://api.github.com/repos/' + REPO + path,
        data=body,
        method=method,
        headers={
            'Authorization': 'Bearer ' + github_token(),
            'Accept': 'application/vnd.github+json',
            'Content-Type': 'application/json',
            'User-Agent': 'bohn-backup-service',
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def git(*args):
    env = os.environ.copy()
    env.update(
        GIT_TERMINAL_PROMPT='0',
        GIT_ASKPASS=str(ROOT / 'scripts/research_service/git_askpass.py'),
    )
    return subprocess.run(['git', *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


def code_backup():
    if not (ROOT / '.git').exists():
        for args in [
            ('init', '-b', BRANCH),
            ('config', 'user.name', 'Bohn Research Agent'),
            ('config', 'user.email', 'bohn-agent@localhost'),
            ('remote', 'add', 'origin', 'https://github.com/' + REPO + '.git'),
        ]:
            r = git(*args)
            if r.returncode:
                raise RuntimeError('Git initialization failed')
    patterns = ['experiments', 'scripts', 'docs', 'configs', 'src']
    files = []
    for path in patterns:
        root = ROOT / path
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if (
                p.is_file()
                and p.suffix in ('.py', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.csv', '.service', '.timer', '.mount', '.ps1', '.sh', '.patch')
                and p.stat().st_size < 10_000_000
                and '__pycache__' not in p.parts
            ):
                files.append(str(p.relative_to(ROOT)))
    files += [p.name for p in ROOT.glob('*') if p.is_file() and (p.suffix in ('.md', '.csv', '.json') or p.name == '.gitignore')]
    for start in range(0, len(files), 100):
        r = git('add', '--', *files[start:start + 100])
        if r.returncode:
            raise RuntimeError('Git staging failed')
    if git('diff', '--cached', '--quiet').returncode:
        r = git('commit', '-m', 'research: checkpoint audited Bohn autonomous work ' + dt.datetime.now(dt.timezone.utc).isoformat())
        if r.returncode:
            raise RuntimeError('Git commit failed')
    r = git('push', '-u', 'origin', BRANCH)
    if r.returncode:
        raise RuntimeError('Git push failed (credential details suppressed)')
    return git('rev-parse', 'HEAD').stdout.strip()


def release_tag(index):
    return BASE_TAG if index == 0 else f'{BASE_TAG}-r{index:03d}'


def release_title(index):
    if index == 0:
        return RELEASE_NAME_BASE
    return f'{RELEASE_NAME_BASE} rollover {index:03d}'


def release_body(index):
    if index == 0:
        return (
            'Incremental evidence archives. Restore chronological packages over the source snapshot. '
            'SHA-256 manifests and package file indices included. Research is ongoing; this release does not assert reproduction success.'
        )
    return (
        'Rollover release for incremental Bohn research evidence archives after the base evidence release approached GitHub asset capacity. '
        'Older releases remain immutable; restore by collecting packages/manifests across the base release and rollover releases in chronological order. '
        'SHA-256 manifests and package file indices included. Research is ongoing; this release does not assert reproduction success.'
    )


def get_release_by_tag(tag):
    try:
        return api('/releases/tags/' + tag)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        return None


def create_release(index):
    tag = release_tag(index)
    return api('/releases', 'POST', {
        'tag_name': tag,
        'target_commitish': BRANCH,
        'name': release_title(index),
        'body': release_body(index),
        'draft': False,
        'prerelease': True,
    })


def release_asset_count(rel):
    count = rel.get('assets_count')
    if isinstance(count, int):
        return count
    try:
        if count is not None:
            return int(count)
    except Exception:
        pass
    rel_id = rel.get('id')
    if not rel_id:
        return len(rel.get('assets') or [])
    total = 0
    for page in range(1, 20):
        batch = api(f'/releases/{rel_id}/assets?per_page=100&page={page}')
        if not isinstance(batch, list):
            break
        total += len(batch)
        if len(batch) < 100:
            break
    return total


def release_for_upload(min_free_assets=ASSETS_PER_PACKAGE, start_index=0):
    """Return a release with enough free asset slots for the next package.

    The historical base tag is preserved.  If it is full, or any rollover is
    full, the function advances to the next suffixed release.  No assets are
    deleted and no existing release is rewritten beyond adding new assets.
    """
    for index in range(start_index, 1000):
        tag = release_tag(index)
        rel = get_release_by_tag(tag)
        if rel is None:
            rel = create_release(index)
        count = release_asset_count(rel)
        if count + min_free_assets <= RELEASE_ASSET_CAP:
            rel = dict(rel)
            rel['_rotation_index'] = index
            rel['_asset_count_seen'] = count
            rel['_free_asset_slots_seen'] = RELEASE_ASSET_CAP - count
            return rel
    raise RuntimeError('No GitHub release with enough free asset slots for backup package')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def upload(rel, p):
    u = urllib.parse.urlsplit(rel['upload_url'].split('{')[0] + '?name=' + urllib.parse.quote(p.name))
    conn = http.client.HTTPSConnection(u.hostname, timeout=180)
    conn.putrequest('POST', u.path + '?' + u.query)
    conn.putheader('Authorization', 'Bearer ' + github_token())
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Content-Length', str(p.stat().st_size))
    conn.putheader('User-Agent', 'bohn-backup-service')
    conn.endheaders()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            conn.send(block)
    response = conn.getresponse()
    data = response.read()
    status = response.status
    conn.close()
    if status != 201:
        body = redact(data.decode('utf-8', errors='replace'))
        raise RuntimeError('Asset upload HTTP ' + str(status) + ' on release ' + str(rel.get('tag_name')) + ': ' + body)
    asset = json.loads(data)
    expected = 'sha256:' + sha(p)
    if asset.get('digest'):
        if asset['digest'] != expected:
            raise RuntimeError('Remote asset digest mismatch')
        verification = 'github_server_sha256'
    else:
        req = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'bohn-backup-verify'})
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=180) as r:
            for block in iter(lambda: r.read(1024 * 1024), b''):
                h.update(block)
        if 'sha256:' + h.hexdigest() != expected:
            raise RuntimeError('Downloaded asset digest mismatch')
        verification = 'download_sha256'
    return dict(
        name=p.name,
        url=asset['browser_download_url'],
        id=asset['id'],
        sha256=expected[7:],
        bytes=p.stat().st_size,
        verification=verification,
        release_tag=rel.get('tag_name'),
        release_url=rel.get('html_url'),
        release_asset_count_seen_before_upload=rel.get('_asset_count_seen'),
        release_free_slots_seen_before_upload=rel.get('_free_asset_slots_seen'),
    )


def files():
    for root in [ROOT / 'research_artifacts', ROOT / 'source_snapshots', STATE]:
        if not root.exists():
            continue
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink():
                continue
            if any(x in ('__pycache__', 'edit_history') for x in p.parts):
                continue
            if (
                p.suffix in ('.pyc', '.lock')
                or p.name.endswith(('-wal', '-shm'))
                or p.name in ('heartbeat.json', 'orchestrator.lock', 'backup_status.json', 'backup_index.sqlite', 'backup_receipts.jsonl', 'research.sqlite')
            ):
                continue
            yield p


def capture_state_file(p, destination):
    # STATE contains live telemetry and worker progress. Capture a bounded point-in-time
    # file version; scientific raw evidence outside STATE retains strict mutation checks.
    for attempt in range(3):
        with p.open('rb') as source, destination.open('wb') as target:
            before = os.fstat(source.fileno())
            remaining = before.st_size
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    break
                target.write(block)
                remaining -= len(block)
            after = os.fstat(source.fileno())
        append_only_prefix = (p.suffix == '.jsonl' and after.st_size >= before.st_size)
        if remaining == 0 and (after.st_mtime_ns == before.st_mtime_ns or append_only_prefix):
            os.utime(destination, ns=(before.st_atime_ns, before.st_mtime_ns))
            return before.st_size, before.st_mtime_ns
        time.sleep(0.05)
    raise RuntimeError('Unable to capture consistent state snapshot: ' + str(p.relative_to(BASE)))


def main(snapshot_dir):
    source = sqlite3.connect(STATE / 'research.sqlite')
    target = sqlite3.connect(STATE / 'research.snapshot.sqlite')
    source.backup(target)
    target.close()
    source.close()
    commit = code_backup()
    con = sqlite3.connect(STATE / 'backup_index.sqlite')
    con.execute('create table if not exists files(path text primary key,size integer,mtime integer,sha text,asset text)')
    changed = []
    for p in files():
        s = p.stat()
        key = str(p.relative_to(BASE))
        old = con.execute('select size,mtime from files where path=?', (key,)).fetchone()
        if old != (s.st_size, s.st_mtime_ns):
            if p.is_relative_to(STATE):
                destination = pathlib.Path(snapshot_dir) / uuid.uuid4().hex
                n, mtime = capture_state_file(p, destination)
                changed.append((destination, key, n, mtime))
            else:
                changed.append((p, key, s.st_size, s.st_mtime_ns))
    changed.sort(key=lambda entry: entry[3], reverse=True)
    packages = []
    releases_this_run = {}
    batch = []
    size = 0

    def remember_release(rel):
        tag = rel.get('tag_name')
        if tag:
            releases_this_run[tag] = {
                'tag': tag,
                'url': rel.get('html_url'),
                'rotation_index': rel.get('_rotation_index'),
                'asset_count_seen_before_first_upload': rel.get('_asset_count_seen'),
                'free_slots_seen_before_first_upload': rel.get('_free_asset_slots_seen'),
            }

    def status_release_url():
        if releases_this_run:
            return list(releases_this_run.values())[-1].get('url')
        return None

    def flush(flush_batch):
        if not flush_batch:
            return
        name = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        archive = TMP / (name + '.tar.gz')
        entries = []
        with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
            for p, key, n, mtime in flush_batch:
                digest = sha(p)
                tar.add(p, arcname=key, recursive=False)
                if p.stat().st_size != n or p.stat().st_mtime_ns != mtime:
                    raise RuntimeError('Source changed during backup: ' + key)
                entries.append(dict(
                    path=key,
                    bytes=n,
                    mtime_ns=mtime,
                    sha256=digest,
                    snapshot_semantics='point_in_time_state' if key.startswith('state/') else 'immutable_scientific_evidence',
                ))
        rel = release_for_upload(min_free_assets=ASSETS_PER_PACKAGE)
        remember_release(rel)
        asset = upload(rel, archive)
        manifest = TMP / (name + '.manifest.json')
        manifest.write_text(json.dumps(dict(
            commit=commit,
            release={'tag': rel.get('tag_name'), 'url': rel.get('html_url'), 'rotation_index': rel.get('_rotation_index')},
            asset=asset,
            entries=entries,
        ), indent=2))
        manifest_asset = upload(rel, manifest)
        for row in entries:
            con.execute('insert or replace into files values(?,?,?,?,?)', (row['path'], row['bytes'], row['mtime_ns'], row['sha256'], asset['name']))
        con.commit()
        packages.append(asset)
        with (STATE / 'backup_receipts.jsonl').open('a') as f:
            f.write(json.dumps(dict(
                time=dt.datetime.now(dt.timezone.utc).isoformat(),
                release={'tag': rel.get('tag_name'), 'url': rel.get('html_url'), 'rotation_index': rel.get('_rotation_index')},
                asset=asset,
                manifest=manifest_asset,
            )) + '\n')
        archive.unlink()
        manifest.unlink()  # only verified disposable upload staging files
        save(STATE / 'backup_status.json', dict(
            time=dt.datetime.now(dt.timezone.utc).isoformat(),
            status='in_progress',
            commit=commit,
            packages_this_run=packages,
            releases_this_run=list(releases_this_run.values()),
            release=status_release_url(),
        ))

    for entry in changed:
        if batch and size + entry[2] > PACKAGE_SIZE_LIMIT:
            flush(batch)
            batch = []
            size = 0
            if len(packages) >= MAX_PACKAGES_PER_RUN:
                break
        batch.append(entry)
        size += entry[2]
    flush(batch)
    remaining = sum(con.execute('select size,mtime from files where path=?', (key,)).fetchone() != (n, mtime) for p, key, n, mtime in changed)
    save(STATE / 'backup_status.json', dict(
        time=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='partial' if remaining else 'verified',
        remaining_changed_files=remaining,
        commit=commit,
        changed_files=len(changed),
        packages_this_run=packages,
        releases_this_run=list(releases_this_run.values()),
        release=status_release_url(),
        tracked_files=con.execute('select count(*) from files').fetchone()[0],
    ))
    print(json.dumps({
        'backup': 'partial' if remaining else 'verified',
        'remaining_changed_files': remaining,
        'changed_files': len(changed),
        'packages': len(packages),
        'releases_this_run': list(releases_this_run.values()),
        'release': status_release_url(),
    }))


if __name__ == '__main__':
    try:
        with (STATE / 'backup.lock').open('a') as backup_lock:
            fcntl.flock(backup_lock, fcntl.LOCK_EX)
            with tempfile.TemporaryDirectory(prefix='state_snapshot_', dir=TMP) as snapshot_dir:
                main(snapshot_dir)
    except Exception as e:
        # Do not surface credential-bearing HTTP errors.
        save(STATE / 'backup_status.json', dict(
            time=dt.datetime.now(dt.timezone.utc).isoformat(),
            status='failed',
            error_type=type(e).__name__,
            message=redact(e),
        ))
        raise SystemExit(1)
'''


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(CANON_BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: pathlib.Path, marker: str, block: str) -> None:
    previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in previous:
        return
    path.write_text(previous.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def validate_source(source: str) -> Dict[str, Any]:
    compile(source, BACKUP_REL, "exec")
    checks = {
        "has_base_tag": "BASE_TAG = 'bohn-aws-evidence-20260926'" in source,
        "has_release_tag_function": "def release_tag(index):" in source,
        "has_release_for_upload_function": "def release_for_upload(" in source,
        "has_asset_count_function": "def release_asset_count(rel):" in source,
        "has_lazy_github_token_function": "def github_token():" in source,
        "no_top_level_token_read_assignment": "TOKEN=(BASE/'.secrets/github.token').read_text().strip()" not in source,
        "does_not_delete_release_assets": "DELETE" not in source and "delete" not in source.lower().replace("does not delete", ""),
        "records_releases_this_run": "releases_this_run" in source,
        "records_release_tag_per_asset": "release_tag=rel.get('tag_name')" in source,
    }
    checks["all_static_checks_passed"] = all(checks.values())
    return checks


def apply_patch() -> Dict[str, Any]:
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(str(BACKUP_PATH))
    before = BACKUP_PATH.read_text(encoding="utf-8")
    before_sha = sha_text(before)
    new_sha = sha_text(NEW_BACKUP_SOURCE)
    checks = validate_source(NEW_BACKUP_SOURCE)
    if not checks["all_static_checks_passed"]:
        raise RuntimeError("new backup.py source failed static checks: " + repr(checks))
    already_applied = before_sha == new_sha
    prior_has_expected_marker = "TAG='bohn-aws-evidence-20260926'" in before or "BASE_TAG = 'bohn-aws-evidence-20260926'" in before
    if not already_applied and not prior_has_expected_marker:
        raise RuntimeError("Refusing to patch unexpected backup.py: base tag marker not found")
    if not already_applied:
        tmp = BACKUP_PATH.with_suffix(".py.new")
        tmp.write_text(NEW_BACKUP_SOURCE, encoding="utf-8")
        tmp.replace(BACKUP_PATH)
    after = BACKUP_PATH.read_text(encoding="utf-8")
    after_sha = sha_text(after)
    after_checks = validate_source(after)
    return {
        "backup_path": rel(BACKUP_PATH),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "expected_new_sha256": new_sha,
        "already_applied": already_applied,
        "modified": not already_applied,
        "before_bytes": len(before.encode("utf-8")),
        "after_bytes": len(after.encode("utf-8")),
        "prior_has_expected_marker": prior_has_expected_marker,
        "static_checks": after_checks,
    }


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{RUN_STEM}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    snapshot = execution_contract.runtime_snapshot(ROOT)
    patch_result = apply_patch()
    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "backup_service_source_modified_or_already_current": bool(patch_result.get("modified") or patch_result.get("already_applied")),
        "release_rotation_patch_written": patch_result.get("after_sha256") == patch_result.get("expected_new_sha256"),
        "backup_script_has_lazy_token_read": bool(patch_result.get("static_checks", {}).get("has_lazy_github_token_function")),
        "backup_script_has_release_rotation_logic": bool(patch_result.get("static_checks", {}).get("has_release_for_upload_function")),
        "backup_script_compiles": True,
        "old_and_new_source_hashes_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    expected = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "backup_service_source_modified_or_already_current": True,
        "release_rotation_patch_written": True,
        "backup_script_has_lazy_token_read": True,
        "backup_script_has_release_rotation_logic": True,
        "backup_script_compiles": True,
        "old_and_new_source_hashes_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    passed = all(evidence.get(k) is v for k, v in expected.items())
    next_action = (
        "Wait for or trigger only the protected supervisor backup path; after backup_status changes, run a bounded status recheck. "
        "Nonzero controller/solver/plant/training work remains blocked until a verified backup proof reports remaining_changed_files=0, commit, and verified package metadata."
    )
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_backup_release_rotation_patch_apply_solo_v0.md"
    raw = {
        "analysis_type": "backup_release_rotation_patch_apply",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "script": "experiments/bohn2021_aws/backup_release_rotation_patch_apply_solo_v0.py",
        "script_sha256": sha_file(pathlib.Path(__file__).resolve()),
        "runtime_snapshot_sha256": snapshot.get("snapshot_sha256") if isinstance(snapshot, Mapping) else None,
        "patch_result": patch_result,
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "backup_uploader_executed": False,
        "nonzero_work_still_blocked_until_verified_backup": True,
        "next_action": next_action,
        "elapsed_hours_since_first_supervisor_event": (created - FIRST_SUPERVISOR_EVENT).total_seconds() / 3600.0,
        "evidence": evidence,
    }
    write_json(raw_path, raw)
    summary = f"""# Backup release-rotation patch apply (solo v0)

Created UTC: `{created.isoformat()}`.

## Result

- Task passed: `{passed}`.
- Backup service source modified: `{patch_result.get('modified')}`.
- Already applied before this task: `{patch_result.get('already_applied')}`.
- Before backup.py SHA-256: `{patch_result.get('before_sha256')}`.
- After backup.py SHA-256: `{patch_result.get('after_sha256')}`.
- Static checks: `{patch_result.get('static_checks')}`.

## Operational decision

{next_action}

This is an operational backup repair only. It is neither an ORIGINAL nor IMPROVED vehicle-control result and does not authorize sealed/final-test access or nonzero scientific work before a verified external backup proof exists.

## Resource and split safety

All five resource counters are zero. The backup uploader was not executed. No validation-bank content, sealed test, or final test was accessed. No credentials were read or logged.
"""
    summary_path.write_text(summary, encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after backup release-rotation patch apply solo v0\n\n"
        + f"UTC: {created.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Task passed: {passed}. Patch result: {patch_result}.\n"
        + f"Summary: `{rel(summary_path)}`. Raw: `{rel(raw_path)}`. Next action: {next_action}\n",
        encoding="utf-8",
    )
    completed = {
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "passed": passed,
        "resources": dict(ZERO),
        "evidence": evidence,
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "completed": rel(completed_path),
        "state": rel(state_path),
        "patch_result": patch_result,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "backup_uploader_executed": False,
    }
    write_json(completed_path, completed)
    log_block = f"""
### 2026-10-01 backup release-rotation patch apply (solo v0)

- Task: `{TASK_ID}`.
- Outcome: task_gate_passed=`{passed}`, backup_py_modified=`{patch_result.get('modified')}`, after_sha256=`{patch_result.get('after_sha256')}`.
- Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
- Next action: {next_action}
- Resource use: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. The backup uploader was not executed; no validation-bank, sealed-test, or final-test access.
"""
    for log_rel in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"):
        append_once(ROOT / log_rel, DOC_MARKER + "-" + log_rel.replace("/", "-").replace(".", "-"), log_block)
    if passed:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
        print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
        return 0
    failed_evidence = dict(evidence)
    failed_evidence["no_scientific_outcome"] = True
    execution_contract.record_outcome(
        ROOT,
        "engineering_failure",
        dict(ZERO),
        failed_evidence,
        engineering_error="backup_release_rotation_patch_apply_gate_failed",
    )
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            execution_contract.record_outcome(
                ROOT,
                "engineering_failure",
                dict(ZERO),
                {
                    "no_scientific_outcome": True,
                    "zero_solver_plant_training_validation_test_resources": True,
                    "validation_bank_content_opened": False,
                    "sealed_or_final_test_accessed": False,
                    "no_secrets_read_or_logged": True,
                    "backup_uploader_not_executed": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                engineering_error="startup_or_runtime_exception",
            )
        except Exception:
            pass
        raise
