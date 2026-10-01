"""Incremental, restorable GitHub release backups, with content verification.

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
