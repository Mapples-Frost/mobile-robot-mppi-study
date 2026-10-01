"""Content-based external recoverability check (Python 3.7+), no credentials/APIs.

Code is proved against the pushed Git ref; raw inputs/checkpoints against the verified
release index. Mutable live logs are not scientific input prerequisites. This replaces
per-wrapper clean-working-tree gates, not the requirement for recoverable evidence.
"""
import hashlib
import json
import pathlib
import re
import sqlite3
import subprocess


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_status(status):
    if status.get('status')!='verified' or status.get('remaining_changed_files')!=0 or not status.get('commit'):
        raise ValueError('External backup is not verified and complete')
    packages=status.get('packages_this_run') or []
    if not packages or any(p.get('verification') not in ('github_server_sha256','download_sha256') or not re.fullmatch('[a-f0-9]{64}',str(p.get('sha256',''))) for p in packages):
        raise ValueError('Verified release package digest proof missing')


def verify(root,state,required_paths):
    root=pathlib.Path(root).resolve();state=pathlib.Path(state)
    status=json.loads((state/'backup_status.json').read_text());validate_status(status)
    def git(*args):return subprocess.run(['git',*args],cwd=str(root),capture_output=True,timeout=30)
    remote=git('rev-parse','@{u}')
    if remote.returncode:raise ValueError('Pushed Git tracking ref unavailable')
    ref=remote.stdout.decode().strip()
    if git('merge-base','--is-ancestor',status['commit'],ref).returncode:
        raise ValueError('Verified backup commit is not present in the pushed history')
    evidence=[]
    with sqlite3.connect('file:'+str(state/'backup_index.sqlite')+'?mode=ro',uri=True,timeout=10) as c:
        for value in sorted(set(str(x) for x in required_paths)):
            path=(root/value).resolve()
            if root not in path.parents or any(x in ('.git','.secrets') for x in path.parts) or not path.is_file():
                raise ValueError('Invalid or missing immutable backup input: '+value)
            rel=str(path.relative_to(root));current=sha(path)
            stored=git('show',ref+':'+rel)
            if stored.returncode==0 and hashlib.sha256(stored.stdout).hexdigest()==current:
                evidence.append(dict(path=rel,sha256=current,proof='pushed_git_blob',commit=ref));continue
            key=root.name+'/'+rel
            row=c.execute('select sha,asset from files where path=?',(key,)).fetchone()
            if row and row[0]==current and row[1]:
                evidence.append(dict(path=rel,sha256=current,proof='verified_release_index',asset=row[1]));continue
            raise ValueError('Current immutable input lacks matching external content proof: '+rel)
    if not evidence:raise ValueError('At least one immutable scientific input must be checked')
    return dict(verified_before_solver_or_plant_steps=True,status='verified',remaining_changed_files=0,
                backup_status_time=status.get('time'),backup_commit=status['commit'],pushed_code_commit=ref,
                packages_this_run=status['packages_this_run'],required_file_proofs=evidence,
                policy='content_identity_for_immutable_inputs; mutable live logs are point-in-time evidence, not clean-tree prerequisites')
