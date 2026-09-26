"""Incremental, restorable GitHub release backups, with content verification."""
import datetime as dt, hashlib, json, os, pathlib, sqlite3, subprocess, tarfile, time, urllib.request, urllib.error, uuid
BASE=pathlib.Path('/data/openai-agent');ROOT=BASE/'mobile-robot-mppi-study';STATE=BASE/'state';TMP=BASE/'backup_staging';TMP.mkdir(exist_ok=True)
REPO='Mapples-Frost/mobile-robot-mppi-study';BRANCH='codex/bohn-aws-20260926';TAG='bohn-aws-evidence-20260926';TOKEN=(BASE/'.secrets/github.token').read_text().strip()

def save(p,data):
 t=p.with_suffix('.new');t.write_text(json.dumps(data,indent=2));t.replace(p)
def api(path,method='GET',data=None):
 body=json.dumps(data).encode() if data is not None else None
 req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=body,method=method,headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)

def git(*args):
 env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GIT_ASKPASS=str(ROOT/'scripts/research_service/git_askpass.py'))
 return subprocess.run(['git',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)

def code_backup():
 if not (ROOT/'.git').exists():
  for args in [('init','-b',BRANCH),('config','user.name','Bohn Research Agent'),('config','user.email','bohn-agent@localhost'),('remote','add','origin','https://github.com/'+REPO+'.git')]:
   r=git(*args)
   if r.returncode:raise RuntimeError('Git initialization failed')
 patterns=['experiments/bohn2021_reproduction','scripts/research_service','docs']
 files=[]
 for path in patterns:
  for p in (ROOT/path).rglob('*'):
   if p.is_file() and p.suffix in ('.py','.md','.json','.yaml','.yml','.toml','.txt','.csv','.service','.mount','.ps1','.sh','.patch') and p.stat().st_size<10_000_000 and '__pycache__' not in p.parts:files.append(str(p.relative_to(ROOT)))
 files += [p.name for p in ROOT.glob('*') if p.is_file() and (p.suffix in ('.md','.csv','.json') or p.name=='.gitignore')]
 for start in range(0,len(files),100):
  r=git('add','--',*files[start:start+100])
  if r.returncode:raise RuntimeError('Git staging failed')
 if git('diff','--cached','--quiet').returncode:
  r=git('commit','-m','research: checkpoint audited Bohn autonomous work '+dt.datetime.now(dt.timezone.utc).isoformat())
  if r.returncode:raise RuntimeError('Git commit failed')
 r=git('push','-u','origin',BRANCH)
 if r.returncode:raise RuntimeError('Git push failed (credential details suppressed)')
 return git('rev-parse','HEAD').stdout.strip()

def release():
 try:return api('/releases/tags/'+TAG)
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  return api('/releases','POST',{'tag_name':TAG,'target_commitish':BRANCH,'name':'Bohn research evidence snapshots (AWS 2026-09-26)','body':'Incremental evidence archives. Restore chronological packages over the source snapshot. SHA-256 manifests and package file indices included. Research is ongoing; this release does not assert reproduction success.','draft':False,'prerelease':True})

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def upload(rel,p):
 import http.client,urllib.parse
 u=urllib.parse.urlsplit(rel['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(p.name))
 conn=http.client.HTTPSConnection(u.hostname,timeout=180)
 conn.putrequest('POST',u.path+'?'+u.query);conn.putheader('Authorization','Bearer '+TOKEN);conn.putheader('Content-Type','application/octet-stream');conn.putheader('Content-Length',str(p.stat().st_size));conn.endheaders()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):conn.send(block)
 response=conn.getresponse();data=response.read();status=response.status;conn.close()
 if status!=201:raise RuntimeError('Asset upload HTTP '+str(status))
 asset=json.loads(data);expected='sha256:'+sha(p)
 if asset.get('digest'):
  if asset['digest']!=expected:raise RuntimeError('Remote asset digest mismatch')
  verification='github_server_sha256'
 else:
  req=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':'bohn-backup-verify'})
  h=hashlib.sha256()
  with urllib.request.urlopen(req,timeout=180) as r:
   for block in iter(lambda:r.read(1024*1024),b''):h.update(block)
  if 'sha256:'+h.hexdigest()!=expected:raise RuntimeError('Downloaded asset digest mismatch')
  verification='download_sha256'
 return dict(name=p.name,url=asset['browser_download_url'],id=asset['id'],sha256=expected[7:],bytes=p.stat().st_size,verification=verification)

def files():
 for root in [ROOT/'research_artifacts',ROOT/'source_snapshots',STATE]:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.is_symlink():continue
   if any(x in ('__pycache__','edit_history') for x in p.parts):continue
   if p.suffix in ('.pyc','.lock') or p.name.endswith(('-wal','-shm')) or p.name in ('heartbeat.json','orchestrator.lock','backup_status.json','backup_index.sqlite','backup_receipts.jsonl','research.sqlite'):continue
   yield p

def main():
 source=sqlite3.connect(STATE/'research.sqlite'); target=sqlite3.connect(STATE/'research.snapshot.sqlite'); source.backup(target); target.close(); source.close()
 commit=code_backup();rel=release();con=sqlite3.connect(STATE/'backup_index.sqlite')
 con.execute('create table if not exists files(path text primary key,size integer,mtime integer,sha text,asset text)')
 changed=[]
 for p in files():
  s=p.stat();key=str(p.relative_to(BASE));old=con.execute('select size,mtime from files where path=?',(key,)).fetchone()
  if old!=(s.st_size,s.st_mtime_ns):changed.append((p,key,s.st_size,s.st_mtime_ns))
 packages=[];batch=[];size=0
 def flush(batch):
  if not batch:return
  name=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
  archive=TMP/(name+'.tar.gz');entries=[]
  with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
   for p,key,n,mtime in batch:
    digest=sha(p);tar.add(p,arcname=key,recursive=False)
    if p.stat().st_size!=n or p.stat().st_mtime_ns!=mtime:raise RuntimeError('Source changed during backup: '+key)
    entries.append(dict(path=key,bytes=n,mtime_ns=mtime,sha256=digest))
  asset=upload(rel,archive);manifest=TMP/(name+'.manifest.json');manifest.write_text(json.dumps(dict(commit=commit,asset=asset,entries=entries),indent=2))
  manifest_asset=upload(rel,manifest)
  for row in entries:con.execute('insert or replace into files values(?,?,?,?,?)',(row['path'],row['bytes'],row['mtime_ns'],row['sha256'],asset['name']))
  con.commit();packages.append(asset)
  with (STATE/'backup_receipts.jsonl').open('a') as f:f.write(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),asset=asset,manifest=manifest_asset))+'\n')
  archive.unlink();manifest.unlink() # only verified disposable upload staging files
  save(STATE/'backup_status.json',dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),status='in_progress',commit=commit,packages_this_run=packages,release=rel['html_url']))
 for entry in changed:
  if batch and size+entry[2]>384*1024**2:flush(batch);batch=[];size=0
  batch.append(entry);size+=entry[2]
 flush(batch)
 save(STATE/'backup_status.json',dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),status='verified',commit=commit,changed_files=len(changed),packages_this_run=packages,release=rel['html_url'],tracked_files=con.execute('select count(*) from files').fetchone()[0]))
 print(json.dumps({'backup':'verified','changed_files':len(changed),'packages':len(packages),'release':rel['html_url']}))

if __name__=='__main__':
 try:main()
 except Exception as e:
  # Do not surface credential-bearing HTTP errors.
  save(STATE/'backup_status.json',dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),status='failed',error_type=type(e).__name__,message=str(e).replace(TOKEN,'[REDACTED]')[:1000]));raise SystemExit(1)
