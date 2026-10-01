"""Freeze and export the complete non-secret AWS handoff; no experiments/models."""
import collections, datetime as dt, fcntl, gzip, hashlib, json, os, pathlib, re, sqlite3, subprocess, sys, tempfile, time, urllib.error
BASE=pathlib.Path('/data/openai-agent');ROOT=BASE/'mobile-robot-mppi-study';STATE=BASE/'state';OUT=ROOT/'docs/bohn2021_takeover/local_opus_handoff_20261001';sys.path.insert(0,str(ROOT/'scripts/research_service'))
import backup
STATUS=STATE/'handoff_export_status.json';TAG='bohn-local-opus-handoff-20261001'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,x):
 t=p.with_name(p.name+'.new');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str));t.replace(p)
def phase(value,**extra):save(STATUS,dict(status=value,time=now(),**extra));print(value,flush=True)
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=240)
 if r.returncode:raise RuntimeError('Command failed: '+args[0]+' '+str(r.returncode))
 return r.stdout

def all_source_files():
 for top in [ROOT,STATE,BASE/'runtime',BASE/'flow_optimization_stage']:
  for path,dirs,files in os.walk(top,followlinks=False):
   dirs[:]=[d for d in dirs if d not in ('.git','.venv','.secrets','__pycache__') and not (pathlib.Path(path)/d).is_symlink()]
   for name in sorted(files):
    p=pathlib.Path(path)/name
    if p.is_symlink() or p.suffix in ('.pyc','.lock') or name.endswith(('-wal','-shm')) or name.endswith('.new'):continue
    yield p
 if (BASE/'api_smoke.py').exists():yield BASE/'api_smoke.py'

def secret_values():
 values=[]
 for p in (BASE/'.secrets').iterdir():
  if not p.is_file():continue
  text=p.read_text(errors='replace').strip()
  if p.name=='github.token':values.append(text.encode());continue
  for line in text.splitlines():
   if '=' in line:
    k,v=line.split('=',1)
    if any(w in k.upper() for w in ('KEY','TOKEN','PASSWORD')) and len(v.strip())>=16:values.append(v.strip().strip('"').encode())
 return values

def scan_and_hash(paths):
 import ast
 known=secret_values();pattern=re.compile(rb'sk-[A-Za-z0-9_-]{32,}');results={};hits=[];byte_total=0;cached=0;rescanned=0
 previous=OUT/'SECRET_SCAN_FINDINGS.json';previous_data=json.loads(previous.read_text()) if previous.exists() else {};flagged=set(previous_data.get('paths',[]))
 # The first complete stream scan flagged suffixes inside "risk-..." and six
 # explicitly named fake_secret unit-test fixtures. Actual credential values
 # remain forbidden. Reuse unchanged externally verified file hashes; re-read
 # every previously flagged, changed or newly discovered file in full.
 fixtures=set()
 tree=ast.parse((ROOT/'scripts/research_service/test_astra_routing.py').read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and any(isinstance(t,ast.Attribute) and t.attr=='fake_secret' for t in node.targets):
   value=node.value.value.encode()
   if value not in known:fixtures.add(hashlib.sha256(value).hexdigest())
 con=sqlite3.connect(STATE/'backup_index.sqlite');prior={r[0]:(r[1],r[2],r[3]) for r in con.execute('select path,size,mtime,sha from files')};con.close()
 for i,p in enumerate(paths):
  key=str(p.relative_to(BASE));stat=p.stat();old=prior.get(key)
  if flagged and key not in flagged and old and old[:2]==(stat.st_size,stat.st_mtime_ns):
   results[key]=dict(sha256=old[2],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,proof='unchanged_size_and_mtime_plus_existing_verified_release_sha256_and_completed_stream_secret_scan');cached+=1
  else:
   h=hashlib.sha256();tail=b'';hit=False
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):
     h.update(block);data=tail+block
     if any(v and v in data for v in known):hit=True
     for m in pattern.finditer(data):
      if m.start() and data[m.start()-1] in b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-':continue
      if hashlib.sha256(m.group()).hexdigest() not in fixtures:hit=True
     tail=data[-512:];byte_total+=len(block)
   if hit:hits.append(key)
   results[key]=dict(sha256=h.hexdigest(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,proof='fresh_full_sha256_and_known_secret_and_boundary_aware_long_sk_scan');rescanned+=1
  if i%3000==0:phase('hashing_and_secret_scan',files_done=i,bytes_rescanned=byte_total,verified_unchanged_hashes_reused=cached)
 if hits:
  save(previous,dict(blocked=True,paths=sorted(set(hits)),no_values_recorded=True));raise RuntimeError('Real credential candidate remains; export blocked')
 save(OUT/'SECRET_SCAN_REVIEW.json',dict(time=now(),initial_full_stream_scan_completed=True,initial_flagged_paths=len(flagged),all_initial_flagged_files_reinspected=True,reason='Broad literal scan also matched suffixes of risk-prefixed labels and base64/checksum strings. Boundary-aware review found only explicit fake_secret unit-test fixtures; no exact server credential value.',whitelisted_noncredential_fixture_sha256=sorted(fixtures),actual_secret_matches=0,credentials_excluded=True))
 save(OUT/'SECRET_SCAN_RECEIPT.json',dict(time=now(),files=len(paths),new_changed_or_flagged_files_fully_rehashed=rescanned,bytes_rescanned=byte_total,unchanged_previously_verified_hashes_reused=cached,verification_semantics='Full original stream credential scan plus boundary-aware full rescans of all flagged/changed/new files; SHA256 from fresh reads or unchanged verified release manifest.',actual_credential_hits=0,secret_directories_excluded=True))
 if previous.exists():
  previous_data.update(blocked=False,resolved=True,resolution='All matches were reviewed; noncredential prefixes/checksums and explicit fake_secret fixtures only. No research file modified or omitted.')
  save(previous,previous_data)
 for receipt in OUT.glob('SECRET_SCAN*.json'):
  stat=receipt.stat();results[str(receipt.relative_to(BASE))]=dict(sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,proof='fresh_generated_scan_review_receipt')
 return results

def capture_environment():
 env=OUT/'environment';env.mkdir(exist_ok=True)
 for name in ['bohn-research','bohn-progress-watchdog','bohn-opus-lead','bohn-astra-reviewer']:
  p=pathlib.Path('/etc/systemd/system')/(name+'.service')
  if p.exists():(env/p.name).write_bytes(p.read_bytes())
 for name in ['modern','bohn2021-python37']:
  py=BASE/'runtime'/name/'bin/python'
  (env/(name+'.pip-freeze.txt')).write_text(cmd([str(py),'-m','pip','freeze','--all']))
  (env/(name+'.python-version.txt')).write_text(cmd([str(py),'-V']))
 (env/'ubuntu-release.txt').write_text(pathlib.Path('/etc/os-release').read_text())
 (env/'system-packages.tsv').write_text(cmd(['dpkg-query','-W','-f=${Package}\t${Version}\t${Architecture}\n']))
 (env/'runtime-library-dependencies.txt').write_text(cmd(['ldd',str(BASE/'runtime/bohn2021-python37/bin/python')]))
 links=[]
 for root in [BASE/'runtime',ROOT]:
  for path,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=[d for d in dirs if d not in ('.git','__pycache__')]
   for name in dirs+files:
    p=pathlib.Path(path)/name
    if p.is_symlink():links.append(dict(path=str(p.relative_to(BASE)),target=str(p.readlink())))
 save(env/'SYMLINKS.json',links)
 for name in ['research','task_control','backup_index']:
  source=sqlite3.connect(STATE/(name+'.sqlite'));target=sqlite3.connect(STATE/(name+'.snapshot.sqlite'));source.backup(target)
  if target.execute('pragma integrity_check').fetchone()[0]!='ok':raise RuntimeError('Database snapshot integrity failed: '+name)
  target.close();source.execute('pragma wal_checkpoint(TRUNCATE)');source.close()


def main():
 for attempt in range(360):
  p=STATE/'handoff_author_status.json';status=json.loads(p.read_text()) if p.exists() else {}
  if status.get('status')=='completed':break
  if status.get('status') in ('failed','incomplete'):raise RuntimeError('GPT handoff author requires repair before export')
  phase('waiting_for_gpt55_report');time.sleep(20)
 else:raise RuntimeError('Handoff report wait exceeded')
 if json.loads((STATE/'active_experiment.json').read_text()).get('status') not in ('idle',None):raise RuntimeError('Science still active')
 cmd(['sudo','systemctl','disable','--now','bohn-research','bohn-progress-watchdog','bohn-opus-lead','bohn-astra-reviewer'])
 save(STATE/'LOCAL_OPUS_HANDOFF.json',dict(time=now(),human_requested=True,server_science_frozen=True,server_services_disabled=True,server_complete=False,continuation_owner='Local Opus 5.5',report=str((OUT/'GPT55_HANDOFF_REPORT.md').relative_to(ROOT))))
 capture_environment();phase('inventory')
 source_paths=list(all_source_files());hashes=scan_and_hash(source_paths)
 # Archive everything not covered by Git, including ignored large audit files,
 # current runtimes, edit history, staging source and state. Credentials/caches are excluded.
 phase('pushing_code');commit=backup.code_backup()
 tracked=set(backup.git('ls-files').stdout.splitlines())
 original=backup.files
 def complete_files():
  seen=set()
  for p in original():seen.add(str(p));yield p
  for p in source_paths:
   if str(p) in seen:continue
   key=str(p.relative_to(BASE))
   if p.is_relative_to(ROOT) and str(p.relative_to(ROOT)) in tracked:continue
   if p.name in ('research.sqlite','backup_index.sqlite','backup_receipts.jsonl','backup_status.json','heartbeat.json','handoff_export_status.json'):continue
   seen.add(str(p));yield p
 backup.files=complete_files
 with (STATE/'backup.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for attempt in range(12):
   phase('uploading_evidence',batch=attempt+1)
   with tempfile.TemporaryDirectory(prefix='handoff_snapshot_',dir=backup.TMP) as tmp:backup.main(tmp)
   status=json.loads((STATE/'backup_status.json').read_text())
   if status.get('status')=='verified' and not status.get('remaining_changed_files'):break
  else:raise RuntimeError('Backup did not drain bounded batches')
 phase('building_restore_index')
 con=sqlite3.connect(STATE/'backup_index.sqlite');con.row_factory=sqlite3.Row
 rows=[dict(path=r['path'],bytes=r['size'],mtime_ns=r['mtime'],sha256=r['sha'],asset=r['asset']) for r in con.execute('select * from files order by path')];con.close()
 receipts=[json.loads(x) for x in (STATE/'backup_receipts.jsonl').read_text().splitlines() if x.strip()]
 assets={r['asset']['name']:r['asset'] for r in receipts};needed=set(r['asset'] for r in rows)
 if needed-set(assets):raise RuntimeError('Restore index references assets missing verified upload receipts')
 misses=[];mismatch=[];rowmap={r['path']:r for r in rows}
 for key,info in hashes.items():
  rel=key.removeprefix('mobile-robot-mppi-study/')
  if key.startswith('mobile-robot-mppi-study/') and rel in tracked:
   # Tracked binary files are verified separately below; code hash uses bytes.
   r=subprocess.run(['git','show',commit+':'+rel],cwd=ROOT,capture_output=True,timeout=60)
   if r.returncode or hashlib.sha256(r.stdout).hexdigest()!=info['sha256']:mismatch.append(key)
  elif key in rowmap:
   if key.startswith('state/'):continue # final mutable state captured after initial audit
   if rowmap[key]['sha256']!=info['sha256']:mismatch.append(key)
  elif key.startswith('state/') and pathlib.PurePosixPath(key).name in ('research.sqlite','backup_index.sqlite','backup_receipts.jsonl','backup_status.json','heartbeat.json','handoff_export_status.json'):continue
  else:misses.append(key)
 save(OUT/'COVERAGE_VERIFICATION.json',dict(time=now(),scanned_files=len(hashes),indexed_files=len(rows),missing_paths=misses,hash_mismatches=mismatch,known_exclusions=['credentials','Git internals','Python/pip caches','transient locks/WAL/SHM','disposable upload staging'],sqlite_snapshot_integrity='ok',source_commit=commit))
 if misses or mismatch:raise RuntimeError('Full transfer coverage/hash verification failed; see coverage report')
 index=dict(schema_version=1,handoff_id=TAG,created=now(),code_commit=commit,branch=backup.BRANCH,repository='https://github.com/'+backup.REPO,files=rows,assets=[assets[n] for n in sorted(needed)],restore_semantics='Restore only latest index-selected files from each immutable archive; per-archive and per-file SHA256 required. Clone code at code_commit separately.',exclusions=['secrets','caches','Git object internals','WAL/SHM replaced by consistent SQLite snapshots'])
 index_file=backup.TMP/'HANDOFF_INDEX.json.gz'
 with gzip.open(index_file,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(index,f,separators=(',',':'))
 try:release=backup.api('/releases/tags/'+TAG)
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  release=backup.api('/releases','POST',dict(tag_name=TAG,target_commitish=commit,name='Complete AWS handoff for local Opus 5.5 — 2026-10-01',body='GPT-5.5 authored evidence-grounded handoff. Source and non-secret research state, raw results, checkpoints and runtimes are preserved across immutable evidence releases. HANDOFF_INDEX.json.gz selects the final version of every evidence file. Restore with scripts/research_service/restore_handoff.py. No reproduction success claim. Server research services disabled for local takeover.',prerelease=True,draft=False))
 phase('publishing_handoff_release')
 published=[]
 for p in [index_file,OUT/'GPT55_HANDOFF_REPORT.md',OUT/'OPUS_RESUME_MEMORY.json',OUT/'GPT55_HANDOFF_READY.json',OUT/'COVERAGE_VERIFICATION.json',ROOT/'scripts/research_service/restore_handoff.py']:
  published.append(backup.upload(release,p))
 final=dict(time=now(),status='verified',repository='https://github.com/'+backup.REPO,branch=backup.BRANCH,source_commit=commit,release=release['html_url'],index=published[0],assets=published,restore_files=len(rows),restore_packages=len(needed),uncompressed_bytes=sum(r['bytes'] for r in rows),download_bytes=sum(assets[n]['bytes'] for n in needed),full_file_coverage_verified=True,server_services_disabled=True,no_reproduction_claim=True)
 save(OUT/'DELIVERY_VERIFIED.json',final);save(STATE/'handoff_export_status.json',final)
 backup.code_backup();print(json.dumps(final),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(STATUS,dict(status='failed',error_type=type(e).__name__,message=backup.redact(e),time=now()));print('EXPORT_FAILED',type(e).__name__,flush=True);raise SystemExit(1)
