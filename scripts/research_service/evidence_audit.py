"""Migration evidence audit: no scored sealed-test outcomes are opened."""
import pathlib,json,hashlib,datetime,collections,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2];ART=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17';OUT=ART/'results/latency_tree_2026-09-26'
CACHE={}
def sha(p):
 if str(p) in CACHE:return CACHE[str(p)]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 CACHE[str(p)]=h.hexdigest();return h.hexdigest()
def read(p):return json.loads(p.read_text())
files=list(ART.rglob('*'));counter=collections.Counter();total=0;models=[];manifests=[]
for p in files:
 if p.is_file():
  counter[p.suffix]+=1;total+=p.stat().st_size
  if p.name in ('model.zip','manifest.json'): (models if p.name=='model.zip' else manifests).append(str(p.relative_to(ROOT)))
registration=read(OUT/'registration.json');errors=[]
for name,digest in registration['hashes'].items():
 p=ROOT/name
 if not p.is_file() or sha(p)!=digest:errors.append(name)
completed=[]
for task in ('vehicle','pendulum'):
 for seed in range(3):
  p=OUT/'train'/('%s_s%d'%(task,seed));done=p/'completed.json';mismatches=[]
  if done.exists():
   data=read(done)
   pending=[data];visited=set()
   while pending:
    current=pending.pop()
    for name,digest in current.get('hashes',{}).items():
     src=pathlib.Path(name)
     if not src.exists() or sha(src)!=digest:mismatches.append(name)
     elif src.name=='completed.json' and name not in visited:
      visited.add(name);pending.append(read(src))
  completed.append(dict(task=task,seed=seed,completion_marker=done.exists(),started=(p/'started.json').exists(),hash_mismatches=mismatches,condition_completions=len(list(p.rglob('completed.json')))))
report=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),material_passport={'mode':'audit','source':'D early snapshot + WSL raw files and registered source hashes','verification':'hash and filesystem audit only; no reproduction-success claim'},bytes=total,file_count=sum(counter.values()),suffix_counts=dict(counter),models=models,manifests=manifests,unique_hashed_files=len(CACHE),frozen_files=len(registration['hashes']),registration_mismatches=errors,latest_jobs=completed,sealed_test_outcomes_read=False,original_status_file=read(OUT/'status.json'),live_old_pid=pathlib.Path('/proc/1694525').exists())
dest=ROOT/'docs/bohn2021_takeover/server_evidence_audit.json';dest.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('models','manifests','original_status_file')},indent=2));assert not errors;assert not any(x['hash_mismatches'] for x in completed)
