"""Read-only compact experiment facts, rebuilt from the authoritative registry."""
import datetime as dt
import json
import pathlib
import re
import sqlite3
DENY=re.compile(r'sealed|final[_-]?test|test[_-]?(results|scenarios)|\.secrets|\.env|\.pem|\.key',re.I)
FIELDS=('script','purpose','method','seed','split','exit_status','runtime_seconds','training_budget','validation_budget','test_budget')

def protected_split(meta):
    split=str(meta.get('split','')).strip().lower()
    if split in ('test','final','sealed','sealed_test','sealed-test','final_test','final-test'):return True
    budget=meta.get('test_budget') or {}
    numeric=[v for v in budget.values() if isinstance(v,(int,float,bool))]
    if any(v>0 for v in numeric):return True
    if not DENY.search(split):return False
    # Free-text development split names often literally say "no sealed/final test".
    # Accept that explicit exclusion only with a development prefix and zero test budget;
    # raw sealed/final paths remain separately blocked by each analyst's safe_path.
    development=split.startswith(('opened development','opened-development','opened_development','development','targeted development','targeted_development','diagnostic','preflight','audit'))
    excluded=bool(re.search(r'(?:^|[^a-z0-9])no(?:$|[^a-z0-9])|not[ _-]accessed|excluded',split))
    return not (development and excluded and numeric and all(v==0 for v in numeric))

def registry_context(state,limit=6):
    state=pathlib.Path(state)
    with sqlite3.connect('file:'+str(state/'research.sqlite')+'?mode=ro',uri=True,timeout=30) as c:
        rows=c.execute('select id,timestamp,status,metadata from experiments order by timestamp desc limit 40').fetchall()
    recent=[]
    for eid,timestamp,status,metadata in rows:
        meta=json.loads(metadata)
        if protected_split(meta):continue
        item={k:meta.get(k) for k in FIELDS}
        item.update(experiment_id=eid,timestamp=timestamp,process_status=status,registry_path='research_artifacts/aws_runs/'+eid+'/registry.json',scientific_acceptance='unverified_here_read_raw_gates')
        item['artifact_paths']=[x.get('path') for x in meta.get('artifact_inventory',[]) if x.get('exists') and x.get('path') and not DENY.search(x['path'])][:12]
        recent.append(item)
        if len(recent)>=max(1,min(10,int(limit))):break
    return dict(source='read_only_SQLite_experiment_registry',captured=dt.datetime.now(dt.timezone.utc).isoformat(),
                rule='Registry IDs/status/budgets override stale narrative pointers. complete means process exit0, not scientific acceptance. Read raw gates; hypotheses/next-action notes may be stale.',
                latest_registered=recent[0] if recent else None,
                latest_process_complete=next((x for x in recent if x['process_status']=='complete'),None),recent=recent)
