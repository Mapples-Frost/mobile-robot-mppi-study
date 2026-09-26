"""Compact progress and process-alive snapshot; never changes experiments."""
import json
import sys
from pathlib import Path
from runtime import ART

out=ART/'results/optimized'
rows=[];completed=[];evaluations=0
for p in sorted(out.glob('*/manifest.json')):
    folder=p.parent
    if folder.name.startswith('smoke'):continue
    if (folder/'completed.json').exists():
        completed.append(folder.name)
        evaluations+=sum((folder/m/'completed.json').exists() for m in ['holdout_value','holdout_no_value'])
    elif (folder/'progress.json').exists():
        r=json.loads((folder/'progress.json').read_text())
        running=json.loads((folder/'running.json').read_text())
        rows.append({'name':folder.name,'steps':r['steps'],'seconds':round(r['elapsed_s']),
            'alive':Path('/proc/%d'%running['pid']).exists(),'last_episode':r['last_episode']})
if '--compact' in sys.argv:
    print(json.dumps({'done':len(completed),'active':{r['name']:r['steps'] for r in rows},
        'dead':[r['name'] for r in rows if not r['alive']],'holdout_done':evaluations,
        'finalized':(out/'finalized.json').exists(),
        'finalization_failed':(out/'finalization_failure.json').exists()},ensure_ascii=False),flush=True)
else:
    print(json.dumps({'training_completed':len(completed),'completed_names':completed,
        'active':rows,'holdout_evaluations_completed':evaluations},ensure_ascii=False),flush=True)
