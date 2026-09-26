"""Compact status for long-running reproduction; read-only."""
import json
from runtime import ART
for group in ['diagnosis','refined','paper_defaults']:
    out=ART/'results'/group
    rows=[]
    for p in sorted(out.glob('*/progress.json')):
        d=json.loads(p.read_text())
        rows.append({'run':p.parent.name,'steps':d['steps'],'complete':(p.parent/'completed.json').exists(),
                     'last_termination':(d.get('last_episode') or {}).get('termination')})
    done=sum(r['complete'] for r in rows)
    running=[{'run':r['run'],'steps':r['steps']} for r in rows if not r['complete']]
    tests=list(out.glob('*/holdout_*/completed.json'))
    print(json.dumps({'group':group,'training_complete':done,'training_started':len(rows),
                      'active':running,'holdout_evaluations_complete':len(tests)}),flush=True)
