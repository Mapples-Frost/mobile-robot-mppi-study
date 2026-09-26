"""Execute the declared pendulum holdout after its grid and replications finish."""
import json
from concurrent.futures import ThreadPoolExecutor
from optimized_suite import OUT, evaluate
from run import write

selected=json.loads((OUT/'pendulum_replication_selection.json').read_text())
folders=sorted(f for f in OUT.glob('pendulum_*_s*') if (f/'manifest.json').exists())
assert len(folders)==15
assert all((f/'completed.json').exists() for f in folders)
assert all((OUT/('pendulum_fixed_h%d_s%d'%(selected['h'],s))/'completed.json').exists() for s in [0,1,2])
with ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(evaluate,[(f,v) for f in folders for v in [True,False]]))
assert all(r['exit_code']==0 for r in results)
write(OUT/'pendulum_holdout_overlap_completed.json',{'complete':True,'jobs':results,
    'scope':'Declared unchanged holdout; scheduled after all pendulum training, concurrent with vehicle training. No model or protocol changes.'})
