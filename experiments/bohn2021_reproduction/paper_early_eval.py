"""Evaluate declared final pendulum models while vehicle training finishes."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from runtime import ART,ROOT
from run import write

out=ART/'results/paper_defaults'
folders=sorted(p.parent for p in out.glob('pendulum*/completed.json'))
assert len(folders)==6,'All predeclared pendulum models must be trained'
def ev(job):
    folder,value=job
    cmd=[sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/evaluate_saved.py'),
         '--model-dir',str(folder),'--bank',str(ART/'configs/pendulum_holdout_bank.json'),
         '--out',str(folder/('holdout_value' if value else 'holdout_no_value'))]
    if not value:cmd.append('--no-value')
    with open(out/(folder.name+('_early_value.log' if value else '_early_no_value.log')),'w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    return {'model':folder.name,'terminal_value':value,'exit_code':p.returncode}
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(ev,[(f,v) for f in folders for v in [True,False]]))
write(out/'early_eval_completed.json',{'complete':all(r['exit_code']==0 for r in rows),'jobs':rows,
      'note':'Final candidates already locked by protocol; no selection. File lock prevents duplicate evaluation.'})
