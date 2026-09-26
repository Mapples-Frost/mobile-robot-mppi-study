"""Start the four already-declared pending baselines while resources are free."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from runtime import ART,ROOT
from run import write

out=ART/'results/paper_defaults'
jobs=[('vehicle',15),('pendulum',20),('pendulum',30),('pendulum',40)]
declared=json.loads((out/'protocol.json').read_text())['jobs']
assert all([t,0,h] in declared for t,h in jobs)
def run(job):
    task,h=job;name=task+'_fixed_h%d'%h
    cmd=[sys.executable,'-u',str(ROOT/'experiments/bohn2021_reproduction/run.py'),'--task',task,
        '--seed','0','--out',str(out/name),'--aligned','--scaled-obs','--batch-size','256',
        '--buffer-size','1000000','--ent-coef','1.0','--fixed-horizon',str(h),
        '--test-bank',str(ART/'configs'/(task+'_validation_bank.json'))]
    with open(out/(name+'_prelaunch.log'),'w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    return {'name':name,'exit_code':p.returncode}
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,jobs))
write(out/'extra_completed.json',{'jobs':rows,'complete':all(r['exit_code']==0 for r in rows),
    'note':'Already declared jobs, not additional variants; run.py lock prevents duplicate training.'})
