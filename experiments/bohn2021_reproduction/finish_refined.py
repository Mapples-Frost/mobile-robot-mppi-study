"""Queue matched baselines after diagnostics, then evaluate the locked candidates."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from runtime import ART,ROOT
from run import write

scripts=ROOT/'experiments/bohn2021_reproduction'
out=ART/'results/refined'
while not (ART/'results/diagnosis/suite_completed.json').exists():time.sleep(10)
code=subprocess.call([sys.executable,'-u',str(scripts/'refined_suite.py'),'--fixed','--workers','4'])
if code:raise RuntimeError('Fixed suite process failed')
while not (out/'rl_completed.json').exists():time.sleep(10)
for name in ['rl_completed.json','fixed_completed.json']:
    assert json.loads((out/name).read_text())['complete'],name
models=sorted(p.parent for p in out.glob('*/completed.json'))
write(out/'holdout_protocol.json',{'models':[p.name for p in models],
    'locked_before_evaluation':True,'selection':'All final refined models; no selection based on holdout or checkpoint curves.',
    'episodes':20,'ablations':['learned_value','zero_value'],
    'fixed_scope':'Three representative fixed horizons per task; not all possible horizons.'})
def ev(job):
    folder,enabled=job
    spec=json.loads((folder/'manifest.json').read_text())
    dest=folder/('holdout_value' if enabled else 'holdout_no_value')
    cmd=[sys.executable,'-u',str(scripts/'evaluate_saved.py'),'--model-dir',str(folder),
        '--bank',str(ART/'configs'/(spec['task']+'_holdout_bank.json')),'--out',str(dest)]
    if not enabled:cmd.append('--no-value')
    with open(out/(folder.name+('_holdout_value.log' if enabled else '_holdout_no_value.log')),'w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    return {'model':folder.name,'terminal_value':enabled,'exit_code':p.returncode}
with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(ev,[(p,v) for p in models for v in [True,False]]))
write(out/'holdout_completed.json',{'jobs':rows,'complete':all(r['exit_code']==0 for r in rows)})
