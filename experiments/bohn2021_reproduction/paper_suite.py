"""Bibliography-corrected SAC defaults; all final candidates declared in advance."""
import json
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from runtime import ART,ROOT
from run import write

out=ART/'results/paper_defaults';out.mkdir(exist_ok=True)
scripts=ROOT/'experiments/bohn2021_reproduction'
jobs=[]
for seed in range(3):
    for task in ['vehicle','pendulum']:jobs.append((task,seed,None))
for task,hs in [('vehicle',[5,10,15]),('pendulum',[20,30,40])]:
    for h in hs:jobs.append((task,0,h))
write(out/'protocol.json',{'jobs':jobs,'training_steps':15000,'batch_size':256,'buffer_size':1000000,
    'ent_coef':1.0,'aligned':True,'scaled_obs':True,
    'source':'https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf Appendix D Table 1',
    'reason':'Bohn Section 4 adopts Haarnoja 2018 defaults except stated changes. Earlier local reconstruction used batch 64 and automatic entropy.',
    'entropy_note':'2018 SAC uses a fixed entropy weight absorbed into reward scaling. Retain author fork reward division convention.',
    'model_selection':'None: all six final RL models and six fixed models evaluated, no checkpoint or seed selection.',
    'test':'Same independent 20-case holdout as refined. This protocol was set before any refined holdout results.',
    'remaining_assumptions':['observation scaling','initial/reset distributions','learning_starts=100','time_aware=True','missing original experiment configs']})
def worker_count():
    count=0
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            s=p.read_bytes()
            if b'python' in s and any(('bohn2021_reproduction/'+n).encode() in s for n in ['run.py','evaluate_saved.py','checkpoint_eval.py']):count+=1
        except (OSError,PermissionError):pass
    return count
pending=jobs.copy();active=[];finished=[]
while pending or active:
    while pending and len(active)<8 and worker_count()<12:
        task,seed,h=pending.pop(0)
        name=task+('_rl_s%d'%seed if h is None else '_fixed_h%d'%h)
        folder=out/name
        if (folder/'completed.json').exists():finished.append({'name':name,'exit_code':0});continue
        cmd=[sys.executable,'-u',str(scripts/'run.py'),'--task',task,'--seed',str(seed),
             '--out',str(folder),'--aligned','--scaled-obs','--batch-size','256','--buffer-size','1000000',
             '--ent-coef','1.0','--test-bank',str(ART/'configs'/(task+'_validation_bank.json'))]
        if h is not None:cmd+=['--fixed-horizon',str(h)]
        log=open(out/(name+'.log'),'w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        active.append((name,p,log))
        time.sleep(.2)
    for name,p,log in active[:]:
        code=p.poll()
        if code is not None:
            log.close();active.remove((name,p,log));finished.append({'name':name,'exit_code':code})
    write(out/'status.json',{'pending':len(pending),'active':[{'name':n,'pid':p.pid} for n,p,l in active],'finished':finished})
    if pending or active:time.sleep(5)
assert len(finished)==12 and all(r['exit_code']==0 for r in finished),finished
write(out/'training_completed.json',{'complete':True,'jobs':finished})
models=sorted(p.parent for p in out.glob('*/completed.json'))
def evaluate_job(job):
    folder,value=job
    spec=json.loads((folder/'manifest.json').read_text())
    ev=folder/('holdout_value' if value else 'holdout_no_value')
    cmd=[sys.executable,'-u',str(scripts/'evaluate_saved.py'),'--model-dir',str(folder),
         '--bank',str(ART/'configs'/(spec['task']+'_holdout_bank.json')),'--out',str(ev)]
    if not value:cmd.append('--no-value')
    with open(out/(folder.name+('_holdout_value.log' if value else '_holdout_no_value.log')),'w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    return {'model':folder.name,'terminal_value':value,'exit_code':p.returncode}
with ThreadPoolExecutor(max_workers=6) as pool:result=list(pool.map(evaluate_job,[(f,v) for f in models for v in [True,False]]))
write(out/'holdout_completed.json',{'jobs':result,'complete':all(r['exit_code']==0 for r in result)})
