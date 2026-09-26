"""Validation-only strong fixed-grid nomination; no adaptive retuning."""
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS
from gated_horizon_policy import BASE
from gated_horizon_evaluate import HS,arm_name,register
from paper_h_soft_probe import read,digest
from run import write


def load(task,family,seed,h,split='validation'):
    if family in ('grid','matched') and h==BASE[task]:family='primary'
    folder=OUT/'evaluations'/split/task/arm_name(family,seed,h)
    if not (folder/'completed.json').exists():return None
    done=read(folder/'completed.json');assert done['passed']
    for p,value in done['hashes'].items():assert digest(Path(p))==value
    rows=read(folder/'summary.json')['episodes'];n=32 if split=='validation' else 64
    assert len(rows)==n and [r['case'] for r in rows]==list(range(n))
    steps=sum(e['steps'] for e in rows)
    return dict(task=task,family=family,seed=seed,h=h,path=str(folder),episodes=rows,
        total_cost=float(np.mean([e['total_cost'] for e in rows])),
        physical_constraint_cost=float(np.mean([e['physical_constraint_cost'] for e in rows])),
        success=sum(e['success'] for e in rows),constraints=sum(e['constraint'] for e in rows),steps=steps,
        initial_failures=sum(e['initial_failed_steps'] for e in rows),final_failures=sum(e['solver_failure_steps'] for e in rows),
        initial_failure_rate=sum(e['initial_failed_steps'] for e in rows)/steps,
        final_failure_rate=sum(e['solver_failure_steps'] for e in rows)/steps,
        switches=sum(e['switches'] for e in rows),retries=sum(e['retries'] for e in rows))


def safe(a,b):
    return a['success']>=b['success'] and a['constraints']<=b['constraints'] and a['initial_failure_rate']<=b['initial_failure_rate']+1e-12 and a['final_failure_rate']<=b['final_failure_rate']+1e-12


def summarize(a):return {k:v for k,v in a.items() if k!='episodes'}


def select():
    register();assert read(OUT/'validation_status.json')['complete']
    nominations={};pending=[];hashes={}
    for task in TASKS:
        primary=[load(task,'primary',s,BASE[task]) for s in range(3)]
        grid=[load(task,'grid',0,h) for h in HS];assert all(grid) and all(primary)
        chosen=min([a for a in grid if safe(a,primary[0])],key=lambda a:(a['total_cost'],a['h']))
        matched={h:[load(task,'matched',s,h) for s in range(3)] for h in HS}
        assert all(all(arms) for arms in matched.values())
        eligible=[h for h,arms in matched.items() if all(safe(a,b) for a,b in zip(arms,primary))]
        matched_h=min(eligible,key=lambda h:(np.mean([a['total_cost'] for a in matched[h]]),h))
        nominations[task]=dict(independent_h=chosen['h'],matched_h=matched_h,
            independent_grid=[summarize(a) for a in grid],matched_grid={str(h):[summarize(a) for a in arms] for h,arms in matched.items()})
        for seed in (1,2):
            if load(task,'grid',seed,chosen['h']) is None:pending.append(dict(task=task,seed=seed,h=chosen['h'],training_steps=15000))
        for a in grid+sum(list(matched.values()),[]):
            p=Path(a['path'])/'completed.json';hashes[str(p)]=digest(p)
    write(OUT/'baseline_selection.json',dict(nominations=nominations,pending_independent_baselines=pending,hashes=hashes,source_hash=digest(Path(__file__))))
    print(json.dumps(dict(nominations={t:{k:v for k,v in n.items() if k.endswith('_h')} for t,n in nominations.items()},pending=pending),indent=2))


if __name__=='__main__':select()
