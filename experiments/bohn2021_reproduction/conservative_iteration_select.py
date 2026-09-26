"""Validation-only fixed-grid nomination and common-round selection."""
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS,BASE,HS,verify
from conservative_iteration_evaluate import arm_name,register
from paper_h_soft_probe import read,digest
from run import write


def load(task,family,seed,h):
    if family in ('grid','matched') and h==BASE[task]:family='primary'
    folder=OUT/'evaluations/validation'/task/arm_name(family,seed,h)
    if not (folder/'completed.json').exists():return None
    done=read(folder/'completed.json')
    assert done['passed']
    for p,value in done['hashes'].items():assert digest(Path(p))==value
    summary=read(folder/'summary.json');rows=summary['episodes']
    assert len(rows)==24 and [r['case'] for r in rows]==list(range(24))
    return dict(family=family,seed=seed,h=h,path=str(folder),episodes=rows,
        total_cost=float(np.mean([r['total_cost'] for r in rows])),
        physical_constraint_cost=float(np.mean([r['physical_constraint_cost'] for r in rows])),
        success=sum(r['success'] for r in rows),constraints=sum(r['constraint'] for r in rows),
        failed_steps=sum(r['solver_failure_steps'] for r in rows),steps=sum(r['steps'] for r in rows),
        failure_rate=sum(r['solver_failure_steps'] for r in rows)/sum(r['steps'] for r in rows))


def safe(a,b):
    return a['success']>=b['success'] and a['constraints']<=b['constraints'] and a['failure_rate']<=b['failure_rate']+1e-12


def summarize(a):return {k:v for k,v in a.items() if k!='episodes'}


def select():
    verify();register()
    assert read(OUT/'validation_status.json')['complete']
    nominations={};pending=[];hashes={}
    for task in TASKS:
        primary=[load(task,'primary',s,BASE[task]) for s in range(3)]
        assert all(primary)
        grid=[load(task,'grid',0,h) for h in range(5,51,5)]
        assert all(grid)
        fixed=min([a for a in grid if safe(a,primary[0])],key=lambda a:(a['total_cost'],a['h']))
        matched={h:[load(task,'matched',s,h) for s in range(3)] for h in HS}
        assert all(all(arms) for arms in matched.values())
        eligible=[h for h,arms in matched.items() if all(safe(a,b) for a,b in zip(arms,primary))]
        matched_h=min(eligible,key=lambda h:(np.mean([a['total_cost'] for a in matched[h]]),h))
        nominations[task]=dict(independent_h=fixed['h'],matched_h=matched_h,
            independent_grid=[summarize(a) for a in grid],matched_grid={str(h):[summarize(a) for a in arms] for h,arms in matched.items()})
        for seed in (1,2):
            if load(task,'grid',seed,fixed['h']) is None:pending.append(dict(task=task,seed=seed,h=fixed['h'],training_steps=15000))
        for a in grid+sum(list(matched.values()),[]):
            p=Path(a['path'])/'completed.json';hashes[str(p)]=digest(p)
    write(OUT/'baseline_selection.json',dict(nominations=nominations,pending_independent_baselines=pending,
        hashes=hashes,source_hash=digest(Path(__file__))))
    if pending:
        print(json.dumps(dict(status='Independent extra fixed seeds must be trained/evaluated before selection',pending=pending),indent=2));return
    selections={};all_rows=[];passed=True
    for task,n in nominations.items():
        comparisons={name:[load(task,family,s,n[name+'_h']) for s in range(3)]
                     for name,family in [('independent','grid'),('matched','matched')]}
        rounds={r:[load(task,'round%d'%r,s,0) for s in range(3)] for r in range(2)}
        assert all(all(arms) for arms in rounds.values())
        eligible=[]
        for r,arms in rounds.items():
            qualifies=True
            for s,a in enumerate(arms):
                for family,baselines in comparisons.items():
                    b=baselines[s]
                    control_tolerance=.02*abs(b['physical_constraint_cost'])
                    good=safe(a,b) and a['physical_constraint_cost']<=b['physical_constraint_cost']+control_tolerance
                    qualifies &= good
                    all_rows.append(dict(task=task,round=r,seed=s,comparator=family,baseline_h=b['h'],
                        safe=safe(a,b),control_noninferior=good,total_cost_difference=a['total_cost']-b['total_cost'],
                        total_cost_relative=(a['total_cost']-b['total_cost'])/max(abs(b['total_cost']),1e-12),
                        adaptive=summarize(a),fixed=summarize(b)))
            if qualifies:eligible.append(r)
        chosen=min(eligible,key=lambda r:(np.mean([a['total_cost'] for a in rounds[r]]),r)) if eligible else None
        passed &= chosen is not None
        selections[task]=dict(round=chosen,eligible_rounds=eligible,independent_h=n['independent_h'],matched_h=n['matched_h'])
        for arms in list(rounds.values())+list(comparisons.values()):
            for a in arms:
                p=Path(a['path'])/'completed.json';hashes[str(p)]=digest(p)
    write(OUT/'validation_selection.json',dict(selection_eligible=passed,selections=selections,
        all_comparisons=all_rows,hashes=hashes,source_hash=digest(Path(__file__)),
        independent_test_unlocked=False,
        next_gate='Before confirmation, audit selected validation efficacy and serial timing; selection eligibility alone does not establish a gain.'))
    print(json.dumps(dict(selection_eligible=passed,selections=selections),indent=2))


if __name__=='__main__':select()
