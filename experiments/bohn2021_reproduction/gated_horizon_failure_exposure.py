"""Training-only failure exposure of every already audited selected policy.

Checks whether horizon intervention occurred in failed episodes and whether the
scored physical/control trajectories changed. This is descriptive evidence; it
does not establish physical uncontrollability or justify modifying this study.
"""
import csv
import json
from pathlib import Path
from collections import Counter
from gated_horizon_search import OUT,freeze
from gated_horizon_policy import BASE
from paper_h_soft_probe import read,digest
from run import write


def main():
    freeze();source=OUT/'selected_training_diagnostics/report.json';report=read(source);assert report['passed'] and report['training_only']
    hashes={str(source):digest(source)};rows=[];summary=[]
    keys=('previous_state','state','input','horizon','reward','performance','compute','constraint','solver_success','termination')
    for item in report['rows']:
        task,seed,policy=item['task'],item['seed'],item['policy'];root=OUT/'train'/('%s_s%d'%(task,seed));fit=read(root/'fit_completed.json')
        assert fit['selected']['id']==policy
        audit=OUT/'selected_training_diagnostics'/('%s_s%d'%(task,seed))/'audit.json'
        assert digest(audit)==report['hashes'][str(audit)]
        for p,h in read(audit)['hashes'].items():assert digest(Path(p))==h
        for arm in set(('fixed',policy)):
            p=root/arm/'completed.json';hashes[str(p)]=digest(p)
            for name,h in read(p)['hashes'].items():assert digest(Path(name))==h
        group=[]
        for cid in range(24):
            fixed=read(root/'fixed'/('r0_trace_%02d.json'%cid));selected=read(root/policy/('r0_trace_%02d.json'%cid))
            first=lambda indices:next(iter(indices),None)
            short=[i for i,r in enumerate(selected) if r['horizon']<BASE[task]]
            initial=[i for i,r in enumerate(selected) if not r['recovery']['attempts'][0]['success']]
            exact=len(fixed)==len(selected) and all({k:a[k] for k in keys}=={k:b[k] for k in keys} for a,b in zip(selected,fixed))
            success_term='goal' if task=='vehicle' else 'steps'
            failed=fixed[-1]['termination']!=success_term
            fixed_total=sum(-r['reward'] for r in fixed);selected_total=sum(-r['reward'] for r in selected)
            row=dict(task=task,seed=seed,policy=policy,case=cid,baseline_failed=failed,selected_failed=selected[-1]['termination']!=success_term,
                fixed_steps=len(fixed),selected_steps=len(selected),short_steps=len(short),first_short_step=first(short),
                initial_failed_steps=len(initial),first_initial_failure=first(initial),final_failed_steps=sum(not r['solver_success'] for r in selected),
                exact_scored_trajectory=exact,baseline_total_cost=fixed_total,selected_total_cost=selected_total,
                initial_failure_status_counts=json.dumps(dict(Counter(r['recovery']['attempts'][0]['return_status'] for r in selected if not r['recovery']['attempts'][0]['success'])),sort_keys=True))
            rows.append(row);group.append(row)
        total=sum(r['baseline_total_cost'] for r in group);failed=[r for r in group if r['baseline_failed']]
        summary.append(dict(task=task,seed=seed,policy=policy,cases=24,baseline_failed_cases=len(failed),
            failed_cases_with_no_short_h=sum(r['short_steps']==0 for r in failed),failed_cases_exact_trajectory=sum(r['exact_scored_trajectory'] for r in failed),
            failed_cases_initial_failure_at_step_zero=sum(r['first_initial_failure']==0 for r in failed),
            failed_cases_initial_failure_every_step=sum(r['initial_failed_steps']==r['selected_steps'] for r in failed),
            failed_case_cost=sum(r['baseline_total_cost'] for r in failed),all_case_cost=total,
            failed_cost_fraction_of_signed_total=sum(r['baseline_total_cost'] for r in failed)/total if total else None,
            all_cases_with_short_h=sum(r['short_steps']>0 for r in group)))
    dest=OUT/'training_failure_exposure';dest.mkdir(exist_ok=True)
    with (dest/'all_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    value=dict(training_only=True,passed=True,groups=summary,pending_at_source=report['pending'],hashes=hashes,source_hash=digest(Path(__file__)),
        scope='All cases in all completed fits represented by the audited source snapshot. Exactness compares scored physical state/input/H/reward/cost/success/termination; timing and policy metadata excluded. No additional control simulation or numerical integration.',
        limits='Post-selection training description only. Persistent NLP failure with identical controls does not prove physical uncontrollability, global infeasibility, or absence of a better horizon/terminal/solver. No validation or test access and no current policy modification.')
    write(dest/'report.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
