"""Post-selection training diagnostics for every completed fit; no efficacy inference.

Independently audit the selected policy and its paired fixed baseline, then
separate physical cost from the paper's horizon penalty. No control simulation,
validation/test access, new model selection or protocol modification.
"""
import csv
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE,candidates
from gated_horizon_audit import audit_candidate
from paper_h_soft_probe import read,digest
from run import write


def main():
    freeze();dest=OUT/'selected_training_diagnostics';dest.mkdir(exist_ok=True)
    rows=[];pairs=[];pending=[];hashes={};audits=[];safety=[]
    for task in TASKS:
        for seed in range(3):
            root=OUT/'train'/('%s_s%d'%(task,seed));fit_path=root/'fit_completed.json'
            if not fit_path.exists():pending.append(dict(task=task,seed=seed));continue
            fit=read(fit_path);results=fit['results'];base=results[0]
            assert [r['policy']['id'] for r in results]==['fixed']+[p['id'] for p in candidates(task)]
            eligible=[r for r in results if r['fully_evaluated'] and not r['rejected'] and r['mean_physical_cost']<=base['mean_physical_cost']+.02*abs(base['mean_physical_cost'])]
            chosen=min(eligible,key=lambda r:(r['mean_raw_cost'],r['policy']['id']!='fixed',r['policy']['id']))
            assert fit['selected']==chosen['policy'] and fit['policy_hash']==digest(root/'policy.json')
            assert all(read(root/r['policy']['id']/'completed.json')['result']==r for r in results)
            bank=OUT/'banks'/('%s_train_s%d_bank.json'%(task,seed));assert digest(bank)==fit['training_bank_hash']
            cases=read(bank)['cases'];assert len(cases)==24
            # Cached audits must also detect edits below an unchanged completion
            # manifest, rather than trusting only that manifest's own digest.
            for result in [base,chosen]:
                for p,h in read(root/result['policy']['id']/'completed.json')['hashes'].items():assert digest(Path(p))==h
            output=dest/('%s_s%d'%(task,seed));output.mkdir(exist_ok=True);audit_path=output/'audit.json'
            if audit_path.exists():
                audit=read(audit_path);assert audit['passed'] and audit['source_hash']==digest(Path(__file__))
                for p,h in audit['hashes'].items():assert digest(Path(p))==h
            else:
                groups=[];inputs=[fit_path,root/'policy.json',bank]
                for result in [base]+([] if chosen['policy']['id']=='fixed' else [chosen]):
                    folder=root/result['policy']['id']
                    groups.append(audit_candidate(folder,task,cases,None if result is base else base['episodes']))
                    inputs.append(folder/'completed.json')
                audit=dict(passed=True,task=task,seed=seed,groups=groups,source_hash=digest(Path(__file__)),hashes={str(p):digest(p) for p in inputs},
                    scope='Selected and paired fixed full training traces only; exhaustive candidate audit remains required before validation.')
                write(audit_path,audit)
            audits.append(audit);hashes[str(audit_path)]=digest(audit_path)
            safety.append(dict(task=task,seed=seed,cases=24,**{label:dict(successes=sum(e['success'] for e in result['episodes']),
                constraints=sum(e['constraint'] for e in result['episodes']),initial_failed_steps=sum(e['initial_failed_steps'] for e in result['episodes']),
                final_failed_steps=sum(e['solver_failure_steps'] for e in result['episodes'])) for label,result in [('fixed',base),('selected',chosen)]}))
            short_steps=0;total_steps=0;short_cases=0;worse_physical=0;worse_total=0;sums={k:[] for k in ('total_cost','performance_cost','h_penalty','constraint_cost','physical_constraint_cost')}
            for a,b in zip(chosen['episodes'],base['episodes']):
                assert a['case']==b['case'];cid=a['case'];trace=read(root/chosen['policy']['id']/('r0_trace_%02d.json'%cid));baseline=read(root/'fixed'/('r0_trace_%02d.json'%cid))
                assert trace[0]['previous_state']==baseline[0]['previous_state']
                n=sum(r['horizon']<BASE[task] for r in trace);short_steps+=n;total_steps+=len(trace);short_cases+=bool(n)
                delta={k:a[k]-b[k] for k in sums}
                np.testing.assert_allclose(delta['total_cost'],sum(delta[k] for k in ('performance_cost','h_penalty','constraint_cost')),rtol=1e-9,atol=1e-9)
                for k in sums:sums[k].append(delta[k])
                worse_physical+=delta['physical_constraint_cost']>1e-8;worse_total+=delta['total_cost']>1e-8
                pairs.append(dict(task=task,seed=seed,case=cid,policy=chosen['policy']['id'],short_steps=n,steps=len(trace),
                    **{'delta_'+k:v for k,v in delta.items()},selected_success=a['success'],fixed_success=b['success'],selected_constraint=a['constraint'],fixed_constraint=b['constraint'],
                    selected_initial_failures=a['initial_failed_steps'],fixed_initial_failures=b['initial_failed_steps'],selected_final_failures=a['solver_failure_steps'],fixed_final_failures=b['solver_failure_steps']))
            rows.append(dict(task=task,seed=seed,policy=chosen['policy']['id'],cases=24,short_cases=short_cases,short_steps=short_steps,steps=total_steps,
                short_step_fraction=short_steps/total_steps,horizon_switches=sum(e['switches'] for e in chosen['episodes']),
                baseline_raw_mean=base['mean_raw_cost'],selected_raw_mean=chosen['mean_raw_cost'],
                **{'mean_delta_'+k:float(np.mean(v)) for k,v in sums.items()},
                cases_higher_physical=worse_physical,cases_higher_total=worse_total))
            print('%s_s%d audited selected/fixed training components'%(task,seed),flush=True)
    assert rows
    for name,values in [('all_completed_fits.csv',rows),('all_paired_training_episodes.csv',pairs)]:
        with (dest/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    value=dict(passed=True,training_only=True,completed_fits=len(rows),pending=pending,rows=rows,hashes=hashes,source_hash=digest(Path(__file__)),
        audited_episodes=sum(g['episodes'] for a in audits for g in a['groups']),audited_steps=sum(g['steps'] for a in audits for g in a['groups']),
        numerical_integrations=sum(g['numerical_integrations'] for a in audits for g in a['groups']),
        maximum_dynamics_error=max(g['maximum_dynamics_error'] for a in audits for g in a['groups']),
        limitation='Post-selection training summaries have selection bias. H penalty is a reward proxy, not elapsed time. Every completed fit is included; pending fits explicitly listed. No validation/test efficacy claim or method change.')
    p=dest/'all_paired_training_episodes.csv'
    write(dest/'safety_summary.json',dict(groups=safety,input_path=str(p),input_sha256=digest(p),scope='All completed-fit training scenarios, including failures. No validation/test inference.'))
    write(dest/'report.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
