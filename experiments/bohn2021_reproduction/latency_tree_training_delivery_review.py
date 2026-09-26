"""Independent coverage and arithmetic review of the pre-validation delivery.

No production report, candidate-selection or acceptance function is imported.
Dynamics and learned parameter selection rely on their separately frozen audits;
this review independently checks complete delivery coverage and actual budgets.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from latency_tree_protocol import ROOT, OUT, REG, SCRIPTS, TASKS, read, sha, write, verify

DEST = OUT/'training_delivery'


def verify_files(mapping):
    for name, digest in mapping.items():
        assert sha(Path(name)) == digest, name


def exact_record(expected, actual, label):
    assert set(expected) == set(actual), (label, set(expected)^set(actual))
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, float):
            assert math.isfinite(value) and math.isfinite(float(observed))
            assert math.isclose(value,float(observed),rel_tol=1e-11,abs_tol=1e-10), (
                label,key,value,observed)
        elif isinstance(value, bool):
            assert observed == str(value), (label,key,value,observed)
        elif value is None:
            assert observed == '', (label,key,value,observed)
        else:
            assert str(value) == observed, (label,key,value,observed)


def table(name):
    with (DEST/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def stats(rows):
    assert rows
    n=len(rows)
    integer_fields=('steps','success','constraint','solver_failure_steps',
                    'initial_failed_steps','retries','recovered_steps','switches')
    values={k:sum(int(row[k]) for row in rows) for k in integer_fields}
    values['episodes']=n
    assert values['steps']>0
    for k in ('total_cost','performance_cost','h_penalty','constraint_cost','physical_constraint_cost'):
        assert all(math.isfinite(row[k]) for row in rows)
        values[k]=math.fsum(row[k] for row in rows)/n
    for k in ('decision_total_s','gross_total_s','logging_total_s'):
        assert all(row[k]>0 and math.isfinite(row[k]) for row in rows)
        values[k]=math.fsum(row[k] for row in rows)
    values['decision_mean_s']=values['decision_total_s']/values['steps']
    values['initial_failure_rate']=values['initial_failed_steps']/values['steps']
    values['final_failure_rate']=values['solver_failure_steps']/values['steps']
    values['mean_horizon']=math.fsum(row['mean_horizon']*row['steps'] for row in rows)/values['steps']
    return values


def key(policy):
    import hashlib
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def review():
    verify()
    status=read(OUT/'status.json')
    assert status['complete'] and not status['active']
    assert not (OUT/'evaluations').exists(), 'Training delivery review must precede formal outcomes'
    manifest_path=DEST/'manifest.json'
    manifest=read(manifest_path)
    assert manifest['passed'] and manifest['training_only']
    assert manifest['validation_accessed'] is manifest['test_accessed'] is manifest['goal_complete'] is False
    verify_files(manifest['source_hashes'])
    verify_files(manifest['hashes'])
    for name in ('audit_train.json','audit_smoke.json','learning_audit.json'):
        receipt=read(OUT/name)
        assert receipt['passed']
        verify_files(receipt['hashes'])
    candidates=table('all_candidates.csv')
    conditions=table('all_conditions.csv')
    episodes=table('all_episodes.csv')
    counters=table('all_counters.csv')
    policies=read(DEST/'selected_policies.json')['policies']
    assert len(candidates)==288 and len(policies)==6
    candidate_index={(r['task'],int(r['seed']),r['candidate']):r for r in candidates}
    assert len(candidate_index)==288
    condition_index={r['folder']:r for r in conditions}
    assert len(condition_index)==len(conditions)
    episode_index={(r['folder'],int(r['repeat']),int(r['case'])):r for r in episodes}
    assert len(episode_index)==len(episodes)
    expected_conditions=[]
    candidate_count=0
    unique_total=0
    for task in TASKS:
        for seed in range(3):
            root=OUT/'train'/('%s_s%d'%(task,seed))
            fit=read(root/'fit.json')
            selected=read(root/'policy.json')
            assert len(fit['all_candidates'])==48
            assert [r['id'] for r in fit['all_candidates']]==[
                'g%d_c%02d'%(g,c) for g in range(4) for c in range(12)]
            multiplicities=Counter(key(r['policy']) for r in fit['all_candidates'])
            unique_total+=len(multiplicities)
            item=dict(task=task,seed=seed,selected=fit['selected'],policy=selected,
                      learned_tree_selected=selected['kind']=='tree',candidates=48,
                      unique_candidates=len(multiplicities),
                      repeated_candidate_evaluations=48-len(multiplicities))
            saved=[p for p in policies if p['task']==task and p['seed']==seed]
            assert saved==[item]
            expected_conditions.append((root/'threshold_reference',task,seed,'threshold',-1,'fixed',0))
            for g in range(4):
                expected_conditions.append((root/('generation%d'%g)/'fixed',task,seed,
                                            'generation_reference',g,'fixed',0))
                generation=read(root/('generation%d'%g)/'selection.json')
                assert generation['all_candidates']==fit['all_candidates'][12*g:12*(g+1)]
                for item in generation['all_candidates']:
                    folder=root/('generation%d'%g)/item['id']
                    expected_conditions.append((folder,task,seed,'candidate',g,item['id'],0))
                    summary=read(folder/'summary.json')
                    rank=item['rank']
                    expected=dict(task=task,seed=seed,generation=g,candidate=item['id'],
                                  policy_sha256=key(item['policy']),
                                  duplicate_multiplicity=multiplicities[key(item['policy'])],
                                  eligible=rank['eligible'],violations=rank['violations'],
                                  objective=rank['objective'],cost_change=rank['cost_change'],
                                  physical_change=rank['physical_change'],time_ratio=rank['time_ratio'],
                                  selected_final=item['id']==fit['selected'],**stats(summary['episodes']))
                    exact_record(expected,candidate_index[(task,seed,item['id'])],'candidate')
                    candidate_count+=1
            selection=read(root/'selection_registration.json')
            ids=[arm['id'] for arm in selection['arms']]
            assert len(ids)==len(set(ids)) and ids[0]=='fixed'
            assert set(ids)=={'fixed'}|{p['id'] for p in selection['finalists']}|(
                {'certificate'} if task=='pendulum' else set())
            assert len(selection['finalists'])<=4 and fit['selected'] in ids
            assert fit['selected']!='certificate'
            assert {r['id'] for r in fit['selection']}==set(ids)-{'certificate'}
            for repeat in range(2):
                for arm in selection['arms']:
                    expected_conditions.append((root/'selection'/('r%d_%s'%(repeat,arm['id'])),
                                                task,seed,'selection',-1,arm['id'],repeat))
    expected_folders={str(row[0]) for row in expected_conditions}
    assert candidate_count==288
    assert expected_folders==set(condition_index)
    assert expected_folders=={str(p.parent) for p in (OUT/'train').rglob('summary.json')}
    checked_episodes=0
    for folder,task,seed,stage,generation,arm,repeat in expected_conditions:
        summary=read(folder/'summary.json')
        assert summary['task']==task and summary['seed']==seed
        assert summary['split']==('select' if stage=='selection' else 'fit') and summary['repeats']==1
        count=16 if stage=='selection' else 12
        rows=summary['episodes']
        assert len(rows)==count and [(e['repeat'],e['case']) for e in rows]==[(0,c) for c in range(count)]
        metadata=dict(task=task,seed=seed,stage=stage,generation=generation,arm=arm,
                      selection_repeat=repeat,folder=str(folder))
        exact_record(dict(metadata,**stats(rows)),condition_index[str(folder)],str(folder))
        for e in rows:
            exact_record(dict(metadata,**e),episode_index[(str(folder),0,e['case'])],'episode')
            checked_episodes+=1
    assert checked_episodes==len(episodes)==manifest['episode_rows']
    assert len(conditions)==manifest['condition_rows'] and len(candidates)==manifest['candidate_rows']
    budget=read(DEST/'budget.json')
    by_folder={r['folder']:r for r in counters}
    assert len(by_folder)==len(counters)==len(budget['conditions'])
    actual_meters={p for part in ('bank_generation','smoke','train')
                   for p in (OUT/part).rglob('attempt_*.json')}
    actual_solvers={p for part in ('bank_generation','smoke','train')
                    for p in (OUT/part).rglob('solver_attempts.json')}
    folders={p.parent for p in actual_solvers}
    assert len(actual_meters)==len(folders) and {p.parent for p in actual_meters}==folders
    assert {str(p) for p in folders}==set(by_folder)
    group_totals={g:dict(scored_steps=0,step_attempts=0,step_completed_inferred=0,
                        reset_attempts=0,reset_completed_inferred=0,solve_attempts=0,
                        solve_completed=0,warmup_attempts=0,retry_attempts=0)
                  for g in ('bank_generation','smoke','train')}
    for folder in sorted(folders):
        part=folder.relative_to(OUT).parts[0]
        meter_paths=list(folder.glob('attempt_*.json'))
        assert len(meter_paths)==1
        m=read(meter_paths[0])
        s=read(folder/'solver_attempts.json')
        completed=read(folder/'completed.json')
        assert completed['passed']
        verify_files(completed['hashes'])
        assert manifest['raw_completion_hashes'][str(folder/'completed.json')]==sha(folder/'completed.json')
        # Recount raw solver records rather than trusting the delivered counters.
        call_count=warmups=retries=initial=0
        with (folder/'solver_calls.jsonl').open() as stream:
            for line in stream:
                call=json.loads(line)
                kind=call['kind']
                assert kind in ('warmup','initial','retry_zero','retry_previous')
                call_count+=1
                warmups+=kind=='warmup'
                retries+=kind.startswith('retry_')
                initial+=kind=='initial'
        assert call_count==s['solve_attempts']==s['solve_completed']
        assert warmups==s['warmup_attempts']==m['reset_calls']
        assert retries==s['retry_attempts'] and initial==m['step_calls']
        if part=='bank_generation':
            assert initial==0 and retries==0 and completed['resets']==warmups
            explicit_episodes=warmups
        else:
            summary=read(folder/'summary.json')
            assert summary['solver_counts']==s
            assert sum(e['steps'] for e in summary['episodes'])==initial==summary['steps']
            explicit_episodes=len(summary['episodes'])
            assert summary['resets']==explicit_episodes==warmups
        expected=dict(group=part,folder=str(folder),scored_steps=initial,
                      step_attempts=m['step_calls'],step_completed_inferred=initial,
                      reset_attempts=m['reset_calls'],reset_completed_inferred=explicit_episodes,**s)
        exact_record(expected,by_folder[str(folder)],'counter')
        delivered=[r for r in budget['conditions'] if r['folder']==str(folder)]
        assert delivered==[expected]
        for k in group_totals[part]:
            group_totals[part][k]+=expected[k]
    assert group_totals==budget['groups']
    totals={k:sum(g[k] for g in group_totals.values()) for k in next(iter(group_totals.values()))}
    assert totals==budget['totals']
    assert totals['solve_attempts']==totals['step_attempts']+totals['reset_attempts']+totals['retry_attempts']
    assert group_totals['bank_generation']['reset_attempts']==556
    assert group_totals['train']['reset_attempts']==len(episodes)
    assert group_totals['train']['scored_steps']==sum(int(r['steps']) for r in conditions)
    assert len(episodes)+group_totals['smoke']['reset_attempts']<=4920
    assert group_totals['train']['scored_steps']+group_totals['smoke']['scored_steps']<=738000
    inventory=read(OUT/'evaluation_registration.json')['model_inventory']
    inherited={r['folder']:r for r in budget['inherited_models']}
    assert len(inherited)==len(budget['inherited_models'])
    assert set(inherited)=={p for p in inventory.values() if p is not None}
    history_steps=0
    for folder,item in inherited.items():
        model=Path(folder)
        m,d=read(model/'manifest.json'),read(model/'completed.json')
        assert m['steps']==d['steps']==item['training_steps']==15000
        assert item['previously_completed'] is True and item['new_compute_in_this_experiment'] is False
        assert item['elapsed_s_recorded']==d.get('elapsed_s')
        assert item['final_diagnostic_episodes_recorded']==d.get('test_episodes')
        history_steps+=d['steps']
    assert history_steps==budget['inherited_terminal_training_steps']
    assert budget['new_gradient_steps']==0
    assert math.isclose(budget['whole_training_wall_s'],status['ended']-status['started'],abs_tol=1e-10)
    assert budget['whole_training_wall_s']>0
    for field in ('per_condition_wall_s','process_cpu_s','constructor_internal_calls','constructor_internal_solves'):
        assert budget[field] is None
    assert budget['completed_offline_integrations']=={
        name:read(OUT/name)['independent_integrations'] for name in ('audit_train.json','audit_smoke.json')}
    raw_expected={str(folder/'completed.json') for folder in folders}
    assert set(manifest['raw_completion_hashes'])==raw_expected
    receipt=dict(passed=True,training_only=True,candidates=288,unique_candidates=unique_total,
                 conditions=len(conditions),episodes=len(episodes),counter_folders=len(folders),
                 recounted_solver_calls=totals['solve_completed'],totals=totals,
                 inherited_training_steps=history_steps,test_accessed=False,validation_accessed=False,
                 goal_complete=False,simulations=0,
                 hashes={str(manifest_path):sha(manifest_path),str(Path(__file__).resolve()):sha(Path(__file__)),
                         str(OUT/'audit_train.json'):sha(OUT/'audit_train.json'),
                         str(OUT/'learning_audit.json'):sha(OUT/'learning_audit.json')},
                 scope='Independent full training delivery coverage, CSV arithmetic and raw solve recount. '
                       'No claim of efficacy; separate audits establish dynamics and selection integrity.')
    target=DEST/'independent_review.json'
    if target.exists():
        assert read(target)==receipt
    else:
        write(target,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='hashes'},indent=2))


if __name__=='__main__':
    review()

