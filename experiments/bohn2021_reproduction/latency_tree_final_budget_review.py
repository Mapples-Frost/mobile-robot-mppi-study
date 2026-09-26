"""Independently reconcile the final actual-work ledger and its raw counters.

No production ledger, selection or acceptance functions are imported. The output
certifies accounting coverage, never efficacy or overall goal completion.
"""
import json
import math
from pathlib import Path
from collections import defaultdict
from latency_tree_protocol import OUT, SCRIPTS, read, sha, write, verify


def recompute_counters(rows):
    step_attempts=reset_attempts=solver_attempts=solver_completed=warmups=retries=0
    fixed_steps=fixed_resets=0
    for row in rows:
        kind,value=row['kind'],row['values']
        if kind=='environment':
            step_attempts+=value['step_attempts']
            reset_attempts+=value['reset_attempts']
        elif kind=='fixed_instrumentation':
            step_attempts+=value['step_attempts']
            reset_attempts+=value['reset_attempts']
            fixed_steps+=value['step_completed']
            fixed_resets+=value['reset_completed']
        else:
            assert kind=='solver'
            solver_attempts+=value['solve_attempts']
            solver_completed+=value['solve_completed']
            warmups+=value['warmup_attempts']
            retries+=value['retry_attempts']
    return dict(recorded_explicit_step_attempts=step_attempts,
                recorded_explicit_reset_attempts=reset_attempts,
                instrumented_solver_attempts=solver_attempts,
                instrumented_solver_completed=solver_completed,
                instrumented_warmup_attempts=warmups,instrumented_retry_attempts=retries,
                fixed_instrumentation_step_completed=fixed_steps,
                fixed_instrumentation_reset_completed=fixed_resets,
                all_solver_attempts=None,all_solver_completed=None,
                constructor_internal_calls=None,constructor_internal_solves=None)


def verify_files(mapping):
    for name,digest in mapping.items():
        assert sha(Path(name))==digest, name


def run():
    verify()
    ledger_path=OUT/'final_budget/ledger.json'
    ledger=read(ledger_path)
    assert ledger['passed_accounting'] and ledger['goal_complete'] is False
    verify_files(ledger['hashes'])
    verify_files(ledger['source_hashes'])
    registration=read(OUT/'final_budget_registration.json')
    verify_files(registration['hashes'])
    checks_path=OUT/'final_budget_checks.json'
    assert sha(checks_path)==registration['checks_sha256']
    checks=read(checks_path)
    assert checks['passed'] and checks['simulations']==0 and not checks['test_outcomes_read']
    verify_files(checks['hashes'])
    actual=[]
    for p in OUT.rglob('*.json'):
        if p.name.startswith('instrumentation_attempt_'):
            kind='fixed_instrumentation'
        elif p.name.startswith('attempt_'):
            kind='environment'
        elif p.name=='solver_attempts.json':
            kind='solver'
        else:
            continue
        parts=p.relative_to(OUT).parts
        stage='evaluation_'+parts[1] if parts[0]=='evaluations' else parts[0]
        assert stage in ('bank_generation','smoke','train','evaluation_smoke','timing_validation',
                         'extra_fixed','timing_test','evaluation_validation','evaluation_test')
        raw=read(p)
        if kind=='environment':
            values={'step_attempts':raw['step_calls'],'reset_attempts':raw['reset_calls']}
        elif kind=='solver':
            names=('solve_attempts','solve_completed','warmup_attempts','retry_attempts')
            values={name:raw[name] for name in names}
            assert raw['solve_attempts']>=raw['solve_completed']
            assert raw['warmup_attempts']+raw['retry_attempts']<=raw['solve_attempts']
        else:
            names=('step_attempts','step_completed','reset_attempts','reset_completed','environment_constructions')
            values={name:raw[name] for name in names}
            assert raw['step_attempts']>=raw['step_completed'] and raw['reset_attempts']>=raw['reset_completed']
        assert all(type(v) is int and v>=0 for v in values.values())
        actual.append(dict(path=str(p),folder=str(p.parent),kind=kind,stage=stage,sha256=sha(p),values=values))
    expected=sorted(actual,key=lambda r:r['path'])
    assert ledger['counters']==expected
    assert len(expected)==len({r['path'] for r in expected})
    counters=[r for r in expected if r['kind'] in ('environment','fixed_instrumentation')]
    assert len({Path(r['path']).name for r in counters})==len(counters), 'Copied attempts require explicit attribution'
    groups=defaultdict(lambda:dict(
        environment=dict(step_attempts=0,reset_attempts=0),
        solver=dict(solve_attempts=0,solve_completed=0,warmup_attempts=0,retry_attempts=0),
        fixed_instrumentation=dict(step_attempts=0,step_completed=0,reset_attempts=0,
                                   reset_completed=0,environment_constructions=0)))
    for row in expected:
        for k,v in row['values'].items():
            groups[row['stage']][row['kind']][k]+=v
    assert dict(groups)==ledger['stages']
    assert recompute_counters(expected)==ledger['totals']
    by_folder=defaultdict(list)
    for row in expected:
        by_folder[row['folder']].append(row)
    assert {r['folder'] for r in ledger['conditions']}==set(by_folder)
    assert len(ledger['conditions'])==len(by_folder)
    outputs={r['folder']:r for r in ledger['conditions']}
    recounted_calls=fixed_train_steps=fixed_eval_steps=0
    unresolved=[]
    for name,rows in sorted(by_folder.items()):
        folder=Path(name)
        kind_set={r['kind'] for r in rows}
        stage=rows[0]['stage']
        result=outputs[name]
        assert result['stage']==stage
        if kind_set=={'fixed_instrumentation'}:
            assert len(rows)==1 and stage=='extra_fixed'
            value=rows[0]['values']
            instrumentation=read(folder/'instrumentation_completed.json')
            audit=read(folder/'independent_log_audit.json')
            done=read(folder/'completed.json')
            assert instrumentation['passed'] and audit['passed'] and done['status']=='complete'
            verify_files(instrumentation['hashes'])
            verify_files(audit['hashes'])
            transition_files=list(folder.glob('transitions_*.jsonl'))
            reset_files=list(folder.glob('resets_*.jsonl'))
            assert len(transition_files)==len(reset_files)==1
            n_steps=n_resets=0
            with transition_files[0].open() as stream:
                for line in stream:
                    row=json.loads(line)
                    n_steps+=1
                    assert row['index']==n_steps
            with reset_files[0].open() as stream:
                for line in stream:
                    row=json.loads(line)
                    n_resets+=1
                    assert row['reset']==n_resets
            assert n_steps==value['step_completed']==value['step_attempts']==audit['logged_steps']
            assert n_resets==value['reset_completed']==value['reset_attempts']==audit['logged_resets']
            assert done['steps']==audit['training_steps']==15000
            calculated=dict(folder=name,stage=stage,kind='fixed_training',explicit_step_attempts=n_steps,
                            explicit_reset_attempts=n_resets,training_steps=15000,
                            final_diagnostic_steps=n_steps-15000,raw_solve_attempts=None,
                            raw_solve_completed=None,constructor_internal_calls=None,
                            new_gradient_updates=done['updates'],
                            logged_environment_constructions=value['environment_constructions'])
            assert result==calculated
            fixed_train_steps+=15000
            fixed_eval_steps+=n_steps-15000
        else:
            assert kind_set=={'environment','solver'} and len(rows)==2
            env=next(r['values'] for r in rows if r['kind']=='environment')
            sol=next(r['values'] for r in rows if r['kind']=='solver')
            receipt=folder/'completed.json'
            if not receipt.exists():
                unresolved.append(dict(folder=name,reason='Incomplete condition; all attempts charged'))
                assert result['kind']=='incomplete_condition'
                assert result['explicit_step_attempts']==env['step_attempts']
                assert result['explicit_reset_attempts']==env['reset_attempts']
                assert result['raw_solve_attempts']==sol['solve_attempts']
                assert result['raw_solve_completed']==sol['solve_completed']
                continue
            done=read(receipt)
            assert done['passed']
            verify_files(done['hashes'])
            calls=warmups=initial=retries=0
            with (folder/'solver_calls.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    assert row['kind'] in ('warmup','initial','retry_zero','retry_previous')
                    calls+=1
                    warmups+=row['kind']=='warmup'
                    initial+=row['kind']=='initial'
                    retries+=row['kind'].startswith('retry_')
            assert calls==sol['solve_attempts']==sol['solve_completed']
            assert warmups==sol['warmup_attempts']==env['reset_attempts']
            assert initial==env['step_attempts'] and retries==sol['retry_attempts']
            recounted_calls+=calls
            if stage=='bank_generation':
                assert initial==retries==0 and done['resets']==warmups
                episodes=0
            else:
                summary=read(folder/'summary.json')
                assert summary['solver_counts']==sol
                episodes=len(summary['episodes'])
                assert summary['resets']==episodes==warmups
                assert summary['steps']==sum(e['steps'] for e in summary['episodes'])==initial
                assert sum(e['retries'] for e in summary['episodes'])==retries
            assert result==dict(folder=name,stage=stage,kind='completed_condition',
                                explicit_step_attempts=initial,explicit_reset_attempts=warmups,
                                episodes=episodes,raw_solve_attempts=calls,raw_solve_completed=calls,
                                warmup_attempts=warmups,retry_attempts=retries,constructor_internal_calls=None)
    assert unresolved==ledger['issues']
    assert ledger['complete_budget']==(len(unresolved)==0)
    assert fixed_train_steps==ledger['new_fixed_terminal_training_steps']
    assert fixed_eval_steps==ledger['new_fixed_final_diagnostic_steps']
    training_budget=read(OUT/'training_delivery/budget.json')
    assert ledger['inherited_models']==training_budget['inherited_models']
    assert ledger['inherited_terminal_training_steps']==training_budget['inherited_terminal_training_steps']
    for part in ('bank_generation','smoke','train'):
        before=training_budget['groups'][part]
        assert groups[part]['environment']['step_attempts']==before['step_attempts']
        assert groups[part]['environment']['reset_attempts']==before['reset_attempts']
        assert all(groups[part]['solver'][k]==before[k] for k in groups[part]['solver'])
    independent=[]
    for p in sorted(OUT.glob('audit*.json')):
        a=read(p)
        if a.get('passed') and 'independent_integrations' in a:
            verify_files(a['hashes'])
            independent.append(dict(path=str(p),independent_integrations=a['independent_integrations'],
                                    kind='saved_trace_dynamics_audit'))
    for p in sorted((OUT/'extra_fixed').glob('*/independent_log_audit.json')):
        a=read(p)
        assert a['passed']
        verify_files(a['hashes'])
        independent.append(dict(path=str(p),independent_integrations=a['logged_steps'],
                                kind='fixed_training_transition_audit'))
    assert ledger['known_offline_integrations']==independent
    assert ledger['known_offline_integration_total']==sum(a['independent_integrations'] for a in independent)
    assert ledger['whole_campaign_cpu_s'] is None
    validation=read(OUT/'validation_delivery/independent_review.json')
    assert validation['passed']
    verify_files(validation['hashes'])
    test_allowed=validation['effect_passed']
    assert ledger['test_phase_included']==test_allowed
    if test_allowed:
        c=read(OUT/'confirmation_registration.json')
        assert c['validation_gate_passed'] and c['independent_review_passed']
        verify_files(c['hashes'])
        t=read(OUT/'test_delivery/independent_review.json')
        assert t['passed']
        verify_files(t['hashes'])
        assert {'evaluation_test','timing_test'}<=set(groups)
    else:
        assert not (OUT/'evaluations/test').exists() and not (OUT/'timing_test').exists()
        assert not (OUT/'confirmation_registration.json').exists()
        assert not {'evaluation_test','timing_test'}&set(groups)
    assert ledger['test_trajectories_read'] is False
    assert ledger['test_summary_receipts_read']==test_allowed
    for value in ledger['recorded_phase_wall_intervals']:
        state=read(Path(value['path']))
        assert state['complete'] and not state['active']
        assert math.isclose(value['wall_s'],state['ended']-state['started'],rel_tol=1e-12,abs_tol=1e-10)
    # Verification occurs only when no measured experiment Python process is live.
    for cmd in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=cmd.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        assert not any(Path(a).name in ('latency_tree_run.py','latency_tree_evaluate.py',
                                       'latency_tree_fixed_train.py') for a in args), args
    result=dict(passed=True,complete_budget=ledger['complete_budget'],counter_files=len(expected),
                conditions=len(by_folder),recounted_completed_solver_calls=recounted_calls,
                new_fixed_training_steps=fixed_train_steps,new_fixed_diagnostic_steps=fixed_eval_steps,
                totals=ledger['totals'],known_offline_integration_total=ledger['known_offline_integration_total'],
                test_trajectories_read=False,test_summary_receipts_read=test_allowed,
                test_phase_included=test_allowed,goal_complete=False,
                hashes={str(ledger_path):sha(ledger_path),str(Path(__file__).resolve()):sha(Path(__file__)),
                        str(OUT/'final_budget_registration.json'):sha(OUT/'final_budget_registration.json')},
                scope='Independent counter coverage, raw call/transition recount and known offline audit budget. '
                      'Uninstrumented constructor/fixed-training solves and undocumented repeated audit work stay unknown.')
    target=OUT/'final_budget/independent_review.json'
    if target.exists():
        assert read(target)==result
    else:
        write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))


if __name__=='__main__':
    run()
