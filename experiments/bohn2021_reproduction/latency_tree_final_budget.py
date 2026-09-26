"""Final actual-work ledger after the native pipeline has stopped.

Never run during measured learning/timing. This is post-experiment accounting,
not a selector, efficacy gate, or test unlock. Unresolved attempts remain charged
and explicitly prevent a complete-budget claim.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from latency_tree_protocol import ROOT, OUT, SCRIPTS, read, write, sha, verify

SOURCES = ('latency_tree_final_budget.py', 'latency_tree_final_budget_review.py')
REGISTRATION = OUT/'final_budget_registration.json'
DEST = OUT/'final_budget'
ENV_FIELDS = ('step_attempts','reset_attempts')
SOLVER_FIELDS = ('solve_attempts','solve_completed','warmup_attempts','retry_attempts')
FIXED_FIELDS = ('step_attempts','step_completed','reset_attempts','reset_completed','environment_constructions')


def verify_hashes(mapping):
    for name,digest in mapping.items():
        assert sha(Path(name)) == digest, name


def frozen(path,value):
    if path.exists():
        assert read(path) == value, str(path)
    else:
        write(path,value)


def stage_for(path):
    parts=path.relative_to(OUT).parts
    if parts[0]=='evaluations':
        assert len(parts)>2 and parts[1] in ('validation','test'), parts
        return 'evaluation_'+parts[1]
    assert parts[0] in (
        'bank_generation','smoke','train','evaluation_smoke','timing_validation',
        'extra_fixed','timing_test'), ('Unclassified counter requires explicit review',str(path))
    return parts[0]


def fields(value, names):
    assert all(type(value[k]) is int and value[k]>=0 for k in names)
    return {k:value[k] for k in names}


def sum_rows(rows, names):
    return {key:sum(row[key] for row in rows) for key in names}


def normalize_counter(kind,data):
    if kind=='environment':
        values=fields(data,('step_calls','reset_calls'))
        return dict(step_attempts=values['step_calls'],reset_attempts=values['reset_calls'])
    if kind=='solver':
        values=fields(data,SOLVER_FIELDS)
        assert values['solve_attempts']>=values['solve_completed']
        assert values['warmup_attempts']+values['retry_attempts']<=values['solve_attempts']
        return values
    assert kind=='fixed_instrumentation'
    values=fields(data,FIXED_FIELDS)
    assert values['step_attempts']>=values['step_completed']
    assert values['reset_attempts']>=values['reset_completed']
    return values


def check():
    checked=[]
    env=normalize_counter('environment',dict(step_calls=20,reset_calls=3))
    assert env==dict(step_attempts=20,reset_attempts=3)
    checked.append('environment_counts')
    solver=normalize_counter('solver',dict(solve_attempts=26,solve_completed=25,warmup_attempts=3,retry_attempts=3))
    assert solver['solve_attempts']-solver['solve_completed']==1
    checked.append('partial_solve_retained')
    fixed=normalize_counter('fixed_instrumentation',dict(step_attempts=30,step_completed=29,
        reset_attempts=4,reset_completed=3,environment_constructions=1))
    assert fixed['step_attempts']==30 and fixed['step_completed']==29
    checked.append('partial_fixed_step_retained')
    assert sum_rows([env,env],ENV_FIELDS)==dict(step_attempts=40,reset_attempts=6)
    checked.append('duplicate_execution_charged')
    cases=[
        ('environment',dict(step_calls=-1,reset_calls=0)),
        ('environment',dict(step_calls=True,reset_calls=0)),
        ('solver',dict(solve_attempts=4,solve_completed=5,warmup_attempts=1,retry_attempts=0)),
        ('solver',dict(solve_attempts=4,solve_completed=4,warmup_attempts=3,retry_attempts=2)),
        ('fixed_instrumentation',dict(step_attempts=2,step_completed=3,reset_attempts=1,
                                     reset_completed=1,environment_constructions=1)),
        ('fixed_instrumentation',dict(step_attempts=2,step_completed=2,reset_attempts=1,
                                     reset_completed=2,environment_constructions=1)),
    ]
    for i,(kind,row) in enumerate(cases):
        try:normalize_counter(kind,row)
        except AssertionError:checked.append('invalid_counter_%d'%i)
        else:raise AssertionError((kind,row))
    for relative,stage in (
        ('train/vehicle_s0/generation0/g0_c00/solver_attempts.json','train'),
        ('evaluations/validation/vehicle/adaptive_h0_s0/solver_attempts.json','evaluation_validation'),
        ('evaluations/test/pendulum/adaptive_h0_s0/solver_attempts.json','evaluation_test'),
        ('extra_fixed/pendulum_h35_s1/instrumentation_attempt_1.json','extra_fixed'),
    ):
        assert stage_for(OUT/relative)==stage
        checked.append('stage_'+stage)
    try:stage_for(OUT/'unknown/attempt_1.json')
    except AssertionError:checked.append('unclassified_attempt_refused')
    else:raise AssertionError('unclassified attempts cannot be silently omitted')
    from latency_tree_final_budget_review import recompute_counters
    trial=[dict(kind='environment',stage='train',values=env),
           dict(kind='solver',stage='train',values=solver),
           dict(kind='fixed_instrumentation',stage='extra_fixed',values=fixed)]
    totals=recompute_counters(trial)
    assert totals['recorded_explicit_step_attempts']==50
    assert totals['recorded_explicit_reset_attempts']==7
    assert totals['instrumented_solver_attempts']==26 and totals['instrumented_solver_completed']==25
    assert totals['fixed_instrumentation_step_completed']==29
    assert totals['all_solver_attempts'] is None and totals['all_solver_completed'] is None
    checked.append('independent_known_answer_totals')
    result=dict(passed=True,checks=checked,count=len(checked),simulations=0,
                test_outcomes_read=False,hashes={str(SCRIPTS/n):sha(SCRIPTS/n) for n in SOURCES})
    frozen(OUT/'final_budget_checks.json',result)
    print(json.dumps(dict(passed=True,checks=len(checked),simulations=0)))


def register():
    verify()
    if REGISTRATION.exists():
        saved=read(REGISTRATION)
        verify_hashes(saved['hashes'])
        return saved
    assert not DEST.exists()
    checks=read(OUT/'final_budget_checks.json')
    assert checks['passed'] and checks['simulations']==0 and not checks['test_outcomes_read']
    verify_hashes(checks['hashes'])
    value=dict(purpose='Post-experiment budget accounting only; no effect criterion or policy change',
               source_files=list(SOURCES),hashes={str(SCRIPTS/n):sha(SCRIPTS/n) for n in SOURCES},
               checks_sha256=sha(OUT/'final_budget_checks.json'),test_unlock=False)
    frozen(REGISTRATION,value)
    return value


def stopped():
    verify()
    status_paths=[OUT/'status.json']+sorted(OUT.glob('evaluation_*_status.json'))
    values=[]
    for path in status_paths:
        state=read(path)
        assert not state['active'] and state.get('complete'), ('Unfinished experiment',str(path))
        command=Path('/proc')/str(state['pid'])/'cmdline'
        if command.exists():
            assert not any(name in command.read_bytes() for name in (
                b'latency_tree_run.py',b'latency_tree_evaluate.py')), 'Scored experiment process remains live'
        values.append((path,state))
    controllers=sorted(OUT.glob('native_pipeline*_status.json'))
    current=[]
    for p in controllers:
        state=read(p)
        if state.get('stage')=='superseded_before_any_child_stage':
            continue
        current.append((p,state))
    assert len(current)==1, 'Exactly one final, non-superseded native pipeline state required'
    path,native=current[0]
    assert native['complete'] and not native['active'], 'Wait until native pipeline finishes'
    assert native['stage'] in ('negative_validation_test_sealed','confirmation_review_complete_delivery_pending')
    values.append((path,native))
    validation=read(OUT/'validation_delivery/independent_review.json')
    assert validation['passed']
    verify_hashes(validation['hashes'])
    if validation['effect_passed']:
        confirm=read(OUT/'confirmation_registration.json')
        assert confirm['validation_gate_passed'] and confirm['independent_review_passed']
        verify_hashes(confirm['hashes'])
        review=read(OUT/'test_delivery/independent_review.json')
        assert review['passed']
        verify_hashes(review['hashes'])
        assert native['test_accessed']
    else:
        assert not (OUT/'confirmation_registration.json').exists()
        assert not (OUT/'evaluations/test').exists() and not (OUT/'timing_test').exists()
        assert not native['test_accessed']
    return values,validation['effect_passed']


def build():
    registration=register()
    status_rows,test_opened=stopped()
    inputs={str(REGISTRATION):sha(REGISTRATION)}
    inputs.update({str(p):sha(p) for p,_ in status_rows})
    rows=[]
    for path in sorted(OUT.rglob('*.json')):
        if path.name.startswith('instrumentation_attempt_'):
            kind='fixed_instrumentation'
        elif path.name=='solver_attempts.json':
            kind='solver'
        elif path.name.startswith('attempt_'):
            kind='environment'
        else:
            continue
        stage=stage_for(path)
        if not test_opened:
            assert stage not in ('evaluation_test','timing_test')
        raw=read(path)
        values=normalize_counter(kind,raw)
        row=dict(path=str(path),folder=str(path.parent),kind=kind,stage=stage,
                 sha256=sha(path),values=values)
        rows.append(row)
        inputs[str(path)]=row['sha256']
    assert rows
    # Duplicate meter ids often mean copied archives: attribution must be explicit,
    # not silently doubled or silently dropped.
    meters=[r for r in rows if r['kind'] in ('environment','fixed_instrumentation')]
    assert len({Path(r['path']).name for r in meters})==len(meters), 'Copied attempt identifiers need manual attribution'
    by_folder=defaultdict(list)
    for row in rows:by_folder[row['folder']].append(row)
    issues=[]
    conditions=[]
    instrument_fields=('step_attempts','reset_attempts')
    for folder_name,items in sorted(by_folder.items()):
        folder=Path(folder_name)
        environment=[r for r in items if r['kind']=='environment']
        solvers=[r for r in items if r['kind']=='solver']
        fixed=[r for r in items if r['kind']=='fixed_instrumentation']
        stage=items[0]['stage']
        assert all(r['stage']==stage for r in items)
        if fixed:
            assert stage=='extra_fixed' and not environment and not solvers and len(fixed)==1
            counts=fixed[0]['values']
            instrumentation=read(folder/'instrumentation_completed.json')
            done=read(folder/'completed.json')
            audit=read(folder/'independent_log_audit.json')
            assert instrumentation['passed'] and done['status']=='complete' and audit['passed']
            verify_hashes(instrumentation['hashes'])
            verify_hashes(audit['hashes'])
            assert done['steps']==audit['training_steps']==15000
            assert counts['step_attempts']==counts['step_completed']==audit['logged_steps']
            assert counts['reset_attempts']==counts['reset_completed']==audit['logged_resets']
            assert instrumentation['extra_evaluation_steps']==counts['step_completed']-15000
            record=dict(folder=folder_name,stage=stage,kind='fixed_training',
                        explicit_step_attempts=counts['step_attempts'],
                        explicit_reset_attempts=counts['reset_attempts'],
                        training_steps=15000,final_diagnostic_steps=counts['step_completed']-15000,
                        raw_solve_attempts=None,raw_solve_completed=None,
                        constructor_internal_calls=None,new_gradient_updates=done['updates'],
                        logged_environment_constructions=counts['environment_constructions'])
            for name in ('completed.json','manifest.json','instrumentation_completed.json','independent_log_audit.json'):
                inputs[str(folder/name)]=sha(folder/name)
        else:
            assert len(environment)==len(solvers)==1, ('Unpaired environment/solver ledger',folder_name)
            env,solver=environment[0]['values'],solvers[0]['values']
            done_path=folder/'completed.json'
            if not done_path.exists():
                issues.append(dict(folder=folder_name,reason='Incomplete condition; all attempts charged'))
                record=dict(folder=folder_name,stage=stage,kind='incomplete_condition',
                            explicit_step_attempts=env['step_attempts'],
                            explicit_reset_attempts=env['reset_attempts'],
                            raw_solve_attempts=solver['solve_attempts'],raw_solve_completed=solver['solve_completed'])
            else:
                done=read(done_path)
                assert done['passed']
                verify_hashes(done['hashes'])
                inputs[str(done_path)]=sha(done_path)
                assert solver['solve_attempts']==solver['solve_completed']
                assert solver['warmup_attempts']==env['reset_attempts']
                assert solver['solve_attempts']==env['step_attempts']+env['reset_attempts']+solver['retry_attempts']
                if stage=='bank_generation':
                    assert env['step_attempts']==solver['retry_attempts']==0
                    assert done['resets']==env['reset_attempts']
                    episode_count=0
                else:
                    summary=read(folder/'summary.json')
                    assert summary['solver_counts']==solver
                    assert summary['steps']==env['step_attempts']
                    episode_count=len(summary['episodes'])
                    assert summary['resets']==episode_count==env['reset_attempts']
                    assert sum(e['steps'] for e in summary['episodes'])==env['step_attempts']
                    assert sum(e['retries'] for e in summary['episodes'])==solver['retry_attempts']
                record=dict(folder=folder_name,stage=stage,kind='completed_condition',
                            explicit_step_attempts=env['step_attempts'],
                            explicit_reset_attempts=env['reset_attempts'],episodes=episode_count,
                            raw_solve_attempts=solver['solve_attempts'],raw_solve_completed=solver['solve_completed'],
                            warmup_attempts=solver['warmup_attempts'],retry_attempts=solver['retry_attempts'],
                            constructor_internal_calls=None)
        conditions.append(record)
    stages={}
    for stage in sorted({r['stage'] for r in rows}):
        environment=[r['values'] for r in rows if r['stage']==stage and r['kind']=='environment']
        solvers=[r['values'] for r in rows if r['stage']==stage and r['kind']=='solver']
        fixed=[r['values'] for r in rows if r['stage']==stage and r['kind']=='fixed_instrumentation']
        stages[stage]=dict(environment=sum_rows(environment,ENV_FIELDS),solver=sum_rows(solvers,SOLVER_FIELDS),
                           fixed_instrumentation=sum_rows(fixed,FIXED_FIELDS))
    # Independent totals are also re-derived by the separate review script.
    env_totals={k:sum(g['environment'][k] for g in stages.values()) for k in ENV_FIELDS}
    fixed_totals={k:sum(g['fixed_instrumentation'][k] for g in stages.values()) for k in FIXED_FIELDS}
    solve_totals={k:sum(g['solver'][k] for g in stages.values()) for k in SOLVER_FIELDS}
    totals=dict(recorded_explicit_step_attempts=env_totals['step_attempts']+fixed_totals['step_attempts'],
                recorded_explicit_reset_attempts=env_totals['reset_attempts']+fixed_totals['reset_attempts'],
                instrumented_solver_attempts=solve_totals['solve_attempts'],
                instrumented_solver_completed=solve_totals['solve_completed'],
                instrumented_warmup_attempts=solve_totals['warmup_attempts'],
                instrumented_retry_attempts=solve_totals['retry_attempts'],
                fixed_instrumentation_step_completed=fixed_totals['step_completed'],
                fixed_instrumentation_reset_completed=fixed_totals['reset_completed'],
                all_solver_attempts=None,all_solver_completed=None,
                constructor_internal_calls=None,constructor_internal_solves=None)
    training_budget=read(OUT/'training_delivery/budget.json')
    for part in ('bank_generation','smoke','train'):
        g=training_budget['groups'][part]
        assert stages[part]['environment']['step_attempts']==g['step_attempts']
        assert stages[part]['environment']['reset_attempts']==g['reset_attempts']
        for k in SOLVER_FIELDS:assert stages[part]['solver'][k]==g[k]
    inputs[str(OUT/'training_delivery/budget.json')]=sha(OUT/'training_delivery/budget.json')
    audits=[]
    # Every completed audit run is computational work, including initial validation
    # rechecked by the later full validation audit. Do not deduplicate integration runs.
    for path in sorted(OUT.glob('audit*.json')):
        value=read(path)
        if not value.get('passed') or 'independent_integrations' not in value:
            continue
        verify_hashes(value['hashes'])
        audits.append(dict(path=str(path),independent_integrations=value['independent_integrations'],
                           kind='saved_trace_dynamics_audit'))
        inputs[str(path)]=sha(path)
    for path in sorted((OUT/'extra_fixed').glob('*/independent_log_audit.json')):
        audit=read(path)
        assert audit['passed']
        verify_hashes(audit['hashes'])
        audits.append(dict(path=str(path),independent_integrations=audit['logged_steps'],
                           kind='fixed_training_transition_audit'))
        inputs[str(path)]=sha(path)
    phase_times=[]
    for path,value in status_rows:
        if value.get('started') is not None and value.get('ended') is not None:
            elapsed=value['ended']-value['started']
            assert math.isfinite(elapsed) and elapsed>=0
            phase_times.append(dict(path=str(path),wall_s=elapsed,kind='recorded_controller_interval'))
    summary=dict(passed_accounting=True,complete_budget=not issues,issues=issues,stages=stages,totals=totals,
                 counters=rows,conditions=conditions,known_offline_integrations=audits,
                 known_offline_integration_total=sum(r['independent_integrations'] for r in audits),
                 inherited_models=training_budget['inherited_models'],
                 inherited_terminal_training_steps=training_budget['inherited_terminal_training_steps'],
                 new_fixed_terminal_training_steps=sum(r.get('training_steps',0) for r in conditions),
                 new_fixed_final_diagnostic_steps=sum(r.get('final_diagnostic_steps',0) for r in conditions),
                 recorded_phase_wall_intervals=phase_times,whole_campaign_cpu_s=None,
                 repeat_audit_runs_not_recorded='Only retained completed audit receipts counted; undocumented retries are unknown.',
                 scope='All registered experiment counters, including failed/partial work if present. '
                       'Raw NLP solves of fixed-terminal training and constructor-internal calls are uninstrumented; '
                       'instrumented totals are not whole-study solve totals. Test counters included only after positive '
                       'validation and registered independently reviewed confirmation.',
                 test_trajectories_read=False,test_summary_receipts_read=test_opened,
                 test_phase_included=test_opened,goal_complete=False,
                 hashes=inputs,source_hashes={str(SCRIPTS/n):sha(SCRIPTS/n) for n in SOURCES})
    DEST.mkdir(exist_ok=True)
    frozen(DEST/'ledger.json',summary)
    print(json.dumps(dict(complete_budget=not issues,counter_files=len(rows),conditions=len(conditions),
                          totals=totals,known_offline_integration_total=summary['known_offline_integration_total']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    check() if args.check else register() if args.register else build()
