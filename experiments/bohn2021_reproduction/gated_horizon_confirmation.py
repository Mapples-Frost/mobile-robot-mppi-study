"""Sealed-test preparation and identical all-seed efficacy assessment; never bypass validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE
from gated_horizon_evaluate import grid_model,arm_name,freeze_policies
from gated_horizon_select import load,summarize
from gated_horizon_report import assess,combine,interval,ratio,csvwrite
from gated_horizon_audit import audit_candidate
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read,digest
from run import write
from gated_horizon_amendment import registration as amended_registration,verify as amendment,audit_starts


def verify_hashes(mapping):
    for p,h in mapping.items():assert digest(Path(p))==h


def register():
    freeze();source=Path(__file__).parent
    original=(source/'gated_horizon_timing_audit.py').read_text()
    expected=original.replace("register();phase='smoke' if smoke else 'validation';root=OUT/('timing_'+phase)","assert not smoke,'Test timing audit has no smoke mode'\n    register();phase='test';root=OUT/('timing_'+phase)").replace('n=2 if smoke else 32','n=64')
    assert (source/'gated_horizon_test_timing_audit.py').read_text()==expected
    paths=[Path(__file__).resolve()]+[source/n for n in ('gated_horizon_test_timing_audit.py','gated_horizon_report.py','gated_horizon_evaluate.py','gated_horizon_audit.py','gated_horizon_timing.py')]
    paths += [OUT/n for n in ('protocol.json','effect_registration.json','evaluation_registration.json','timing_registration.json')]
    value=dict(hashes={str(p):digest(p) for p in paths},test_cases_per_task=64,
        test_rule='Same per-seed physical/control/safety/adaptation and cost-or-speed thresholds, paired10000-scene bootstrap and2 timing repeats as validation. All3 seeds, both tasks and both fixed comparator labels. No retuning on test.',
        access='Freeze only after complete independently audited positive validation and all controllers terminal. No automatic unlock from a partial/one-seed gain.',
        audit_reuse='Test timing auditor exactly equals validated smoke/validation auditor except fixed test path,64 cases and disabled smoke. Efficacy math calls the frozen same validation functions.',
        test_access=False)
    p=OUT/'test_execution_registration.json'
    if p.exists():amended_registration(p,value)
    else:
        assert not (OUT/'evaluations/test').exists() and not (OUT/'timing_test').exists();write(p,value)


def freeze_confirmation():
    register();freeze_policies()
    gate_path=OUT/'validation_delivery/effect_gate.json';gate=read(gate_path)
    assert gate['validation_effect_passed'] and gate['timing_audited'] and gate['extra_baselines_audited'],'Positive independently audited validation required'
    verify_hashes(gate['hashes'])
    finish=read(OUT/'validation_finish_status.json');assert finish['complete'] and not finish['active'];verify_hashes(finish['output_hashes'])
    selection=read(OUT/'baseline_selection.json');assert not selection['pending_independent_baselines'];verify_hashes(selection['hashes'])
    jobs=[];paths=[gate_path,OUT/'baseline_selection.json',OUT/'fitted_policy_registration.json',OUT/'audit_validation.json',OUT/'audit_timing_validation.json',OUT/'test_execution_registration.json',amendment()]
    for task in TASKS:
        assert gate['effects'][task]['passed']
        for seed in range(3):
            jobs.append((task,'adaptive',seed,0));paths += [OUT/'train'/('%s_s%d'%(task,seed))/'policy.json']
            source=model_dir(task,'fixed',seed);paths += [source/n for n in ('model.zip','completed.json','manifest.json')]
            for label,family in [('independent','grid'),('matched','matched')]:
                h=selection['nominations'][task][label+'_h'];family='primary' if h==BASE[task] else family
                jobs.append((task,family,seed,h))
                if family=='grid':paths += [grid_model(task,h,seed)/n for n in ('model.zip','completed.json','manifest.json')]
        bank=OUT/'banks'/(task+'_test_bank.json');assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)];paths.append(bank)
    value=json.loads(json.dumps(dict(validation_gate_passed=True,jobs=sorted(set(jobs)),hashes={str(p):digest(p) for p in paths},nominations=selection['nominations'],
        rule='Final independent evidence; no checkpoint, scenario or policy selection after outcomes. Same frozen criteria as validation. All failures retained.')))
    p=OUT/'confirmation_registration.json'
    if p.exists():assert read(p)==value
    else:
        assert not (OUT/'evaluations/test').exists() and not (OUT/'timing_test').exists();write(p,value)
    print('Positive validation verified and exact confirmation artifacts frozen')


def confirmation():
    register();c=read(OUT/'confirmation_registration.json');assert c['validation_gate_passed'];verify_hashes(c['hashes']);return c


def audit():
    c=confirmation();assert read(OUT/'test_status.json')['complete'];groups=[];hashes={}
    for task,family,seed,h in c['jobs']:
        folder=OUT/'evaluations/test'/task/arm_name(family,seed,h)
        bank=OUT/'banks'/(task+'_test_bank.json');cases=read(bank)['cases'];assert len(cases)==64
        a=audit_candidate(folder,task,cases,fixed_h=None if family=='adaptive' else h);a.update(task=task,family=family,seed=seed,h=h);groups.append(a)
        summary=read(folder/'summary.json');d=read(folder/'completed.json');assert summary['episodes']==d['result']['episodes'] and summary['solver_counts']==a['counts']
        hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    value=dict(passed=True,conditions=len(groups),episodes=sum(g['episodes'] for g in groups),steps=sum(g['steps'] for g in groups),
        maximum_dynamics_error=max(g['maximum_dynamics_error'] for g in groups),groups=groups,hashes=hashes,confirmation_hash=digest(OUT/'confirmation_registration.json'),source_hash=digest(Path(__file__)))
    value['shared_initialization']=audit_starts(c['jobs'],'test')
    write(OUT/'audit_test.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('groups','hashes')},indent=2))


def timing_rows(a):
    rows=[]
    for repeat in range(2):
        p=OUT/'timing_test'/('r%d_%s_%s'%(repeat,a['task'],arm_name(a['family'],a['seed'],a['h'])))
        d=read(p/'completed.json');assert d['passed'] and d['exact_replay'];verify_hashes(d['hashes']);rows.append(read(p/'summary.json')['episodes'])
    return rows


def report():
    c=confirmation();hashes={str(OUT/'confirmation_registration.json'):digest(OUT/'confirmation_registration.json')}
    for name in ('audit_test.json','audit_timing_test.json'):
        a=read(OUT/name);assert a['passed'];verify_hashes(a['hashes']);hashes[str(OUT/name)]=digest(OUT/name)
    output=OUT/'test_delivery';output.mkdir(exist_ok=True);conditions=[];episodes=[];effects={};comparisons=[];paired=[]
    for task,family,seed,h in c['jobs']:
        a=load(task,family,seed,h,split='test');assert a;conditions.append(summarize(a));episodes += [dict(task=task,family=family,seed=seed,h=h,**e) for e in a['episodes']]
        p=Path(a['path'])/'completed.json';hashes[str(p)]=digest(p)
    for task,n in c['nominations'].items():
        effects[task]={}
        for label,family in [('independent','grid'),('matched','matched')]:
            rows=[];deltas=[]
            for seed in range(3):
                a=load(task,'adaptive',seed,0,split='test');b=load(task,family,seed,n[label+'_h'],split='test');assert a and b
                row=assess(a,b,ratio(timing_rows(a),timing_rows(b)));row.update(task=task,seed=seed,comparator=label,fixed_h=b['h'],adaptive=summarize(a),fixed=summarize(b));rows.append(row);comparisons.append(row)
                delta=[]
                for aa,bb in zip(a['episodes'],b['episodes']):
                    delta.append(aa['total_cost']-bb['total_cost']);paired.append(dict(task=task,seed=seed,comparator=label,case=aa['case'],adaptive_cost=aa['total_cost'],fixed_cost=bb['total_cost'],cost_difference=delta[-1],
                        adaptive_success=aa['success'],fixed_success=bb['success'],adaptive_constraint=aa['constraint'],fixed_constraint=bb['constraint'],
                        adaptive_initial_failures=aa['initial_failed_steps'],fixed_initial_failures=bb['initial_failed_steps'],adaptive_final_failures=aa['solver_failure_steps'],fixed_final_failures=bb['solver_failure_steps']))
                deltas.append(delta)
            effects[task][label]=dict(combine(rows),paired_scene_interval=interval(np.asarray(deltas)))
        effects[task]['passed']=all(effects[task][label]['passed'] for label in ('independent','matched'))
    value=dict(independent_effect_passed=all(e['passed'] for e in effects.values()),effects=effects,comparisons=comparisons,hashes=hashes,source_hash=digest(Path(__file__)),
        scope='Single frozen independent confirmation after validation. An effect pass supports this modified finite-search policy on reconstructed tasks, not exact original SAC reproduction. Full goal completion still requires delivery and requirements audit.')
    write(output/'effect_gate.json',value);csvwrite(output/'all_conditions.csv',conditions);csvwrite(output/'all_episodes.csv',episodes);csvwrite(output/'paired_episodes.csv',paired)
    (output/'report_CN.md').write_text('# 门控时域：封存测试结果\n\n独立效果门槛：'+str(value['independent_effect_passed'])+'。全部任务、种子、基线、失败与逐回合差异见同目录CSV及effect_gate.json。\n\n这是有限门控策略直接搜索这一改进方法在重建任务上的证据，不等于原论文SAC逐数字复现；不得通过测试重新选模型或场景。\n')
    print(json.dumps(dict(independent_effect_passed=value['independent_effect_passed'],effects=effects),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','freeze','audit','report'],required=True);a=ap.parse_args()
    if a.mode=='register':register()
    elif a.mode=='freeze':freeze_confirmation()
    elif a.mode=='audit':audit()
    else:report()
