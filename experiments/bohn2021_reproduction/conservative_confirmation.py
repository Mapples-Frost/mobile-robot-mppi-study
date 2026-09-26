"""Freeze confirmation inputs only after the registered validation effect gate.

Registering this analysis code does not unseal test outcomes. The freeze command
requires completed validation and serial timing; it never starts evaluation.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS,BASE,verify
from conservative_iteration_evaluate import arm_name,grid_model
from conservative_iteration_report import bootstrap,export_csv
from conservative_iteration_select import safe
from conservative_iteration_timing import jobs
from conservative_protocol_audit import main as audit_protocol
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read,digest
from run import write
from fixed_policy_branches import metrics
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context


def register_analysis():
    files=[Path(__file__),Path(__file__).with_name('conservative_iteration_report.py'),
           Path(__file__).with_name('conservative_iteration_timing.py')]
    spec=dict(hashes={str(p):digest(p) for p in files},scenes_per_task=48,
        statistical_unit='Paired full scenarios, all3 training seeds retained. 10000 bootstrap resamples; report per-seed and pooled differences conditional on these models.',
        cost_rule='Every seed total cost improves>=3% of abs baseline mean; each seed paired95% interval upper<0.',
        time_rule='Every seed mean decision time ratio<=0.9 and paired95% upper<1, each of two repeat ratios<1; total cost noninferiority2% of abs baseline mean.',
        safety_rule='Each seed preserves success count, constraint count, solver-failure rate, physical-plus-constraint mean cost within2% abs baseline mean.',
        adaptation_rule='Each seed must actually vary H within episodes; report counts, not a separate new outcome search.',
        comparators='Both validation-selected independently trained fixed H and selected matched-terminal fixed H.',
        overall='Both tasks pass independently. Data audit passing is not efficacy. Failed tests are reported and become observed historical data; never reused as fresh confirmation after intervention.',
        timing_stability='Additional conservative requirement: acceleration must have ratio<1 in both serial repetitions for every seed. No outcome has been used to set this rule.')
    path=OUT/'confirmation_analysis_registration.json'
    if path.exists():assert read(path)==spec
    else:write(path,spec)


def freeze_confirmation():
    verify();audit_protocol();register_analysis()
    from conservative_iteration_report import report
    report()
    gate_path=OUT/'validation_delivery/effect_gate.json';gate=read(gate_path)
    assert gate['validation_effect_passed'],'Validation effect failed: do not open test'
    for p,h in gate['hashes'].items():assert digest(Path(p))==h
    timing=read(OUT/'timing_validation/completed.json');assert timing['passed']
    for p,h in timing['hashes'].items():assert digest(Path(p))==h
    selected=jobs('validation')
    paths=[gate_path,OUT/'validation_selection.json',OUT/'baseline_selection.json',
           OUT/'timing_validation/completed.json',OUT/'confirmation_analysis_registration.json',
           OUT/'collection_inputs_sha256.json',OUT/'banks/hashes.json']
    for task,family,seed,h in selected:
        source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
        paths += [source/'model.zip',source/'completed.json']
        if family.startswith('round'):
            d=OUT/('%s_s%d_r%d'%(task,seed,int(family[-1])))
            paths += [d/'policy.json',d/'fit_completed.json',d/'collection_audit.json']
    paths += [OUT/'banks'/(task+'_test.json') for task in TASKS]
    hashes={str(p):digest(p) for p in set(paths)}
    path=OUT/'confirmation_registration.json'
    data=dict(validation_gate_passed=True,jobs=[list(j) for j in selected],hashes=hashes,
        selections=read(OUT/'validation_selection.json')['selections'],
        analysis_registration_hash=digest(OUT/'confirmation_analysis_registration.json'))
    if path.exists():assert read(path)==data
    else:
        assert not (OUT/'evaluations/test').exists(),'Existing test outcomes need contamination review'
        write(path,data)
    print('Confirmation inputs frozen; no test outcomes evaluated by this command')


def load_condition(task,family,seed,h):
    folder=OUT/'evaluations/test'/task/arm_name(family,seed,h)
    done=read(folder/'completed.json');assert done['passed']
    for p,value in done['hashes'].items():assert digest(Path(p))==value
    bank=read(OUT/'banks'/(task+'_test.json'));assert len(bank['cases'])==48
    summary=read(folder/'summary.json');assert len(summary['episodes'])==48
    for cid,(case,episode) in enumerate(zip(bank['cases'],summary['episodes'])):
        trace=read(folder/('trace_%02d.json'%cid));audit_trace(task,case,trace);audit_context(task,case,trace,0)
        actual=metrics(task,trace)
        assert all(episode[k]==v for k,v in actual.items()) and episode['case']==cid
    rows=summary['episodes'];timings=[];all_times=[];hashes={str(folder/'completed.json'):digest(folder/'completed.json')}
    for repeat in range(2):
        td=OUT/'timing_test'/('r%d_%s_%s'%(repeat,task,arm_name(family,seed,h)))
        tm=read(td/'completed.json');assert tm['passed'] and tm['exact_replay']
        for p,value in tm['hashes'].items():assert digest(Path(p))==value
        tr=read(td/'summary.json')['episodes'];assert len(tr)==48
        for cid,e in enumerate(tr):
            expected=read(folder/('trace_%02d.json'%cid));trace=read(td/('trace_%02d.json'%cid))
            assert [{k:v for k,v in r.items() if k!='timing'} for r in trace]==expected
            ts=np.array([r['timing']['decision_s'] for r in trace]);assert np.isfinite(ts).all() and np.all(ts>=0)
            np.testing.assert_allclose(e['decision_total_s'],ts.sum(),rtol=0,atol=1e-10)
            all_times.extend(ts.tolist())
        timings.append(tr);hashes[str(td/'completed.json')]=digest(td/'completed.json')
    return dict(task=task,family=family,seed=seed,h=h,episodes=rows,timings=timings,
        total_cost=float(np.mean([e['total_cost'] for e in rows])),
        physical_constraint_cost=float(np.mean([e['physical_constraint_cost'] for e in rows])),
        performance_cost=float(np.mean([e['performance_cost'] for e in rows])),
        constraint_cost=float(np.mean([e['constraint_cost'] for e in rows])),
        h_penalty=float(np.mean([e['h_penalty'] for e in rows])),
        success=sum(e['success'] for e in rows),constraints=sum(e['constraint'] for e in rows),
        solver_failure_steps=sum(e['solver_failure_steps'] for e in rows),steps=sum(e['steps'] for e in rows),
        failure_rate=sum(e['solver_failure_steps'] for e in rows)/sum(e['steps'] for e in rows),
        mean_horizon=float(np.average([e['mean_horizon'] for e in rows],weights=[e['steps'] for e in rows])),
        horizon_switches=sum(e['horizon_switches'] for e in rows),
        decision_mean_s=float(np.mean(all_times)),decision_median_s=float(np.median(all_times)),
        decision_p95_s=float(np.percentile(all_times,95)),deadline_exceed_rate=float(np.mean(np.asarray(all_times)>(.1 if task=='vehicle' else .04))),
        decision_episode_mean_s=float(np.mean([e['decision_total_s'] for r in timings for e in r])),hashes=hashes)


def timing_difference(a,b):
    at=np.mean([[r['decision_total_s'] for r in rr] for rr in a],axis=0)
    bt=np.mean([[r['decision_total_s'] for r in rr] for rr in b],axis=0)
    an=np.array([r['steps'] for r in a[0]]);bn=np.array([r['steps'] for r in b[0]])
    assert len(at)==len(bt)==48
    ix=np.random.RandomState(2609247201).randint(48,size=(10000,48))
    samples=(at[ix].sum(axis=1)/an[ix].sum(axis=1))/(bt[ix].sum(axis=1)/bn[ix].sum(axis=1))
    repeats=[(sum(r['decision_total_s'] for r in ar)/sum(r['steps'] for r in ar))/
             (sum(r['decision_total_s'] for r in br)/sum(r['steps'] for r in br)) for ar,br in zip(a,b)]
    return dict(ratio=float((at.sum()/an.sum())/(bt.sum()/bn.sum())),
                lower=float(np.percentile(samples,2.5)),upper=float(np.percentile(samples,97.5)),repeat_ratios=repeats)


def report_confirmation():
    verify();audit_protocol();register_analysis()
    registration=read(OUT/'confirmation_registration.json');assert registration['validation_gate_passed']
    for p,value in registration['hashes'].items():assert digest(Path(p))==value
    assert read(OUT/'test_status.json')['complete'] and read(OUT/'timing_test/completed.json')['passed']
    data={};hashes={};seed_rows=[];episode_rows=[]
    for j in registration['jobs']:
        key=tuple(j);d=load_condition(*j);data[key]=d;hashes.update(d['hashes'])
        seed_rows.append({k:v for k,v in d.items() if k not in ('episodes','timings','hashes')})
        for e in d['episodes']:episode_rows.append(dict(task=j[0],family=j[1],seed=j[2],h=j[3],**e))
    comparisons=[];effects={};paired=[]
    for task,s in registration['selections'].items():
        family='round%d'%s['round'];effects[task]={};task_pass=[]
        for comparator in ('independent','matched'):
            h=s[comparator+'_h'];bf='grid' if comparator=='independent' else 'matched'
            if h==BASE[task]:bf='primary'
            differences=[];cost_gates=[];time_gates=[];safety_gates=[];adaptation_gates=[]
            for seed in range(3):
                a=data[(task,family,seed,0)];b=data[(task,bf,seed,h)]
                delta=np.array([ae['total_cost']-be['total_cost'] for ae,be in zip(a['episodes'],b['episodes'])]);differences.append(delta)
                interval=bootstrap(delta,seed=2609247200)
                relative=interval['mean']/max(abs(b['total_cost']),1e-12)
                timing=timing_difference(a['timings'],b['timings'])
                control=a['physical_constraint_cost']<=b['physical_constraint_cost']+.02*abs(b['physical_constraint_cost'])
                safety_gates.append(safe(a,b) and control)
                adaptation_gates.append(a['horizon_switches']>0)
                cost_gates.append(relative<=-.03 and interval['upper']<0)
                time_gates.append(timing['ratio']<=.9 and timing['upper']<1 and all(r<1 for r in timing['repeat_ratios']) and
                    a['total_cost']<=b['total_cost']+.02*abs(b['total_cost']))
                comparisons.append(dict(task=task,comparator=comparator,seed=seed,adaptive_round=s['round'],fixed_h=h,
                    total_cost_difference=interval,relative_cost_difference=relative,timing=timing,
                    safety_noninferior=safety_gates[-1],physical_cost_noninferior=control,
                    cost_gain=cost_gates[-1],time_gain=time_gates[-1],within_episode_adaptation=adaptation_gates[-1]))
                for ae,be in zip(a['episodes'],b['episodes']):
                    paired.append(dict(task=task,comparator=comparator,seed=seed,case=ae['case'],adaptive_cost=ae['total_cost'],fixed_cost=be['total_cost'],
                        difference=ae['total_cost']-be['total_cost'],adaptive_success=ae['success'],fixed_success=be['success'],
                        adaptive_constraint=ae['constraint'],fixed_constraint=be['constraint'],adaptive_solver_failures=ae['solver_failure_steps'],fixed_solver_failures=be['solver_failure_steps']))
            passed=all(safety_gates) and all(adaptation_gates) and (all(cost_gates) or all(time_gates));task_pass.append(passed)
            effects[task][comparator]=dict(passed=passed,all_seed_safety=all(safety_gates),all_seed_adaptation=all(adaptation_gates),
                per_seed_cost_gain=cost_gates,per_seed_time_gain=time_gates,paired_scene_interval=bootstrap(np.array(differences),seed=2609247200))
        effects[task]['passed']=all(task_pass)
    dest=OUT/'confirmation_delivery';dest.mkdir(exist_ok=True)
    export_csv(dest/'all_test_seeds.csv',seed_rows);export_csv(dest/'all_test_episodes.csv',episode_rows);export_csv(dest/'paired_test_episodes.csv',paired)
    result=dict(data_audit_passed=True,core_effect_passed=len(effects)==2 and all(e['passed'] for e in effects.values()),
        effects=effects,comparisons=comparisons,hashes=hashes,registration_hash=digest(OUT/'confirmation_registration.json'),
        source_hash=digest(Path(__file__)),scope='Independent test of conservative five-step supervised policy iteration extension, not original author SAC. Full project completion still requires artifact/budget/claim audit.')
    write(dest/'effect_audit.json',result)
    lines=['# 保守五步策略迭代：独立测试','',
        '这是允许方法改动的扩展实验，不能标成原作者 SAC 重建成功。测试场景在模型选择前封存，仅在冻结的验证门槛通过后开启。','',
        '|任务|方法|种子|H|总成本|物理+约束|成功/48|约束|求解失败/步|均值H|决策均值ms|p95ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        lines.append('|%s|%s|%d|%d|%.4f|%.4f|%d|%d|%d/%d|%.2f|%.3f|%.3f|'%(r['task'],r['family'],r['seed'],r['h'],r['total_cost'],
            r['physical_constraint_cost'],r['success'],r['constraints'],r['solver_failure_steps'],r['steps'],r['mean_horizon'],1000*r['decision_mean_s'],1000*r['decision_p95_s']))
    lines+=['','预定核心效果门槛：'+str(result['core_effect_passed'])+'。数据审计通过不能替代效果门槛。','',
        '所有种子、失败与配对场景见 CSV。时间含实际部署特征/选择和控制器调用，不含模拟、磁盘、审计及 reset；reset 另存。作者最大50步屏蔽 NLP 保留；短 H 不自动意味着加速。','',
        '配对区间对三份已训练模型条件化，不能代表未知训练种子总体。WSL和宿主机调度仍有限制；同两次重复不当作新场景。各方法共享热身前初态，独立终端权重会影响未计分 H50 热身；同终端固定比较器共享该热身。','',
        '若门槛失败，本测试即成为已观察历史数据。任何后续方法干预都须另建新验证/封存测试，不能复用本测试宣称新的独立成功。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(core_effect_passed=result['core_effect_passed'],effects=effects),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register-analysis','freeze','report'],required=True);a=ap.parse_args()
    if a.mode=='register-analysis':register_analysis()
    elif a.mode=='freeze':freeze_confirmation()
    else:report_confirmation()
