"""All-seed raw validation outputs and preregistered effect gate, no test unlock."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE
from gated_horizon_evaluate import jobs,arm_name
from gated_horizon_select import load,safe,summarize
from paper_h_soft_probe import read,digest
from run import write


def register():
    freeze();p=OUT/'effect_registration.json'
    value=dict(hashes={str(p):digest(p) for p in [Path(__file__).resolve(),OUT/'protocol.json',OUT/'evaluation_registration.json',OUT/'timing_registration.json']},
        draws=10000,scene_resample_seed=2609257100,ratio_resample_seed=2609257101,
        interpretation='All3 seeds must pass safety/control/adaptation and the same gain route per task/comparator. Both tasks and both comparator labels required. Timing completion required even for a cost-route result; no claim before audit. No multiplicity-adjusted discovery claim; validation selection must be confirmed on sealed test.',
        test_unlocked=False)
    if p.exists():assert read(p)==value
    else:
        assert not (OUT/'evaluations').exists(),'Register effect code before outcomes';write(p,value)


def interval(values):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[None,:]
    ix=np.random.RandomState(2609257100).randint(values.shape[-1],size=(10000,values.shape[-1]))
    means=values[:,ix].mean(axis=(0,2))
    return dict(mean=float(values.mean()),lower=float(np.percentile(means,2.5)),upper=float(np.percentile(means,97.5)))


def ratio(adaptive,fixed):
    assert len(adaptive)==len(fixed)==2
    n=len(adaptive[0]);assert all(len(r)==n for r in adaptive+fixed)
    assert all([e['case'] for e in r]==list(range(n)) for r in adaptive+fixed)
    at=np.array([[e['decision_total_s'] for e in r] for r in adaptive]);bt=np.array([[e['decision_total_s'] for e in r] for r in fixed])
    astep=np.array([e['steps'] for e in adaptive[0]]);bstep=np.array([e['steps'] for e in fixed[0]])
    assert all(np.array_equal(astep,[e['steps'] for e in r]) for r in adaptive)
    assert all(np.array_equal(bstep,[e['steps'] for e in r]) for r in fixed)
    assert np.all(at>0) and np.all(bt>0)
    ix=np.random.RandomState(2609257101).randint(n,size=(10000,n));am=at.mean(axis=0);bm=bt.mean(axis=0)
    ratios=(am[ix].sum(axis=1)/astep[ix].sum(axis=1))/(bm[ix].sum(axis=1)/bstep[ix].sum(axis=1))
    return dict(mean_ratio=float((am.sum()/astep.sum())/(bm.sum()/bstep.sum())),lower=float(np.percentile(ratios,2.5)),upper=float(np.percentile(ratios,97.5)),
        repeat_ratios=[float((a.sum()/astep.sum())/(b.sum()/bstep.sum())) for a,b in zip(at,bt)])


def assess(a,b,timing):
    assert [e['case'] for e in a['episodes']]==[e['case'] for e in b['episodes']]
    delta=np.array([aa['total_cost']-bb['total_cost'] for aa,bb in zip(a['episodes'],b['episodes'])]);ci=interval(delta)
    relative=ci['mean']/max(abs(b['total_cost']),1e-12)
    control=a['physical_constraint_cost']<=b['physical_constraint_cost']+.02*abs(b['physical_constraint_cost'])
    total=a['total_cost']<=b['total_cost']+.02*abs(b['total_cost'])
    return dict(cost_difference=ci,relative_cost_difference=relative,safe=safe(a,b),control_noninferior=control,total_noninferior=total,
        adapted=a['switches']>0,cost_gain=relative<=-.03 and ci['upper']<0,
        time_gain=timing is not None and timing['mean_ratio']<=.9 and timing['upper']<1 and all(r<1 for r in timing['repeat_ratios']) and total,timing=timing)


def combine(rows):
    assert len(rows)==3
    safety=all(r['safe'] and r['control_noninferior'] for r in rows);adaptation=all(r['adapted'] for r in rows)
    cost=all(r['cost_gain'] for r in rows);timing=all(r['time_gain'] for r in rows)
    complete=all(r['timing'] is not None for r in rows)
    return dict(passed=safety and adaptation and complete and (cost or timing),safe_all_seeds=safety,adapted_all_seeds=adaptation,
        cost_route_all_seeds=cost,time_route_all_seeds=timing,timing_complete=complete)


def timing_rows(a):
    rows=[]
    for repeat in range(2):
        folder=OUT/'timing_validation'/('r%d_%s_%s'%(repeat,a['task'],arm_name(a['family'],a['seed'],a['h'])))
        if not (folder/'completed.json').exists():return None
        done=read(folder/'completed.json');assert done['passed'] and done['exact_replay']
        for p,h in done['hashes'].items():assert digest(Path(p))==h
        rows.append(read(folder/'summary.json')['episodes'])
    return rows


def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report():
    register();assert read(OUT/'validation_status.json')['complete'];audit=read(OUT/'audit_validation.json');assert audit['passed']
    for p,h in audit['hashes'].items():assert digest(Path(p))==h
    selection=read(OUT/'baseline_selection.json');assert not selection['pending_independent_baselines']
    output=OUT/'validation_delivery';output.mkdir(exist_ok=True)
    conditions=list(jobs())
    for task,n in selection['nominations'].items():
        if n['independent_h']!=BASE[task]:conditions += [(task,'grid',seed,n['independent_h']) for seed in (1,2)]
    all_rows=[];all_episodes=[];hashes={}
    for task,family,seed,h in sorted(set(conditions)):
        a=load(task,family,seed,h);assert a
        all_rows.append(summarize(a));all_episodes += [dict(task=task,family=family,seed=seed,h=h,**e) for e in a['episodes']]
        p=Path(a['path'])/'completed.json';hashes[str(p)]=digest(p)
    csvwrite(output/'all_conditions.csv',all_rows);csvwrite(output/'all_episodes.csv',all_episodes)
    effects={};comparisons=[];paired=[]
    for task,n in selection['nominations'].items():
        effects[task]={}
        for label,family in [('independent','grid'),('matched','matched')]:
            seeds=[];deltas=[]
            for seed in range(3):
                a=load(task,'adaptive',seed,0);b=load(task,family,seed,n[label+'_h']);assert a and b
                at=timing_rows(a);bt=timing_rows(b);timing=ratio(at,bt) if at is not None and bt is not None else None
                row=assess(a,b,timing);row.update(task=task,comparator=label,seed=seed,fixed_h=b['h'],adaptive=summarize(a),fixed=summarize(b))
                seeds.append(row);comparisons.append(row);delta=[]
                for aa,bb in zip(a['episodes'],b['episodes']):
                    delta.append(aa['total_cost']-bb['total_cost']);paired.append(dict(task=task,comparator=label,seed=seed,case=aa['case'],adaptive_cost=aa['total_cost'],fixed_cost=bb['total_cost'],cost_difference=delta[-1],
                        adaptive_success=aa['success'],fixed_success=bb['success'],adaptive_constraint=aa['constraint'],fixed_constraint=bb['constraint'],
                        adaptive_initial_failures=aa['initial_failed_steps'],fixed_initial_failures=bb['initial_failed_steps'],adaptive_final_failures=aa['solver_failure_steps'],fixed_final_failures=bb['solver_failure_steps']))
                deltas.append(delta)
            effects[task][label]=dict(combine(seeds),paired_scene_interval=interval(deltas))
        effects[task]['passed']=all(effects[task][label]['passed'] for label in ('independent','matched'))
    # A passing numeric comparison alone does not certify timing bookkeeping or extra baseline traces.
    timing_audit_path=OUT/'audit_timing_validation.json';timing_audited=False
    if timing_audit_path.exists():
        timing_audit=read(timing_audit_path);assert timing_audit['passed']
        for p,h in timing_audit['hashes'].items():assert digest(Path(p))==h
        timing_audited=True;hashes[str(timing_audit_path)]=digest(timing_audit_path)
    extra_audited=all(str(OUT/'evaluations/validation'/task/arm_name('grid',s,n['independent_h'])/'completed.json') in audit['hashes'] for task,n in selection['nominations'].items() if n['independent_h']!=BASE[task] for s in (1,2))
    gate=dict(validation_effect_passed=all(e['passed'] for e in effects.values()) and timing_audited and extra_audited,effects=effects,comparisons=comparisons,
        timing_audited=timing_audited,extra_baselines_audited=extra_audited,independent_test_unlocked=False,hashes=hashes,source_hash=digest(Path(__file__)))
    write(output/'effect_gate.json',gate);csvwrite(output/'paired_episodes.csv',paired)
    flat=[]
    for r in comparisons:
        flat.append(dict(task=r['task'],comparator=r['comparator'],seed=r['seed'],fixed_h=r['fixed_h'],safe=r['safe'],control_noninferior=r['control_noninferior'],adapted=r['adapted'],
            cost_delta=r['cost_difference']['mean'],cost_lower=r['cost_difference']['lower'],cost_upper=r['cost_difference']['upper'],relative_cost=r['relative_cost_difference'],
            time_ratio=None if r['timing'] is None else r['timing']['mean_ratio'],time_lower=None if r['timing'] is None else r['timing']['lower'],time_upper=None if r['timing'] is None else r['timing']['upper'],cost_gain=r['cost_gain'],time_gain=r['time_gain']))
    csvwrite(output/'all_comparisons.csv',flat)
    lines=['# 门控时域：全部验证结果','', '验证门槛：%s；计时审计：%s；测试未解封。'%(gate['validation_effect_passed'],timing_audited),'',
        '所有三种子和两类固定基线均保留。区间为10,000次配对场景重采样，对实际训练模型条件化；两个计时重复不是新增独立场景。平均H不作为速度证据。','',
        '|任务|方法|种子|H|总成本|物理+约束|成功/32|约束|最初失败|最终失败|控制步|H切换|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in all_rows:lines.append('|%s|%s|%d|%d|%.6f|%.6f|%d|%d|%d|%d|%d|%d|'%(a['task'],a['family'],a['seed'],a['h'],a['total_cost'],a['physical_constraint_cost'],a['success'],a['constraints'],a['initial_failures'],a['final_failures'],a['steps'],a['switches']))
    lines+=['','当前为验证结果，不代表独立测试效果。成功为车辆到达或倒立摆无越界存活至时限，后者不是精确调节证明。失败、提前终止与额外重试全部保留。','']
    (output/'report_CN.md').write_text('\n'.join(lines));print(json.dumps(dict(gate=gate['validation_effect_passed'],effects=effects),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    if a.register:register()
    else:report()
