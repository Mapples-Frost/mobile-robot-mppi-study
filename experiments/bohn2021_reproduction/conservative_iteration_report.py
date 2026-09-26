"""All-seed validation accounting and registered effect gate; no test access."""
import csv
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS,BASE,HS
from conservative_iteration_evaluate import jobs,arm_name
from conservative_iteration_select import load,safe
from paper_h_soft_probe import read,digest
from run import write


def bootstrap(values,seed=2609247100):
    """Paired differences: scenes on last axis, seeds retained as a block."""
    values=np.asarray(values,float)
    if values.ndim==1:values=values[None,:]
    rng=np.random.RandomState(seed)
    samples=rng.randint(values.shape[-1],size=(10000,values.shape[-1]))
    means=values[:,samples].mean(axis=(0,2))
    return dict(mean=float(values.mean()),lower=float(np.percentile(means,2.5)),upper=float(np.percentile(means,97.5)))


def timing_rows(task,family,seed,h):
    rows=[]
    for repeat in range(2):
        folder=OUT/'timing_validation'/('r%d_%s_%s'%(repeat,task,arm_name(family,seed,h)))
        if not (folder/'completed.json').exists():return None
        done=read(folder/'completed.json');assert done['passed'] and done['exact_replay']
        for p,value in done['hashes'].items():assert digest(Path(p))==value
        rows.append(read(folder/'summary.json')['episodes'])
    return rows


def time_ratio(adaptive,baseline):
    # Same scenarios are resampled together across the two timing repetitions.
    a_time=np.asarray([[e['decision_total_s'] for e in r] for r in adaptive]).mean(axis=0)
    b_time=np.asarray([[e['decision_total_s'] for e in r] for r in baseline]).mean(axis=0)
    a_steps=np.asarray([e['steps'] for e in adaptive[0]]);b_steps=np.asarray([e['steps'] for e in baseline[0]])
    assert len(a_time)==len(b_time)==24
    rng=np.random.RandomState(2609247101);ix=rng.randint(24,size=(10000,24))
    ratios=(a_time[ix].sum(axis=1)/a_steps[ix].sum(axis=1))/(b_time[ix].sum(axis=1)/b_steps[ix].sum(axis=1))
    repeat_ratio=[]
    for aa,bb in zip(adaptive,baseline):
        repeat_ratio.append((sum(r['decision_total_s'] for r in aa)/sum(r['steps'] for r in aa))/
                            (sum(r['decision_total_s'] for r in bb)/sum(r['steps'] for r in bb)))
    return dict(mean_ratio=float((a_time.sum()/a_steps.sum())/(b_time.sum()/b_steps.sum())),
                lower=float(np.percentile(ratios,2.5)),upper=float(np.percentile(ratios,97.5)),repeat_ratios=repeat_ratio)


def export_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report():
    assert read(OUT/'validation_status.json')['complete']
    output=OUT/'validation_delivery';output.mkdir(exist_ok=True)
    seeds=[];episodes=[];hashes={}
    for task,family,seed,h in jobs():
        result=load(task,family,seed,h);assert result
        seeds.append(dict(task=task,family=family,seed=seed,h=h,total_cost=result['total_cost'],
            physical_constraint_cost=result['physical_constraint_cost'],success=result['success'],
            constraints=result['constraints'],solver_failure_steps=result['failed_steps'],solver_failure_rate=result['failure_rate'],
            steps=result['steps'],mean_horizon=np.average([r['mean_horizon'] for r in result['episodes']],weights=[r['steps'] for r in result['episodes']]),
            total_horizon_switches=sum(r['horizon_switches'] for r in result['episodes'])))
        for e in result['episodes']:
            episodes.append(dict(task=task,family=family,seed=seed,h=h,**e))
        p=Path(result['path'])/'completed.json';hashes[str(p)]=digest(p)
    export_csv(output/'all_validation_seeds.csv',seeds);export_csv(output/'all_validation_episodes.csv',episodes)
    selection_path=OUT/'validation_selection.json'
    selection=read(selection_path) if selection_path.exists() else None
    comparisons=[];effects={};paired=[]
    if selection:
        for task,s in selection['selections'].items():
            chosen=s['round'];effects[task]=dict(selected_round=chosen,passed=False)
            if chosen is None:continue
            family='round%d'%chosen
            comparator_gates=[]
            for comparator in ('independent','matched'):
                bh=s[comparator+'_h'];bf='grid' if comparator=='independent' else 'matched'
                if bh==BASE[task]:bf='primary'
                differences=[];cost_ok=[];time_ok=[];safety_ok=[];adaptation=[]
                for seed in range(3):
                    a=load(task,family,seed,0);b=load(task,bf,seed,bh)
                    assert a and b
                    delta=np.array([aa['total_cost']-bb['total_cost'] for aa,bb in zip(a['episodes'],b['episodes'])])
                    interval=bootstrap(delta);differences.append(delta)
                    relative=interval['mean']/max(abs(b['total_cost']),1e-12)
                    control_ok=a['physical_constraint_cost']<=b['physical_constraint_cost']+.02*abs(b['physical_constraint_cost'])
                    safety_ok.append(safe(a,b) and control_ok)
                    changed=sum(e['horizon_switches'] for e in a['episodes'])>0
                    adaptation.append(changed)
                    timing_a=timing_rows(task,family,seed,0);timing_b=timing_rows(task,bf,seed,bh)
                    timing=time_ratio(timing_a,timing_b) if timing_a is not None and timing_b is not None else None
                    cost_ok.append(relative<=-.03 and interval['upper']<0)
                    time_ok.append(timing is not None and timing['mean_ratio']<=.9 and timing['upper']<1 and
                        a['total_cost']<=b['total_cost']+.02*abs(b['total_cost']))
                    comparisons.append(dict(task=task,comparator=comparator,seed=seed,adaptive_round=chosen,fixed_h=bh,
                        cost_difference=interval,relative_cost_difference=relative,safe=safe(a,b),control_noninferior=control_ok,
                        within_episode_adaptation=changed,timing=timing))
                    for aa,bb in zip(a['episodes'],b['episodes']):
                        paired.append(dict(task=task,comparator=comparator,seed=seed,case=aa['case'],
                            adaptive_cost=aa['total_cost'],fixed_cost=bb['total_cost'],cost_difference=aa['total_cost']-bb['total_cost'],
                            adaptive_success=aa['success'],fixed_success=bb['success'],adaptive_constraint=aa['constraint'],fixed_constraint=bb['constraint'],
                            adaptive_solver_failures=aa['solver_failure_steps'],fixed_solver_failures=bb['solver_failure_steps']))
                pooled=bootstrap(np.asarray(differences))
                passed=all(safety_ok) and all(adaptation) and (all(cost_ok) or all(time_ok))
                comparator_gates.append(passed)
                effects[task][comparator]=dict(passed=passed,safe_all_seeds=all(safety_ok),state_adaptation_all_seeds=all(adaptation),
                    cost_criterion_per_seed=cost_ok,timing_criterion_per_seed=time_ok,paired_scene_interval=pooled)
            effects[task]['passed']=all(comparator_gates)
    if paired:export_csv(output/'selected_paired_episodes.csv',paired)
    gate=dict(validation_effect_passed=len(effects)==2 and all(e['passed'] for e in effects.values()),effects=effects,
        comparisons=comparisons,selection_available=selection is not None,independent_test_unlocked=False,
        inference_scope='Validation-only; 10000 paired scene resamples conditional on fitted seeds. Does not establish independent reproduction.',
        hashes=hashes,report_source_hash=digest(Path(__file__)))
    write(output/'effect_gate.json',gate)
    lines=['# 保守策略迭代：完整验证结果','',
        '本文件仅报告新验证集，未读取测试结果。所有种子与固定 H 候选保留，表中的 H 是代价代理，实际速度只根据单独串行计时。','',
        '|任务|方法|种子|H|总成本|物理+约束|成功/24|约束|求解失败/步|均值H|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in seeds:
        lines.append('|%s|%s|%d|%d|%.4f|%.4f|%d|%d|%d/%d|%.2f|'%(s['task'],s['family'],s['seed'],s['h'],
            s['total_cost'],s['physical_constraint_cost'],s['success'],s['constraints'],s['solver_failure_steps'],s['steps'],s['mean_horizon']))
    lines+=['','验证效果门槛：'+str(gate['validation_effect_passed'])+'。未计时不得声称加速，门槛未满足不得解封测试。','',
        '成功指车辆到达终点，或倒立摆无越界生存到时限；后者不代表精确调节。区间以场景为配对单位，对三个实际模型条件化，不把训练锚点或同场景的重复计时当作新增独立样本。','',
        '完整逐回合 CSV 保存每个失败与负结果。相同物理失败次数也不能用平均 H 或提前终止的短耗时冒充收益。','']
    (output/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gate=gate['validation_effect_passed'],effects=effects),indent=2))


if __name__=='__main__':report()
