"""Descriptive, all-round comparisons after complete validation; never opens test.

This supplementary report preserves rejected rounds. It does not select a model,
alter the frozen effect gate, or treat validation confidence intervals as final
confirmation. Optional timing comes exclusively from the separate all-candidate
serial replay experiment.
"""
import argparse
from pathlib import Path
import numpy as np
from conservative_iteration import OUT, TASKS, BASE
from conservative_iteration_evaluate import arm_name
from conservative_iteration_select import load, safe
from conservative_iteration_report import bootstrap, time_ratio, export_csv
from paper_h_soft_probe import read, digest
from run import write


def main(include_timing=False):
    assert read(OUT/'validation_status.json')['complete']
    selection_path=OUT/'validation_selection.json'
    selection=read(selection_path)
    inventory=OUT/'validation_inventory/coverage_audit.json'
    assert read(inventory)['passed']
    hashes={str(p):digest(p) for p in (selection_path,inventory)}
    for p,value in selection['hashes'].items():
        assert digest(Path(p))==value
    timing_root=OUT/'timing_validation_all_candidates'
    if include_timing:
        p=timing_root/'completed.json'
        completion=read(p);assert completion['passed']
        for name,value in completion['hashes'].items():
            assert digest(Path(name))==value
        hashes[str(p)]=digest(p)
    timing_cache={}

    def measured(task,family,seed,h):
        if not include_timing:return None
        key=(task,family,seed,h)
        if key not in timing_cache:
            repetitions=[]
            for repeat in range(2):
                folder=timing_root/('r%d_%s_%s'%(repeat,task,arm_name(family,seed,h)))
                completion=read(folder/'completed.json')
                assert completion['passed'] and completion['exact_replay']
                for name,value in completion['hashes'].items():
                    assert digest(Path(name))==value
                rows=read(folder/'summary.json')['episodes']
                assert [e['case'] for e in rows]==list(range(24))
                repetitions.append(rows)
                p=folder/'completed.json';hashes[str(p)]=digest(p)
            timing_cache[key]=repetitions
        return timing_cache[key]

    comparisons=[];pairs=[]
    for task in TASKS:
        nomination=selection['selections'][task]
        for round_id in range(2):
            family='round%d'%round_id
            for comparator in ('independent','matched'):
                h=nomination[comparator+'_h']
                baseline_family=('grid' if comparator=='independent' else 'matched') if h!=BASE[task] else 'primary'
                for seed in range(3):
                    adaptive=load(task,family,seed,0);fixed=load(task,baseline_family,seed,h)
                    assert adaptive and fixed
                    for result in (adaptive,fixed):
                        p=Path(result['path'])/'completed.json';hashes[str(p)]=digest(p)
                    differences=[]
                    for a,b in zip(adaptive['episodes'],fixed['episodes']):
                        assert a['case']==b['case']
                        differences.append(a['total_cost']-b['total_cost'])
                        pairs.append(dict(task=task,round=round_id,comparator=comparator,seed=seed,case=a['case'],
                            adaptive_total_cost=a['total_cost'],fixed_total_cost=b['total_cost'],
                            total_cost_difference=a['total_cost']-b['total_cost'],
                            adaptive_physical_constraint_cost=a['physical_constraint_cost'],fixed_physical_constraint_cost=b['physical_constraint_cost'],
                            adaptive_success=a['success'],fixed_success=b['success'],
                            adaptive_constraint=a['constraint'],fixed_constraint=b['constraint'],
                            adaptive_solver_failure_steps=a['solver_failure_steps'],fixed_solver_failure_steps=b['solver_failure_steps'],
                            adaptive_steps=a['steps'],fixed_steps=b['steps']))
                    ci=bootstrap(differences)
                    ta=measured(task,family,seed,0);tb=measured(task,baseline_family,seed,h)
                    ratio=time_ratio(ta,tb) if include_timing else None
                    row=dict(task=task,round=round_id,comparator=comparator,seed=seed,fixed_h=h,
                        selected_round=nomination['round']==round_id,
                        adaptive_total_cost=adaptive['total_cost'],fixed_total_cost=fixed['total_cost'],
                        cost_difference=ci['mean'],cost_difference_lower=ci['lower'],cost_difference_upper=ci['upper'],
                        relative_cost_difference=ci['mean']/max(abs(fixed['total_cost']),1e-12),
                        adaptive_physical_constraint_cost=adaptive['physical_constraint_cost'],fixed_physical_constraint_cost=fixed['physical_constraint_cost'],
                        adaptive_successes=adaptive['success'],fixed_successes=fixed['success'],
                        adaptive_constraint_episodes=adaptive['constraints'],fixed_constraint_episodes=fixed['constraints'],
                        adaptive_solver_failure_steps=adaptive['failed_steps'],fixed_solver_failure_steps=fixed['failed_steps'],
                        adaptive_solver_failure_rate=adaptive['failure_rate'],fixed_solver_failure_rate=fixed['failure_rate'],
                        adaptive_steps=adaptive['steps'],fixed_steps=fixed['steps'],safe=safe(adaptive,fixed),
                        physical_noninferior=adaptive['physical_constraint_cost']<=fixed['physical_constraint_cost']+.02*abs(fixed['physical_constraint_cost']),
                        episodes_with_horizon_switches=sum(e['horizon_switches']>0 for e in adaptive['episodes']),
                        timing_ratio=ratio['mean_ratio'] if ratio else None,
                        timing_ratio_lower=ratio['lower'] if ratio else None,timing_ratio_upper=ratio['upper'] if ratio else None,
                        timing_repeat0_ratio=ratio['repeat_ratios'][0] if ratio else None,timing_repeat1_ratio=ratio['repeat_ratios'][1] if ratio else None)
                    if include_timing:
                        for label,repeats in (('adaptive',ta),('fixed',tb)):
                            row[label+'_decision_ms']=1000*sum(e['decision_total_s'] for rr in repeats for e in rr)/sum(e['steps'] for rr in repeats for e in rr)
                            row[label+'_deadline_exceed_rate']=sum(e['deadline_exceed_steps'] for rr in repeats for e in rr)/sum(e['steps'] for rr in repeats for e in rr)
                            row[label+'_mean_episode_decision_s']=np.mean([e['decision_total_s'] for rr in repeats for e in rr]).item()
                            row[label+'_mean_reset_s']=np.mean([e['reset']['reset_s'] for rr in repeats for e in rr]).item()
                    comparisons.append(row)
    assert len(comparisons)==24 and len(pairs)==24*24
    dest=OUT/('all_candidate_delivery_timed' if include_timing else 'all_candidate_delivery');dest.mkdir(exist_ok=True)
    export_csv(dest/'all_comparisons.csv',comparisons);export_csv(dest/'all_paired_episodes.csv',pairs)
    report=dict(comparisons=comparisons,timing_available=include_timing,hashes=hashes,source_hash=digest(Path(__file__)),
        inference_scope='Descriptive validation comparisons, all rounds and all seeds; scene-paired intervals conditional on each trained model. Multiple comparisons are not adjusted and are not independent success claims.',
        selection_eligible=selection['selection_eligible'],independent_test_unlocked=False,
        limitations=['Both comparator labels may refer to the same model; those rows are not additional evidence.',
            'A relative-safe branch or aggregate safety gate does not prove per-scenario safety.',
            'Episode duration and termination can differ. Report solver failure rates and decision latency per step alongside episode totals.',
            'Timing repetitions share scenarios and trained models, and do not increase the number of independent scenes or seeds.',
            'Validation selection creates selection bias; final causal efficacy claims require the registered sealed evaluation.',
            'No gate is relaxed and rejected rounds remain rejected.'])
    write(dest/'report.json',report)
    lines=['# 全轮次、全种子的验证对照','',
        '此表是完整验证的描述性补充，保留未入选轮次；不改写冻结的选择或效果门槛，不读取测试。各比较器可能引用同一模型，不构成重复独立证据。','',
        '|任务|轮|比较器H|种子|成本差%|成功：自适应/固定|约束：自适应/固定|安全门槛|实测步耗时比|',
        '|---|---:|---|---:|---:|---|---|---|---|']
    for r in comparisons:
        lines.append('|%s|%d|%s H%d|%d|%.3f|%d/%d|%d/%d|%s|%s|'%(r['task'],r['round'],r['comparator'],r['fixed_h'],r['seed'],100*r['relative_cost_difference'],
            r['adaptive_successes'],r['fixed_successes'],r['adaptive_constraint_episodes'],r['fixed_constraint_episodes'],r['safe'],
            '%.3f [%.3f, %.3f]'%(r['timing_ratio'],r['timing_ratio_lower'],r['timing_ratio_upper']) if include_timing else '尚未实测'))
    lines+=['','成本差以自适应减固定计算，负数为低成本；安全门槛同时比较成功数、约束回合数与求解失败率。物理成本非劣和实际回合内切换另列CSV，不能单凭本表安全列宣称通过全部准入。',
        '所有区间采用24个场景配对重采样，对实际训练模型条件化；不把三种子的同场景或两次计时当作新增独立场景。多重比较未校正，不用单个显著结果宣称成功。',
        '计时含策略提取与控制器调用，排除物理模拟、审计和写盘；reset热身另列。每步耗时、回合总耗时、截止期超时率并列，避免提前失败的短回合被当作计算优势。',
        '完整原始轨迹、逐回合配对、逐种子比较和文件散列可追溯。此报告没有独立测试证据。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(dict(comparisons=len(comparisons),paired_episodes=len(pairs),timing=include_timing,output=str(dest)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--timing',action='store_true');a=ap.parse_args();main(a.timing)
