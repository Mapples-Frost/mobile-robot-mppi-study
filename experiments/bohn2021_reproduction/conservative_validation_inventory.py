"""Complete validation tables, including independently added fixed-H seeds.

This is an additional coverage/serialization audit, not a new selector. It never
loads test scenarios or outcomes, or changes an existing effect gate.
"""
import csv
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT, TASKS, BASE
from conservative_iteration_evaluate import jobs, arm_name, grid_model
from conservative_iteration_select import load
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read, digest
from fixed_policy_branches import metrics
from run import write


def export(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert read(OUT/'validation_status.json')['complete']
    selected=read(OUT/'baseline_selection.json')
    conditions=list(jobs())
    for task in TASKS:
        h=selected['nominations'][task]['independent_h']
        if h!=BASE[task]:conditions += [(task,'grid',s,h) for s in (1,2)]
    assert len(conditions)==len(set(conditions))
    actual={}
    for task in TASKS:
        for p in (OUT/'evaluations/validation'/task).glob('*/completed.json'):
            summary=read(p.parent/'summary.json')
            key=(task,summary['family'],summary['seed'],summary['h'])
            assert key not in actual
            actual[key]=p
    assert set(actual)==set(conditions),dict(missing=list(set(conditions)-set(actual)),unexpected=list(set(actual)-set(conditions)))
    seed_rows=[];episode_rows=[];hashes={};budget_steps=0;budget_resets=0
    for task,family,seed,h in conditions:
        result=load(task,family,seed,h);assert result
        folder=Path(result['path']);summary=read(folder/'summary.json')
        for cid,episode in enumerate(result['episodes']):
            trace=read(folder/('trace_%02d.json'%cid))
            recalculated=metrics(task,trace)
            assert all(episode[k]==v for k,v in recalculated.items())
            assert episode['physical_constraint_cost']==episode['performance_cost']+episode['constraint_cost']
            assert episode['unique_horizons']==len(set(r['horizon'] for r in trace))
            assert episode['horizon_switches']==sum(a['horizon']!=b['horizon'] for a,b in zip(trace,trace[1:]))
            episode_rows.append(dict(task=task,family=family,seed=seed,h=h,**episode))
        source=grid_model(task,h,seed) if family=='grid' else model_dir(task,'fixed',seed)
        manifest=read(source/'manifest.json');done=read(source/'completed.json')
        assert manifest['seed']==seed and done['steps']==15000 and done['status']=='complete'
        rows=result['episodes'];steps=sum(e['steps'] for e in rows)
        seed_rows.append(dict(task=task,family=family,seed=seed,fixed_h=h,
            mean_total_cost=float(np.mean([e['total_cost'] for e in rows])),
            mean_physical_cost=float(np.mean([e['performance_cost'] for e in rows])),
            mean_h_penalty=float(np.mean([e['h_penalty'] for e in rows])),
            mean_constraint_cost=float(np.mean([e['constraint_cost'] for e in rows])),
            successes=sum(e['success'] for e in rows),episodes=len(rows),
            constraint_episodes=sum(e['constraint'] for e in rows),
            solver_failure_steps=sum(e['solver_failure_steps'] for e in rows),steps=steps,
            solver_failure_rate=sum(e['solver_failure_steps'] for e in rows)/steps,
            mean_h=float(np.average([e['mean_horizon'] for e in rows],weights=[e['steps'] for e in rows])),
            horizon_switches=sum(e['horizon_switches'] for e in rows),
            source_model=str(source),source_training_steps=done['steps'],source_updates=done['updates'],
            source_model_hash=digest(source/'model.zip')))
        for p in [folder/'completed.json',source/'manifest.json',source/'completed.json']:
            hashes[str(p)]=digest(p)
        budget_steps+=summary['explicit_step_calls'];budget_resets+=summary['explicit_reset_calls']
    dest=OUT/'validation_inventory';dest.mkdir(exist_ok=True)
    export(dest/'all_conditions.csv',seed_rows);export(dest/'all_episodes.csv',episode_rows)
    result=dict(passed=True,conditions=len(conditions),episodes=len(episode_rows),
        added_fixed_conditions=len(conditions)-len(jobs()),
        validation_explicit_step_calls=budget_steps,validation_explicit_reset_calls=budget_resets,
        total_recorded_steps=sum(r['steps'] for r in seed_rows),
        hashes=hashes,source_hash=digest(Path(__file__)),
        scope='Complete validation outcome table including extra independently trained fixed seeds. Metrics recomputed from saved traces; no new effect selection, no independent test access.',
        limitations='Inherited source training steps are per model, repeated across matched-terminal policies; do not sum that column as unique training budget. Use dedicated budget ledger. This audit does not replay the solver or replace the independent scenario/physics audit.')
    write(dest/'coverage_audit.json',result)
    lines=['# 保守策略迭代：验证完整清单','',
           '所有固定候选、自适应轮次、训练种子及新补训固定种子均列出；未读取测试。H只是代理，不能由此声称实测加速。','',
           '|任务|方法|种子|固定H|总成本均值|成功/24|约束|求解失败/步|平均H|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        lines.append('|%s|%s|%d|%d|%.5f|%d|%d|%d/%d|%.2f|'%(r['task'],r['family'],r['seed'],r['fixed_h'],r['mean_total_cost'],r['successes'],r['constraint_episodes'],r['solver_failure_steps'],r['steps'],r['mean_h']))
    lines += ['', '完整CSV另含物理成本、约束成本、H罚项、求解失败率、切换次数及来源模型。不同matched H共享终端权重，不可把重复显示的源训练步数重复累计。',
              '此清单不改变冻结的验证选择或效果门槛，也不因覆盖审计通过而宣称控制收益。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))


if __name__=='__main__':main()
