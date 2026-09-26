"""Complete recovery-diagnosis results and attempted budget; never test access."""
import csv
import json
from pathlib import Path
import numpy as np
from conservative_recovery_diagnosis import OUT,TASKS,BASE,register
from conservative_iteration_evaluate import arm_name
from paper_h_soft_probe import read,digest
from run import write


def main():
    register();audit=read(OUT/'audit_formal/audit.json');assert audit['passed']
    for p,h in audit['hashes'].items():assert digest(Path(p))==h
    summaries={};rows=[];pairs=[]
    for task in TASKS:
        for seed in range(3):
            for family in ('primary','round0','round1'):
                h=BASE[task] if family=='primary' else 0
                d=read(OUT/'formal'/task/arm_name(family,seed,h)/'summary.json');summaries[task,seed,family]=d
    for (task,seed,family),d in summaries.items():
        episodes=d['episodes'];baseline=summaries[task,seed,'primary']['episodes']
        old=[e['original_metrics'] for e in episodes]
        cost=float(np.mean([e['total_cost'] for e in episodes]));bcost=float(np.mean([e['total_cost'] for e in baseline]))
        row=dict(task=task,seed=seed,family=family,mean_cost=cost,original_mean_cost=float(np.mean([e['total_cost'] for e in old])),
            fixed_recovery_cost=bcost,percent_vs_fixed=100*(cost-bcost)/abs(bcost),
            success=sum(e['success'] for e in episodes),original_success=sum(e['success'] for e in old),
            constraints=sum(e['constraint'] for e in episodes),original_constraints=sum(e['constraint'] for e in old),
            initial_solver_failed_steps=sum(e['initial_solver_failed_steps'] for e in episodes),
            recovered_steps=sum(e['recovered_steps'] for e in episodes),final_failed_steps=sum(e['solver_failure_steps'] for e in episodes),
            original_failed_steps=sum(e['solver_failure_steps'] for e in old),retries=sum(e['retry_attempts'] for e in episodes),
            steps=sum(e['steps'] for e in episodes),physical_constraint_cost=float(np.mean([e['physical_constraint_cost'] for e in episodes])))
        rows.append(row)
        if family!='primary':
            for a,b in zip(episodes,baseline):
                assert a['case']==b['case']
                pairs.append(dict(task=task,seed=seed,family=family,case=a['case'],adaptive_cost=a['total_cost'],fixed_cost=b['total_cost'],
                    difference=a['total_cost']-b['total_cost'],adaptive_success=a['success'],fixed_success=b['success'],
                    adaptive_initial_solver_failures=a['initial_solver_failed_steps'],adaptive_final_solver_failures=a['solver_failure_steps'],
                    fixed_initial_solver_failures=b['initial_solver_failed_steps'],fixed_final_solver_failures=b['solver_failure_steps']))
    ledger=[]
    for p in sorted(OUT.rglob('attempt_*.json')):
        d=read(p);phase=p.relative_to(OUT).parts[0]
        ledger.append(dict(path=str(p),phase=phase,steps=d['step_calls'],resets=d['reset_calls'],constructors=1,hash=digest(p)))
    solver_ledger=[]
    for p in sorted(OUT.rglob('solver_attempts.json')):
        d=read(p);solver_ledger.append(dict(path=str(p),phase=p.relative_to(OUT).parts[0],**d,hash=digest(p)))
    assert len(ledger)==len(solver_ledger)==23
    budget=dict(steps=sum(x['steps'] for x in ledger),resets=sum(x['resets'] for x in ledger),constructors=len(ledger),
        raw_solver_attempts=sum(x['solve_attempts'] for x in solver_ledger),completed_raw_solver_calls=sum(x['solve_completed'] for x in solver_ledger),
        retries=sum(x['retry_attempts'] for x in solver_ledger),warmups=sum(x['warmup_attempts'] for x in solver_ledger),
        scope='Formal, repeated smoke and interrupted smoke included. Each explicit step normally invokes one initial solver call; failed final smoke call may abort before plant propagation. Constructor-internal solves uninstrumented and not guessed.')
    assert budget['raw_solver_attempts']==budget['completed_raw_solver_calls']==budget['steps']+budget['resets']+budget['retries']
    dest=OUT/'delivery';dest.mkdir(exist_ok=True)
    for filename,rr in [('all_conditions.csv',rows),('all_paired_episodes.csv',pairs)]:
        with (dest/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    result=dict(rows=rows,budget=budget,environment_attempts=ledger,solver_attempts=solver_ledger,
        hashes={str(OUT/'audit_formal/audit.json'):digest(OUT/'audit_formal/audit.json'),str(OUT/'protocol.json'):digest(OUT/'protocol.json')},
        source_hash=digest(Path(__file__)),test_access=False,independent_efficacy_evidence=False,
        conclusion='Recovery removed only the two vehicle seed2 case2 failures; no pendulum recovery. Remaining control losses persist; no cross-task all-seed success.')
    write(dest/'report.json',result)
    lines=['# 同一步求解恢复：完整机制诊断结果','',
        '18条件、432回合全部完成并通过独立审计。使用此前暴露场景，因此是机制诊断，不是新独立验证或测试。固定H与两轮策略获得相同恢复机制。','',
        '|任务|种子|方法|原成本|恢复后成本|相对恢复后固定H变化%|成功/24|原始失败步|最终失败步|恢复步|额外求解|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|%s|%d|%s|%.4f|%.4f|%.3f|%d|%d|%d|%d|%d|'%(r['task'],r['seed'],r['family'],r['original_mean_cost'],r['mean_cost'],r['percent_vs_fixed'],r['success'],r['initial_solver_failed_steps'],r['final_failed_steps'],r['recovered_steps'],r['retries']))
    lines+=['','只有车辆seed2的两个轮次场景2发生恢复，各一次首步重试后到达；第0轮均值从226.7449降至18.8783，第1轮从346.0406降至19.8232，固定H为19.9037。种子0/1和全部倒立摆轨迹保持原结果，不能用修复一个失败覆盖其他条件的损失。',
        '倒立摆所有尝试均未恢复，原始和最终失败相同；因此这个机制在本批倒立摆场景中只增加计算开销。车辆seed1第0轮仍在没有求解失败的情况下未到达，不能把所有损失归因于求解器。',
        '独立审计重新核算回合成本、约束、策略动作、日志顺序、重试上限、失败保留与动力学；31,063条正式转移最大独立状态差2.87e-8。无成功恢复回合与旧轨迹逐项相同。重复冒烟另284转移，类型修复中断9次环境步尝试单列。',
        '总预算（含冒烟与中断）：%d次环境步尝试、%d次reset、%d次环境构造、%d次已完成原始NLP调用，其中%d次额外重试、%d次显式reset热身。构造内部调用数量不猜测；这些数值不能与父研究已计入的同一路径重复相加。'%(budget['steps'],budget['resets'],budget['constructors'],budget['completed_raw_solver_calls'],budget['retries'],budget['warmups']),
        '本轮未测公平串行时延，不宣称加速；保存的局部solver时间仅供诊断。没有修改旧验证结论、删场景或解封测试。下一步须针对剩余策略失效开展分析，不能仅把恢复包装为复现成功。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(conditions=len(rows),paired_episodes=len(pairs),budget=budget),indent=2))


if __name__=='__main__':main()
