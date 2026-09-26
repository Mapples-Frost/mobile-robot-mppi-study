"""All conditions, seeds, failures and scoped budgets for the diagnostic probe."""
import csv
import json
from pathlib import Path
from failure_state_protocol import ROOT, OLD, OUT, REG, read, write, sha, verify, jobs


def aggregate(episodes):
    assert len(episodes)==24 and [e['case'] for e in episodes]==list(range(24))
    result={k:sum(e[v] for e in episodes) for k,v in (
        ('steps','steps'),('success','success'),('constraints','constraint'),
        ('initial_failures','initial_failed_steps'),('final_failures','solver_failure_steps'),
        ('retries','retries'),('switches','switches'))}
    for k in ('total_cost','physical_constraint_cost','performance_cost','h_penalty','constraint_cost'):
        result[k]=sum(e[k] for e in episodes)/24
    result['initial_failure_rate']=result['initial_failures']/result['steps']
    result['final_failure_rate']=result['final_failures']/result['steps']
    return result


def budget():
    # Include partial and failed attempts, even if no completed summary exists.
    meters=sorted(OUT.glob('*/**/attempt_*.json'))
    solvers=sorted(OUT.glob('*/**/solver_attempts.json'))
    assert len(meters)==len(set(meters)) and len(solvers)==len(set(solvers))
    groups={};rows=[]
    for p in meters:
        a=read(p);phase=p.relative_to(OUT).parts[0]
        d=groups.setdefault(phase,dict(step_attempts=0,reset_attempts=0,solve_attempts=0,solve_completed=0,warmups=0,retries=0))
        d['step_attempts']+=a['step_calls'];d['reset_attempts']+=a['reset_calls']
        rows.append(dict(path=str(p),kind='environment',sha256=sha(p),steps=a['step_calls'],resets=a['reset_calls'],solver_attempts=0,solver_completed=0))
    for p in solvers:
        a=read(p);phase=p.relative_to(OUT).parts[0]
        d=groups.setdefault(phase,dict(step_attempts=0,reset_attempts=0,solve_attempts=0,solve_completed=0,warmups=0,retries=0))
        for key,source in (('solve_attempts','solve_attempts'),('solve_completed','solve_completed'),('warmups','warmup_attempts'),('retries','retry_attempts')):
            d[key]+=a[source]
        rows.append(dict(path=str(p),kind='solver',sha256=sha(p),steps=0,resets=0,solver_attempts=a['solve_attempts'],solver_completed=a['solve_completed']))
    return dict(groups=groups,total={k:sum(g[k] for g in groups.values()) for k in ('step_attempts','reset_attempts','solve_attempts','solve_completed','warmups','retries')},
                counter_files=len(rows),counter_rows=rows,
                scope='Only explicit attempted calls in this new diagnostic directory, including partial attempts. Constructors before wrapper installation, inherited model training and reused old trajectories are excluded, not declared zero. Offline integrations reported separately.')


def main():
    verify();audit=read(OUT/'audit_full.json');assert audit['passed']
    hashes={str(REG):sha(REG),str(OUT/'audit_full.json'):sha(OUT/'audit_full.json')}
    for p,h in audit['hashes'].items():assert sha(Path(p))==h
    rows=[];eps=[];lookup={}
    for task,seed,policy in jobs():
        folder=OUT/'full'/('%s_s%d_%s'%(task,seed,policy));s=read(folder/'summary.json')
        done=read(folder/'completed.json')
        for p,h in done['hashes'].items():assert sha(Path(p))==h
        hashes[str(folder/'completed.json')]=sha(folder/'completed.json')
        agg=aggregate(s['episodes']);lookup[(task,seed,policy)]=(agg,s['episodes'])
        ag=next(g for g in audit['groups'] if (g['task'],g['seed'],g['policy'])==(task,seed,policy))
        rows.append(dict(task=task,seed=seed,policy=policy,reused=s['reused'],intervention_steps=ag['intervention_steps'],**agg))
        for e in s['episodes']:eps.append(dict(task=task,seed=seed,policy=policy,**{k:v for k,v in e.items() if k!='intervention_steps'}))
    comparisons=[];paired=[]
    for task,seed,policy in jobs():
        base='fixed30' if task=='pendulum' else 'fixed25'
        if policy==base:continue
        a,ae=lookup[(task,seed,policy)];b,be=lookup[(task,seed,base)]
        safe=a['success']>=b['success'] and a['constraints']<=b['constraints'] and a['initial_failure_rate']<=b['initial_failure_rate'] and a['final_failure_rate']<=b['final_failure_rate']
        physical=a['physical_constraint_cost']<=b['physical_constraint_cost']+.02*abs(b['physical_constraint_cost'])
        inter=next(r['intervention_steps'] for r in rows if (r['task'],r['seed'],r['policy'])==(task,seed,policy))
        comparisons.append(dict(task=task,seed=seed,policy=policy,reference=base,safe=safe,physical_noninferior=physical,
                    intervened=inter>0,total_change_percent=100*(a['total_cost']-b['total_cost'])/abs(b['total_cost']),
                    physical_change_percent=100*(a['physical_constraint_cost']-b['physical_constraint_cost'])/abs(b['physical_constraint_cost']),
                    success_delta=a['success']-b['success'],constraints_delta=a['constraints']-b['constraints'],
                    initial_failure_rate_delta=a['initial_failure_rate']-b['initial_failure_rate'],
                    final_failure_rate_delta=a['final_failure_rate']-b['final_failure_rate'],steps_delta=a['steps']-b['steps']))
        for x,y in zip(ae,be):
            paired.append(dict(task=task,seed=seed,policy=policy,reference=base,case=x['case'],
                               total_delta=x['total_cost']-y['total_cost'],physical_delta=x['physical_constraint_cost']-y['physical_constraint_cost'],
                               success_delta=int(x['success'])-int(y['success']),constraints_delta=int(x['constraint'])-int(y['constraint']),
                               initial_failures_delta=x['initial_failed_steps']-y['initial_failed_steps'],
                               final_failures_delta=x['solver_failure_steps']-y['solver_failure_steps']))
    assert len(rows)==24 and len(eps)==576 and len(comparisons)==18 and len(paired)==432
    viable=[]
    for policy in ('certificate5','certificate10','certificate15'):
        selected=[r for r in comparisons if r['task']=='pendulum' and r['policy']==policy]
        assert len(selected)==3
        if all(r['safe'] and r['physical_noninferior'] and r['intervened'] for r in selected):viable.append(policy)
    ledger=budget()
    smoke=read(OUT/'audit_smoke.json')
    failed=dict(step_attempts=0,reset_attempts=0,solve_attempts=0,solve_completed=0)
    receipt=OUT/'serialization_repair/failure_receipt.json'
    if receipt.exists():
        evidence=read(receipt);failed=evidence['budget'];hashes[str(receipt)]=sha(receipt)
        for key,value in failed.items():assert ledger['groups']['serialization_repair'][key]==value
    assert ledger['total']['step_attempts']==smoke['new_control_steps']+audit['new_control_steps']+failed['step_attempts']
    assert ledger['total']['reset_attempts']==smoke['new_resets']+audit['new_resets']+failed['reset_attempts']
    assert ledger['total']['solve_attempts']==smoke['new_solver_calls']+audit['new_solver_calls']+failed['solve_attempts']
    report=dict(passed=True,post_hoc=True,diagnostic_only=True,new_training_steps=0,conditions=24,episodes=576,
                comparisons=comparisons,timing_screen_candidates=viable,timing_scheduled=False,
                serial_timing_required_before_speed_claim=True,validation_or_test=False,goal_complete=False,
                budget=ledger,failed_recording_attempts=failed,new_numerical_integrations=smoke['new_numerical_integrations']+audit['new_numerical_integrations'],
                hashes=hashes,caveats='All outcomes are from reused, exposed training cases. No confidence interval, generalization or learned-adaptive efficacy claim. Timing screen only schedules further diagnostic measurement, not confirmation.')
    dest=OUT/'delivery'
    if (dest/'report.json').exists():
        old=read(dest/'report.json');assert {k:v for k,v in old.items() if k!='delivery_hashes'}==report
        for p,h in old['delivery_hashes'].items():assert sha(dest/p)==h
        print('Existing diagnostic delivery verified');return
    assert not dest.exists(),'Inspect partial delivery';dest.mkdir()
    for name,data in (('all_conditions.csv',rows),('all_episodes.csv',eps),('all_comparisons.csv',comparisons),('all_paired_cases.csv',paired),('all_budget_counters.csv',ledger['counter_rows'])):
        with (dest/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 失败状态时域诊断：全部训练案例与种子','','仅为已暴露训练案例上的事后机制干预；不是新验证、独立测试或学习策略效果。原成本、动力学、终端及重试不变。',
           '','|任务|种子|策略|总成本变化%|物理成本变化%|成功变化|约束变化|安全不退化|物理不劣2%|',
           '|---|---:|---|---:|---:|---:|---:|---|---|']
    for r in comparisons:
        lines.append('|%s|%d|%s|%.6f|%.6f|%+d|%+d|%s|%s|'%(r['task'],r['seed'],r['policy'],r['total_change_percent'],r['physical_change_percent'],r['success_delta'],r['constraints_delta'],r['safe'],r['physical_noninferior']))
    lines+=['','参照为车辆H25、倒立摆H30，全部三种子终端各自冻结。固定H30/H35车辆不是自适应策略，不能仅凭其改善宣称自适应有效。',
            '', '符合继续诊断计时前提的倒立摆候选：'+(', '.join(viable) if viable else '无')+'。该前提只决定是否测量，不是最终验收；正式串行计时尚未执行，不从H或并行求解时长声称加速。',
            '', '全部24条件、576回合、18项配对汇总与432场景配对均保留。复用216个已有回合，新增正式360回合；额外冒烟72回合。',
            '', '新显式调用预算：'+json.dumps(ledger['total'],ensure_ascii=False)+'。独立积分次数：'+str(report['new_numerical_integrations'])+'。模型无新训练；原构造器未包装调用、继承训练和旧轨迹的预算留在其原研究，不重复入账。',
            '', '首次记录接口失败的尝试也包含在总预算中：'+json.dumps(failed,ensure_ascii=False)+'。原求解日志/计数和24个失败日志保留，但该次状态/动作轨迹未写出，不能声称其已独立积分审计；后续重放开销另计。',
            '', '任何后续正式方法需新训练、验证与封存测试及充分固定H比较；本报告不解封原测试。']
    (dest/'report_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report['delivery_hashes']={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    write(dest/'report.json',report)
    print(json.dumps(dict(passed=True,conditions=24,episodes=576,timing_screen_candidates=viable,budget=ledger['total']),indent=2))


if __name__=='__main__':main()
