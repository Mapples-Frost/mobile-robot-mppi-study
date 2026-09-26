"""Independent raw timing accounting, exact replay and full diagnostic delivery."""
import copy
import csv
import json
import math
from pathlib import Path
import numpy as np
from failure_state_protocol import ROOT, OUT, read, write, sha, verify
from failure_state_report import budget


def clean(trace):
    values=copy.deepcopy(trace)
    for row in values:
        row.pop('timing',None);row.pop('gate',None)
        for a in row['recovery']['attempts']:a.pop('solver_s',None)
    return values


def close(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)


def stats(rows,reset_total):
    d=np.asarray([r['timing']['decision_s'] for r in rows])
    g=np.asarray([r['timing']['decision_gross_s'] for r in rows])
    return dict(steps=len(rows),decision_total_s=float(d.sum()),decision_mean_s=float(d.mean()),
                decision_median_s=float(np.median(d)),decision_p95_s=float(np.percentile(d,95)),
                decision_max_s=float(d.max()),deadline_exceed_steps=int(sum(d>.04)),
                gross_total_s=float(g.sum()),gross_mean_s=float(g.mean()),gross_median_s=float(np.median(g)),
                gross_p95_s=float(np.percentile(g,95)),gross_deadline_exceed_steps=int(sum(g>.04)),
                selection_total_s=sum(r['timing']['selection_s'] for r in rows),
                logging_total_s=sum(r['timing']['logging_s'] for r in rows),reset_gross_total_s=reset_total,
                initial_failed_steps=sum(not r['recovery']['attempts'][0]['success'] for r in rows),
                final_failed_steps=sum(not r['recovery']['final_success'] for r in rows),
                retries=sum(len(r['recovery']['attempts'])-1 for r in rows))


def main():
    verify();state=read(OUT/'timing_status.json');assert state['complete'] and not state['active']
    proc=Path('/proc')/str(state['pid'])/'cmdline'
    if proc.exists():
        # Timing ran in the waiting controller itself. That controller may now
        # wait on this audit subprocess; its explicit terminal phase is checked.
        finish=read(OUT/'finish_status.json')
        assert finish['pid']==state['pid'] and finish['stage']=='independent_timing_audit'
        assert b'failure_state_finish.py' in proc.read_bytes()
    registration=read(OUT/'timing_registration.json')
    hashes=dict(registration['source_hashes'])
    amendment=OUT/'timing_controller_repair/amendment.json'
    if amendment.exists():
        repair=read(amendment)
        assert repair['original_timing_registration_sha256']==sha(OUT/'timing_registration.json')
        for p,h in repair['archived_files'].items():assert sha(ROOT/p)==h
        for p,item in repair['changed_sources'].items():
            assert hashes[p]==item['old_sha256'];hashes[p]=item['new_sha256']
        hashes[str(amendment)]=sha(amendment)
    hashes[str(OUT/'timing_registration.json')]=sha(OUT/'timing_registration.json')
    complete=read(OUT/'timing/completed.json');assert complete['passed'] and complete['conditions']==24
    hashes.update(complete['hashes']);hashes[str(OUT/'timing/completed.json')]=sha(OUT/'timing/completed.json')
    for p,h in hashes.items():assert sha(Path(p))==h
    conditions=[];episodes=[];pooled={};repeat_results={};steps=0;total_solves=0;total_retries=0
    for j in registration['jobs']:
        repeat=j['repeat'];task,seed,policy=j['job']
        folder=OUT/'timing'/('r%d_%s_s%d_%s'%(repeat,task,seed,policy))
        done=read(folder/'completed.json');assert done['passed'] and done['exact_replay']
        for p,h in done['hashes'].items():assert sha(Path(p))==h
        summary=read(folder/'summary.json');assert (task,seed,policy,repeat)==tuple(summary[k] for k in ('task','seed','policy','repeat'))
        assert [e['case'] for e in summary['episodes']]==list(range(24))
        reference=Path(summary['reference'])
        calls=[json.loads(x) for x in (folder/'solver_calls.jsonl').read_text().splitlines()]
        counts=read(folder/'solver_attempts.json');assert summary['solver_counts']==counts
        assert len(calls)==counts['solve_attempts']==counts['solve_completed']
        assert counts['warmup_attempts']==24==sum(c['kind']=='warmup' for c in calls)
        assert counts['retry_attempts']==sum(c['kind'].startswith('retry') for c in calls)
        assert summary['logging_operations']==1+3*len(calls)
        cursor=0;all_rows=[];reset_total=0.;logging_total=0.
        for saved in summary['episodes']:
            cid=saved['case'];rows=read(folder/('r0_trace_%02d.json'%cid));original=read(reference/('r0_trace_%02d.json'%cid))
            assert clean(rows)==clean(original)
            assert saved['steps']==len(rows) and saved['cost']==sum(-r['reward'] for r in rows)
            reset=read(folder/('reset_%02d.json'%cid));assert reset==saved['reset']
            assert reset['reset_gross_s']>=reset['controller_gross_s']>reset['logging_s']>0
            close(reset['controller_s'],reset['controller_gross_s']-reset['logging_s'])
            reset_total+=reset['reset_gross_s'];logging_total+=reset['logging_s']
            assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid and calls[cursor]['step']==-1;cursor+=1
            for step,row in enumerate(rows):
                tm=row['timing'];assert all(math.isfinite(v) and v>0 for v in tm.values())
                close(tm['controller_s'],tm['controller_gross_s']-tm['logging_s'])
                close(tm['decision_s'],tm['controller_s']+tm['selection_s'])
                close(tm['decision_gross_s'],tm['decision_s']+tm['logging_s'])
                attempts=row['recovery']['attempts']
                assert calls[cursor:cursor+len(attempts)]==attempts;cursor+=len(attempts)
                assert all(a['case']==cid and a['step']==step for a in attempts)
                assert tm['controller_s']>=sum(a['solver_s'] for a in attempts)-1e-12
                logging_total+=tm['logging_s']
            a=stats(rows,reset['reset_gross_s'])
            for k in ('decision_total_s','decision_mean_s','decision_median_s','decision_p95_s','gross_total_s','logging_total_s'):
                close(a[k],saved[k])
            assert a['deadline_exceed_steps']==saved['deadline_exceed_steps']
            episodes.append(dict(task=task,seed=seed,policy=policy,repeat=repeat,case=cid,**a))
            all_rows.extend(rows)
        assert cursor==len(calls) and summary['logging_total_s']>logging_total
        meters=[read(p) for p in folder.glob('attempt_*.json')]
        assert sum(m['step_calls'] for m in meters)==len(all_rows) and sum(m['reset_calls'] for m in meters)==24
        a=stats(all_rows,reset_total)
        conditions.append(dict(task=task,seed=seed,policy=policy,repeat=repeat,**a));repeat_results[(seed,policy,repeat)]=a
        item=pooled.setdefault((seed,policy),dict(rows=[],reset_s=0.))
        item['rows'].extend(all_rows);item['reset_s']+=reset_total
        steps+=len(all_rows);total_solves+=len(calls);total_retries+=counts['retry_attempts']
    assert len(conditions)==24 and len(episodes)==576 and len(pooled)==12
    pooled_rows=[dict(task='pendulum',seed=seed,policy=policy,repeats=2,unique_scenes=24,**stats(v['rows'],v['reset_s']))
                 for (seed,policy),v in sorted(pooled.items())]
    comparisons=[]
    for seed in range(3):
        for policy in ('certificate5','certificate10','certificate15'):
            ratios=[repeat_results[(seed,policy,r)]['decision_mean_s']/repeat_results[(seed,'fixed30',r)]['decision_mean_s'] for r in range(2)]
            cost=[x for x in read(OUT/'delivery/report.json')['comparisons'] if (x['task'],x['seed'],x['policy'])==('pendulum',seed,policy)]
            assert len(cost)==1
            comparisons.append(dict(seed=seed,policy=policy,mean_time_change_percent=100*(sum(ratios)/2-1),
                 repeat0_change_percent=100*(ratios[0]-1),repeat1_change_percent=100*(ratios[1]-1),
                 both_repeats_faster=all(r<1 for r in ratios),all_training_safety=cost[0]['safe'],
                 physical_noninferior=cost[0]['physical_noninferior'],cost_change_percent=cost[0]['total_change_percent']))
    ledger=budget();assert ledger['groups']['timing']['step_attempts']==steps
    assert ledger['groups']['timing']['reset_attempts']==576
    assert ledger['groups']['timing']['solve_attempts']==total_solves
    assert ledger['groups']['timing']['retries']==total_retries
    prior=read(OUT/'delivery/report.json')['budget']['total']
    for k in prior:assert ledger['total'][k]==prior[k]+ledger['groups']['timing'][k]
    hashes[str(OUT/'delivery/report.json')]=sha(OUT/'delivery/report.json')
    report=dict(passed=True,diagnostic_only=True,conditions=24,episodes=576,steps=steps,comparisons=comparisons,
                budget=ledger,hashes=hashes,test_accessed=False,core_effect_not_evaluated=True,
                pooling='All raw steps pooled to compute quantiles, not mean of episode quantiles. Two timing repeats are not additional scenes. Mean comparison averages per-repeat per-step ratios.',
                limitations='Exposed training scenarios and post-hoc hand-coded rule. No learned adaptive method, CI or independent effect claim. WSL host scheduling and logged-overhead subtraction remain limitations.')
    dest=OUT/'timing_delivery'
    if (dest/'report.json').exists():
        prior=read(dest/'report.json');assert {k:v for k,v in prior.items() if k!='delivery_hashes'}==report
        for n,h in prior['delivery_hashes'].items():assert sha(dest/n)==h
        print('Existing timing delivery verified');return
    assert not dest.exists(),'Inspect partial delivery';dest.mkdir()
    for name,rows in (('all_conditions.csv',conditions),('all_episodes.csv',episodes),('pooled_arms.csv',pooled_rows),('all_comparisons.csv',comparisons),('all_budget_counters.csv',ledger['counter_rows'])):
        with (dest/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 失败状态诊断：全部实测延迟','','这是已暴露训练场景上的规则诊断，不是学习式自适应的独立效果。所有四策略、三个种子和两次串行随机顺序重复均保留；每个非计时轨迹字段精确重放。',
           '','|种子|策略|平均延迟变化%|重复1%|重复2%|总成本变化%|安全不退化|物理不劣2%|',
           '|---|---|---:|---:|---:|---:|---|---|']
    for c in comparisons:lines.append('|%d|%s|%.6f|%.6f|%.6f|%.6f|%s|%s|'%(c['seed'],c['policy'],c['mean_time_change_percent'],c['repeat0_change_percent'],c['repeat1_change_percent'],c['cost_change_percent'],c['all_training_safety'],c['physical_noninferior']))
    lines+=['','基准为同种子H30。负变化表示更低。净决策包含策略状态读取/选择、控制器与全部重试，扣除实测日志；毛时间、日志、reset、中位数、p95、最大值和严格超过40ms步数均在CSV。物理仿真和外层轨迹写入不在主耗时内。',
            '', '跨重复分位数由全部控制步重新计算；短失败回合不删除。不同轨迹长度和状态难度影响总体时间，不把时域长度当作速度。',
            '', '累计显式调用预算（含原记录接口失败、修复冒烟、正式诊断和本计时）：'+json.dumps(ledger['total'],ensure_ascii=False)+'。复用旧训练/原参考回合及构造器未包装调用仍按原报告另列。',
            '', report['limitations']]
    (dest/'report_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report['delivery_hashes']={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    write(dest/'report.json',report)
    print(json.dumps(dict(passed=True,conditions=24,episodes=576,steps=steps,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
