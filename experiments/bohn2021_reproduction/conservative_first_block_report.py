"""Independent first-block trace/dynamics audit and conditional interpretation."""
import csv
import json
from pathlib import Path
import numpy as np
from conservative_first_block_probe import OUT,PREVIOUS,RECOVERY,BASE,without_recovery
from branch_calibration_audit import audit_trace
from conservative_iteration_audit import audit_context
from conservative_fixed_log_audit import integrate
from fixed_policy_branches import metrics
from paper_h_soft_probe import read,digest
from run import write


def main():
    assert read(OUT/'status.json')['complete'];results=read(OUT/'results.json')['results'];assert len(results)==12
    rows=[];hashes={};steps=0;resets=0;solves=0;retries=0;maxerr=0.;integrations=0
    for r in results:
        task,seed,family,cid=[r[k] for k in ('task','seed','family','case')]
        folder=OUT/('%s_s%d_%s'%(task,seed,family));done=read(folder/'completed.json');assert done['passed'] and done['result']==r
        for p,h in done['hashes'].items():assert digest(Path(p))==h
        bank=read(PREVIOUS/'banks'/(task+'_validation.json'))['cases'];traces={}
        for mode in ('fixed_replay','one_block'):
            tr=read(folder/(mode+'.json'));traces[mode]=tr
            audit_trace(task,bank[cid],tr);audit_context(task,bank[cid],tr,0)
            assert all(r['branches'][mode][k]==v for k,v in metrics(task,tr).items())
            for t,row in enumerate(tr):
                expected_h=r['first_adaptive_h'] if mode=='one_block' and r['anchor']<=t<r['anchor']+5 else BASE[task]
                assert row['horizon']==expected_h
                expected=integrate(task,row['previous_state'],row['input']);integrations+=1
                for k,v in expected.items():
                    np.testing.assert_allclose(row['state'][k],v,rtol=1e-7,atol=1e-7);maxerr=max(maxerr,abs(row['state'][k]-v))
        assert without_recovery(traces['one_block'][:r['anchor']])==without_recovery(traces['fixed_replay'][:r['anchor']])
        excess=r['branches']['one_block']['total_cost']-r['branches']['fixed_replay']['total_cost']
        assert excess==r['one_block_excess']
        attempts=[read(p) for p in folder.glob('attempt_*.json')];s=sum(a['step_calls'] for a in attempts);z=sum(a['reset_calls'] for a in attempts)
        assert s==sum(len(t) for t in traces.values()) and z==2
        counts=read(folder/'solver_attempts.json');assert counts['solve_completed']==counts['solve_attempts']==s+z+counts['retry_attempts']
        calls=[json.loads(l) for l in (folder/'solver_calls.jsonl').read_text().splitlines()];assert len(calls)==counts['solve_completed']
        steps+=s;resets+=z;solves+=counts['solve_attempts'];retries+=counts['retry_attempts']
        rows.append(dict(task=task,seed=seed,family=family,case=cid,anchor=r['anchor'],first_h=r['first_adaptive_h'],
            one_block_excess=excess,full_policy_excess=r['full_policy_excess'],
            fixed_success=r['branches']['fixed_replay']['success'],one_block_success=r['branches']['one_block']['success'],full_policy_success=r['full_policy_metrics']['success'],
            first_block_harmful=excess>1e-8,full_policy_worse_than_one_block=r['full_policy_excess']>excess+1e-8))
        hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    with (OUT/'all_conditions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    evidence=dict(passed=True,rows=rows,hashes=hashes,source_hash=digest(Path(__file__)),
        budget=dict(steps=steps,resets=resets,constructors=12,raw_solve_attempts=solves,retries=retries,extra_audit_integrations=integrations),
        maximum_independent_state_error=maxerr,first_block_harmful_cases=sum(r['first_block_harmful'] for r in rows),
        full_policy_worse_than_one_block_cases=sum(r['full_policy_worse_than_one_block'] for r in rows),
        limitations='Worst-case selection was post-hoc in each trained condition. Overlapping first-block and full-policy effects are not additive, and 12 cases do not estimate population fractions. No training, new validation or test evidence.')
    write(OUT/'audit_report.json',evidence)
    lines=['# 首次H干预与反复自适应：条件机制对照','',
        '按每个任务/种子/轮次最大配对超额成本选取12个已暴露场景，精确重放共享前缀；仅保留第一段5步自适应H后恢复固定H，与全程固定及原连续策略比较。相同终端、恢复机制、物理条件和成本。固定重放精确一致，所有24条轨迹独立审计通过。','',
        '|任务|种子|轮次|场景|首次分歧步|首个H|单块超额成本|连续策略超额成本|固定/单块/连续成功|',
        '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append('|%s|%d|%s|%d|%d|%d|%.6f|%.6f|%s/%s/%s|'%(r['task'],r['seed'],r['family'],r['case'],r['anchor'],r['first_h'],r['one_block_excess'],r['full_policy_excess'],r['fixed_success'],r['one_block_success'],r['full_policy_success']))
    lines+=['','7/12所选案例的首块本身产生正成本差；另外5例首块略有收益但原连续策略总体有损失。10/12案例连续策略成本高于单块后恢复固定H；两例第一块已重现全部代价损失。上述比例仅描述事后所选案例，不能外推总体，也不能将重叠差额相加为可实现收益。',
        '车辆seed1第0轮最差场景，首块H50的超额成本仅0.1247，而连续策略为12,388.9170；无求解失败。因此仅修复初值或首块选择不足以解释/修复该失败。车辆seed0第1轮首块H20与完整策略产生相同16.3870超额成本，则直接展示了局部选择泛化错误。',
        '倒立摆seed0第0轮首次H5已导致459.3104超额成本并使原本成功场景失败；seed2第1轮首次H5有193.9610损失，连续策略进一步增至809.8006。这些结果同时指向局部预测错误与连续策略续控不匹配，不能只归咎求解器。',
        '新增预算：%d次环境步、%d次reset、12次构造、%d次原始NLP调用（其中%d次额外重试）；另%d次独立单步积分审计，不作为训练经验。最大独立状态差%.3g。没有训练或测试访问。'%(steps,resets,solves,retries,integrations,maxerr),
        '下一方法应依据这些证据改善收益预测的泛化和连续部署的分布覆盖，保留合理固定H对照；本诊断不许可重用当前暴露验证作为新方法的最终独立证据。','']
    assert evidence['first_block_harmful_cases']==7 and evidence['full_policy_worse_than_one_block_cases']==10
    (OUT/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('hashes','rows')},indent=2))


if __name__=='__main__':main()
