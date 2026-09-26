"""Training-only mechanical attribution of conservative extraction gates.

No threshold tuning, new policy, validation access or closed-loop efficacy claim.
"""
import csv
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS,HS,BASE,verify
from conservative_policy_model import predict,choose
from paper_h_soft_probe import read,digest
from run import write


def safe(candidate,source):
    return candidate['success']>=source['success'] and candidate['constraint']<=source['constraint'] and candidate['solver_failure_steps']<=source['solver_failure_steps']


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    verify();groups=[];candidates=[];summaries=[];hashes={}
    for task in TASKS:
        for seed in range(3):
            for round_id in range(2):
                folder=OUT/('%s_s%d_r%d'%(task,seed,round_id))
                fit=read(folder/'fit_completed.json');model=read(folder/'policy.json');data=read(folder/'collection_completed.json')
                assert digest(folder/'policy.json')==fit['policy_hash'] and digest(folder/'collection_completed.json')==fit['dataset_hash']
                for name in ('fit_completed.json','policy.json','collection_completed.json'):
                    hashes[str(folder/name)]=digest(folder/name)
                local=[]
                for g in data['groups']:
                    features=g['context']['features'];q,prob,distance=predict(model,features)
                    fallback=choose(model['prior'],features) if model['prior'] else BASE[task]
                    assert fallback==g['source_h']
                    ref=HS.index(fallback);base=HS.index(BASE[task]);source=g['branches'][str(fallback)]
                    costs=np.array([g['branches'][str(h)]['total_cost'] for h in HS])
                    allowed=np.array([safe(g['branches'][str(h)],source) for h in HS])
                    oracle=int(np.argmin(np.where(allowed,costs,np.inf)))
                    chosen=HS.index(choose(model,features))
                    cost_votes=np.sum(q<q[:,ref:ref+1]-.1,axis=0)
                    safety_votes=np.sum(prob>=.8,axis=0)
                    support=distance<=model['support_radius']
                    eligible=[i for i in range(len(HS)) if cost_votes[i]==3 and safety_votes[i]==3]
                    expected=min(eligible,key=lambda i:(float(q[:,i].mean()),HS[i])) if support and eligible else ref
                    assert chosen==expected,'Independent extraction logic disagrees'
                    transformed=np.sign((costs-costs[base])/.1)*np.log1p(np.abs((costs-costs[base])/.1))
                    in_bag=[g['case'] in m['bootstrap_scenes'] for m in fit['members']]
                    cost_blockers=np.flatnonzero(~(q[:,oracle]<q[:,ref]-.1))
                    row=dict(task=task,seed=seed,round=round_id,case=g['case'],anchor=g['anchor'],
                        source_h=fallback,chosen_h=HS[chosen],oracle_h=HS[oracle],source_success=source['success'],
                        source_cost=float(costs[ref]),chosen_cost=float(costs[chosen]),oracle_cost=float(costs[oracle]),
                        selected_label_safe=bool(allowed[chosen]),label_gain=float(costs[ref]-costs[chosen]),
                        oracle_available_gain=float(costs[ref]-costs[oracle]),chosen_regret=float(costs[chosen]-costs[oracle]),
                        support_passed=bool(support),oracle_cost_votes=int(cost_votes[oracle]),
                        oracle_safety_votes=int(safety_votes[oracle]),
                        oracle_cost_blocked_by_out_of_bag=any(not in_bag[i] for i in cost_blockers),
                        oracle_cost_blocked_by_in_bag=any(in_bag[i] for i in cost_blockers),
                        oracle_exact_target_margin_passed=bool(transformed[oracle]<transformed[ref]-.1),
                        in_bag_members=sum(in_bag),distance=float(distance),support_radius=model['support_radius'])
                    groups.append(row);local.append(row)
                    for i,h in enumerate(HS):
                        candidates.append(dict(task=task,seed=seed,round=round_id,case=g['case'],anchor=g['anchor'],h=h,
                            label_cost=float(costs[i]),label_safe=bool(allowed[i]),cost_votes=int(cost_votes[i]),
                            safety_votes=int(safety_votes[i]),support_passed=bool(support),selected=i==chosen))
                improvable=[r for r in local if r['oracle_available_gain']>1e-10]
                missed=[r for r in improvable if r['chosen_regret']>1e-10]
                summaries.append(dict(task=task,seed=seed,round=round_id,anchors=len(local),
                    source_unsuccessful_anchors=sum(not r['source_success'] for r in local),
                    selected_label_unsafe=sum(not r['selected_label_safe'] for r in local),
                    improvable_anchors=len(improvable),missed_oracle_anchors=len(missed),
                    positive_label_regret_sum=sum(max(r['chosen_regret'],0) for r in local),
                    support_blocks=sum(not r['support_passed'] for r in missed),
                    oracle_cost_consensus_blocks=sum(r['oracle_cost_votes']<3 for r in missed),
                    oracle_safety_consensus_blocks=sum(r['oracle_safety_votes']<3 for r in missed),
                    oracle_cost_blocks_with_out_of_bag_member=sum(r['oracle_cost_blocked_by_out_of_bag'] for r in missed),
                    oracle_cost_blocks_with_in_bag_member=sum(r['oracle_cost_blocked_by_in_bag'] for r in missed),
                    exact_target_margin_blocks=sum(not r['oracle_exact_target_margin_passed'] for r in missed)))
    dest=OUT/'training_diagnosis/extraction_gates';dest.mkdir(exist_ok=True)
    table(dest/'all_anchors.csv',groups);table(dest/'all_candidates.csv',candidates);table(dest/'all_conditions.csv',summaries)
    write(dest/'audit.json',dict(passed=True,conditions=len(summaries),anchors=len(groups),candidates=len(candidates),
        summaries=summaries,hashes=hashes,source_hash=digest(Path(__file__)),
        scope='Descriptive training-only gate accounting; independent reimplementation of extraction decision agrees at every sampled anchor. Gate reasons overlap and are not an additive causal decomposition.',
        limitations='Oracle uses realized saved branch futures and is not deployable. Out-of-bag means this member did not bootstrap the scene; normalization and other members may have seen it. Overlapping suffix regret sums are not episode savings or independent samples. A relative-safe label can compare two unsuccessful branches. No outcomes after changing any gate were simulated.'))
    lines=['# 训练分支上的策略提取门控诊断','',
        '只使用固定训练数据；不读取验证/测试，不修改策略。每个锚点重新实现提取规则并与部署函数比较，全部一致。各阻断原因可重叠，不能相加当作独立原因。','',
        '|任务|种子|轮|锚点|漏选安全oracle|覆盖阻断|收益一致性阻断|安全一致性阻断|收益阻断含袋外成员|精确标签也不满足margin|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|%s|%d|%d|%d|%d|%d|%d|%d|%d|%d|'%(r['task'],r['seed'],r['round'],r['anchors'],r['missed_oracle_anchors'],r['support_blocks'],r['oracle_cost_consensus_blocks'],r['oracle_safety_consensus_blocks'],r['oracle_cost_blocks_with_out_of_bag_member'],r['exact_target_margin_blocks']))
    lines += ['', '这里oracle是已采样真实后缀上的相对安全最低成本动作，不是可部署策略；“漏选”不证明去掉任何门控后闭环会改善。所有锚点和候选保存在CSV，包含不安全选取和未成功源回合。','',
              '训练点是覆盖库的一部分，覆盖距离通常为零；因此训练锚点上覆盖门控不触发，不能推出它在新闭环状态中不起作用。大遗憾可能来自少数困难场景，后缀相互重叠，不能把总和当独立回合收益。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
