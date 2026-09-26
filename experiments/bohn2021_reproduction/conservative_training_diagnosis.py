"""Read-only in-bag/out-of-bag scene diagnosis; never selects a checkpoint."""
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,HS,BASE,verify
from conservative_policy_model import predict,choose
from paper_h_soft_probe import read,digest
from run import write


def main():
    verify();conditions=[];member_rows=[];groups_out=[];hashes={}
    for path in sorted(OUT.glob('*_s*_r*/fit_completed.json')):
        folder=path.parent;fit=read(path);data=read(folder/'collection_completed.json');model=read(folder/'policy.json')
        assert digest(folder/'policy.json')==fit['policy_hash'] and digest(folder/'collection_completed.json')==fit['dataset_hash']
        task=fit['task'];base_index=HS.index(BASE[task]);members=fit['members']
        rows=[]
        for g in data['groups']:
            costs=np.array([g['branches'][str(h)]['total_cost'] for h in HS])
            source=g['branches'][str(g['source_h'])]
            safe=np.array([b['success']>=source['success'] and b['constraint']<=source['constraint'] and b['solver_failure_steps']<=source['solver_failure_steps']
                for b in [g['branches'][str(h)] for h in HS]])
            delta=(costs-costs[base_index])/.1
            targets=np.sign(delta)*np.log1p(abs(delta))
            q,safety,distance=predict(model,g['context']['features'])
            selected=HS.index(choose(model,g['context']['features']))
            oracle=int(np.argmin(np.where(safe,costs,np.inf)))
            rows.append(dict(case=g['case'],anchor=g['anchor'],selected_h=HS[selected],source_h=g['source_h'],
                cost_difference_to_source=float(costs[selected]-source['total_cost']),oracle_h=HS[oracle],
                oracle_regret=float(costs[selected]-costs[oracle]),selected_safe=bool(safe[selected])))
            for member in range(3):
                pred=int(np.argmin(q[member]))
                member_rows.append(dict(task=task,seed=fit['seed'],round=fit['round'],member=member,
                    case=g['case'],anchor=g['anchor'],out_of_bag=g['case'] not in members[member]['bootstrap_scenes'],
                    transformed_mae=float(np.mean(abs(q[member]-targets))),
                    raw_argmin_h=HS[pred],raw_argmin_regret=float(costs[pred]-costs.min()),
                    raw_argmin_safe=bool(safe[pred]),
                    safety_false_positive=int(np.sum((safety[member]>=.8)&~safe)),
                    predicted_safe_candidates=int(np.sum(safety[member]>=.8))))
        groups_out.append(dict(task=task,seed=fit['seed'],round=fit['round'],groups=rows))
        matching=[r for r in member_rows if (r['task'],r['seed'],r['round'])==(task,fit['seed'],fit['round'])]
        summaries={}
        for name,out_of_bag in [('in_bag',False),('out_of_bag',True)]:
            rr=[r for r in matching if r['out_of_bag']==out_of_bag]
            summaries[name]=dict(member_anchor_count=len(rr),scene_count=len(set(r['case'] for r in rr)),
                transformed_mae=float(np.mean([r['transformed_mae'] for r in rr])) if rr else None,
                raw_argmin_mean_regret=float(np.mean([r['raw_argmin_regret'] for r in rr])) if rr else None,
                raw_argmin_unsafe=sum(not r['raw_argmin_safe'] for r in rr),
                safety_false_positive=sum(r['safety_false_positive'] for r in rr))
        conditions.append(dict(task=task,seed=fit['seed'],round=fit['round'],groups=len(rows),
            deployed_extraction_changed=sum(r['selected_h']!=r['source_h'] for r in rows),
            deployed_extraction_unsafe=sum(not r['selected_safe'] for r in rows),
            deployed_extraction_mean_regret=float(np.mean([r['oracle_regret'] for r in rows])),member_diagnostics=summaries))
        for p in (path,folder/'policy.json',folder/'collection_completed.json'):hashes[str(p)]=digest(p)
    assert conditions
    dest=OUT/'training_diagnosis';dest.mkdir(exist_ok=True)
    write(dest/'diagnosis.json',dict(conditions=conditions,members=member_rows,extraction=groups_out,
        hashes=hashes,source_hash=digest(Path(__file__)),
        limits='Descriptive training diagnosis only. Individual raw argmin ignores deployed safety/support/ensemble fallback. Out-of-bag scene was omitted by that member, but other ensemble members may have seen it. Shared normalization uses all training inputs. Not independent validation or a policy selection rule.'))
    lines=['# 训练场景内/外袋诊断（不选择模型）','',
        '每个成员按完整场景 bootstrap，外袋指该成员没有抽到的场景；其他成员可能见过该场景。归一化仍用全部训练输入。这不是新独立验证。以下逐成员原始 argmin 忽略部署安全/覆盖/一致性过滤，仅用来诊断回报排序。','',
        '|任务|种子|轮|锚点|部署改变/不安全|袋内 MAE/argmin regret|袋外 MAE/argmin regret|',
        '|---|---:|---:|---:|---:|---|---|']
    for c in conditions:
        a=c['member_diagnostics']['in_bag'];b=c['member_diagnostics']['out_of_bag']
        def fmt(m):return '%.3f / %.3f'%(m['transformed_mae'],m['raw_argmin_mean_regret']) if m['member_anchor_count'] else 'N/A'
        lines.append('|%s|%d|%d|%d|%d/%d|%s|%s|'%(c['task'],c['seed'],c['round'],c['groups'],c['deployed_extraction_changed'],c['deployed_extraction_unsafe'],fmt(a),fmt(b)))
    lines+=['','安全是分支标签上的相对经验判据，不保证部署安全。重叠锚点和三个成员的重复不作为独立样本检验。本诊断不改变冻结训练或部署规则。','']
    (dest/'report_CN.md').write_text('\n'.join(lines));print(json.dumps(conditions,indent=2))


if __name__=='__main__':main()
