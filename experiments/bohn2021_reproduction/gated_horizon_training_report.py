"""Full finite-search ledger, with partial candidates explicitly excluded from ranking."""
import csv
import json
from pathlib import Path
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import candidates
from paper_h_soft_probe import read,digest
from run import write


def candidate_row(task,seed,result,baseline):
    full=result['fully_evaluated'];eligible=full and not result['rejected'] and result['mean_physical_cost']<=baseline['mean_physical_cost']+.02*abs(baseline['mean_physical_cost'])
    episodes=result['episodes'];steps=sum(e['steps'] for e in episodes)
    return dict(task=task,seed=seed,candidate=result['policy']['id'],episodes=len(episodes),steps=steps,fully_evaluated=full,eligible=bool(eligible),
        raw_mean_complete_bank=result['mean_raw_cost'] if full else None,physical_mean_complete_bank=result['mean_physical_cost'] if full else None,
        partial_raw_mean_descriptive_only=result['mean_raw_cost'] if not full else None,
        selection_cost=result['mean_raw_cost'] if eligible else None,successes=sum(e['success'] for e in episodes),constraints=sum(e['constraint'] for e in episodes),
        initial_failed_steps=sum(e['initial_failed_steps'] for e in episodes),final_failed_steps=sum(e['solver_failure_steps'] for e in episodes),
        retry_attempts=sum(e['retries'] for e in episodes),horizon_switches=sum(e['switches'] for e in episodes),
        step_weighted_mean_h=sum(e['mean_horizon']*e['steps'] for e in episodes)/steps,
        rejected_case=None if not result['rejected'] else result['rejected'][0]['case'],
        rejection_reasons=';'.join(reason for r in result['rejected'] for reason in r['reasons']),unrun_cases=';'.join(map(str,result['unrun_cases'])))


def main():
    freeze();audit=read(OUT/'audit_train.json');assert audit['passed']
    expected={(t,s,p['id']) for t in TASKS for s in range(3) for p in [dict(id='fixed')]+candidates(t)}
    audited={(g['task'],g['seed'],g['policy']['id']):g for g in audit['groups']};assert set(audited)==expected
    all_rows=[];all_episodes=[];selected=[];hashes={str(OUT/'audit_train.json'):digest(OUT/'audit_train.json')}
    for task in TASKS:
        for seed in range(3):
            folder=OUT/'train'/('%s_s%d'%(task,seed));fit=read(folder/'fit_completed.json');assert fit['policy_hash']==digest(folder/'policy.json')
            results=fit['results'];assert [r['policy']['id'] for r in results]==['fixed']+[p['id'] for p in candidates(task)]
            base=results[0];rows=[]
            for result in results:
                p=folder/result['policy']['id']/'completed.json';assert digest(p)==audited[(task,seed,result['policy']['id'])]['hash']
                assert read(p)['result']==result;hashes[str(p)]=digest(p)
                row=candidate_row(task,seed,result,base);rows.append(row);all_rows.append(row)
                all_episodes += [dict(task=task,seed=seed,candidate=result['policy']['id'],**e) for e in result['episodes']]
            chosen=min((r for r in rows if r['eligible']),key=lambda r:(r['selection_cost'],r['candidate']!='fixed',r['candidate']))
            assert fit['selected']['id']==chosen['candidate']
            selected.append(dict(task=task,seed=seed,candidate=chosen['candidate'],fallback_fixed=chosen['candidate']=='fixed',eligible_candidates=sum(r['eligible'] for r in rows),
                pruned_candidates=sum(bool(r['rejection_reasons']) for r in rows),baseline_raw_cost=base['mean_raw_cost'],selected_raw_cost=chosen['selection_cost'],
                training_cost_change_percent=100*(chosen['selection_cost']-base['mean_raw_cost'])/max(abs(base['mean_raw_cost']),1e-12),
                selected_physical_cost=chosen['physical_mean_complete_bank'],selected_successes=chosen['successes'],selected_constraints=chosen['constraints'],
                selected_initial_failures=chosen['initial_failed_steps'],selected_final_failures=chosen['final_failed_steps'],
                selected_horizon_switches=chosen['horizon_switches'],selected_mean_h=chosen['step_weighted_mean_h'],policy_path=str(folder/'policy.json')))
            for p in (folder/'fit_completed.json',folder/'policy.json'):hashes[str(p)]=digest(p)
    assert len(all_rows)==222 and len(selected)==6
    dest=OUT/'training_delivery';dest.mkdir(exist_ok=True)
    for name,rows in [('all_candidates.csv',all_rows),('all_training_episodes.csv',all_episodes),('all_selected_policies.csv',selected)]:
        with (dest/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(dest/'report.json',dict(passed=True,training_only=True,conditions=222,policies=selected,episodes=len(all_episodes),
        recorded_training_search_steps=sum(r['steps'] for r in all_rows),hashes=hashes,source_hash=digest(Path(__file__)),
        limitations='Selection-set scores are optimistically selected training summaries, not validation or test effects. Pruned partial means are descriptive only. Interrupted attempts, smoke, bank creation and inherited terminal learning are additional budget in separate ledger.'))
    lines=['# 门控时域：完整训练搜索','', '全部222个固定/候选条件、六个最终选择及逐回合结果均已导出。这里的收益来自选择所用训练场景，不构成泛化或独立复现证据。','',
        '|任务|种子|所选策略|合格候选|剪枝候选|固定成本|所选成本|训练成本变化%|所选H切换数|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:lines.append('|%s|%d|%s|%d|%d|%.6f|%.6f|%.6f|%d|'%(r['task'],r['seed'],r['candidate'],r['eligible_candidates'],r['pruned_candidates'],r['baseline_raw_cost'],r['selected_raw_cost'],r['training_cost_change_percent'],r['selected_horizon_switches']))
    lines += ['', '被剪枝候选保留首次失败场景、已运行回合及未运行场景编号；部分均值不会进入选择。固定H回退或零切换不算自适应效果。三个种子的训练场景各自独立生成，不用跨种子训练均值代替配对验证。','']
    (dest/'report_CN.md').write_text('\n'.join(lines));print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
