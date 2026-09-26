"""Independently audit every pilot branch and describe policy-improvement signal."""
import csv
import json
from pathlib import Path
import numpy as np
from fixed_policy_branches import OUT, TASKS, BASE, HS, ANCHORS, verify, metrics
from branch_calibration_audit import audit_trace
from paper_h_soft_probe import digest, read
from run import write


def admissible(b, base):
    return (int(b['success']) >= int(base['success']) and
            int(b['constraint']) <= int(base['constraint']) and
            b['solver_failure_steps'] <= base['solver_failure_steps'])


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    verify()
    assert read(OUT / 'pilot_status.json')['complete']
    rows, anchors, summaries, budgets, hashes = [], [], [], [], {}
    for task in TASKS:
        for seed in range(3):
            dest = OUT / ('%s_s%d' % (task, seed))
            done = read(dest / 'completed.json')
            assert done['audit_passed'] and not done['smoke']
            assert done['inputs_hash'] == digest(OUT / 'inputs_sha256.json')
            for path, value in done['hashes'].items(): assert digest(Path(path)) == value
            cases = read(dest / 'train_bank.json')['cases']
            assert len(cases) == 2
            expected, skips = set(), []
            budget = dict(task=task, seed=seed, source_steps=0, prefix_steps=0,
                          suffix_steps=0, explicit_resets=2, environment_constructions=done['environment_constructions'])
            local = []
            for cid, case in enumerate(cases):
                original = read(dest / ('source_%02d.json' % cid))
                audit_trace(task, case, original)
                assert all(r['horizon'] == BASE[task] for r in original)
                budget['source_steps'] += len(original)
                budget['explicit_resets'] += 1
                for anchor in ANCHORS:
                    if anchor >= len(original):
                        skips.append(dict(case=cid, anchor=anchor, source=metrics(task, original)))
                        continue
                    expected.add((cid, anchor))
                    base = metrics(task, original[anchor:])
                    branches = []
                    for h in HS:
                        p = dest / ('case%02d_t%03d_h%02d.json' % (cid, anchor, h))
                        b = read(p)
                        assert (b['case'], b['anchor'], b['first_h']) == (cid, anchor, h)
                        tr = b['trace']
                        audit_trace(task, case, tr, offset=anchor)
                        assert tr[0]['previous_state'] == original[anchor]['previous_state']
                        assert tr[0]['observation'] == original[anchor]['observation']
                        assert tr[0]['horizon'] == h
                        assert all(r['horizon'] == BASE[task] for r in tr[1:])
                        assert metrics(task, tr) == b['metrics']
                        if h == BASE[task]: assert tr == original[anchor:]
                        budget['explicit_resets'] += 1
                        budget['prefix_steps'] += anchor
                        budget['suffix_steps'] += len(tr)
                        m = b['metrics']
                        row = dict(task=task, seed=seed, case=cid, anchor=anchor, h=h,
                            base_h=BASE[task], admissible=admissible(m, base),
                            base_cost=base['total_cost'], cost_difference=m['total_cost']-base['total_cost'])
                        row.update(m)
                        rows.append(row)
                        branches.append(row)
                    safe = [b for b in branches if b['admissible']]
                    chosen = min(safe, key=lambda b: (b['total_cost'], b['h'] != BASE[task], b['h']))
                    a = dict(task=task, seed=seed, case=cid, anchor=anchor,
                        base_cost=base['total_cost'], oracle_h=chosen['h'], oracle_cost=chosen['total_cost'],
                        oracle_gain=base['total_cost']-chosen['total_cost'],
                        unsafe_candidates=sum(not b['admissible'] for b in branches))
                    anchors.append(a)
                    local.append(a)
            assert expected == {(g['case'], g['anchor']) for g in done['groups']}
            assert skips == done['skipped']
            expected_steps = sum(budget[k] for k in ('source_steps', 'prefix_steps', 'suffix_steps'))
            attempts = [read(p) for p in dest.glob('attempt_*.json')]
            actual_steps = sum(a['step_calls'] for a in attempts)
            actual_resets = sum(a['reset_calls'] for a in attempts)
            assert actual_steps == done['explicit_step_calls'] >= expected_steps
            assert actual_resets == done['explicit_reset_calls'] >= budget['explicit_resets']
            budget.update(actual_step_calls=actual_steps, actual_reset_calls=actual_resets,
                          extra_or_interrupted_steps=actual_steps-expected_steps,
                          extra_or_interrupted_resets=actual_resets-budget['explicit_resets'])
            budgets.append(budget)
            denom = sum(a['base_cost'] for a in local)
            gain = sum(a['oracle_gain'] for a in local)
            summaries.append(dict(task=task, seed=seed, usable_anchors=len(local), skipped_anchors=len(skips),
                oracle_gain=gain, baseline_suffix_sum=denom, oracle_relative_gain=gain/max(abs(denom), 1e-12),
                improved_anchors=sum(a['oracle_gain'] > 1e-8 for a in local),
                unsafe_candidates=sum(a['unsafe_candidates'] for a in local)))
            hashes[str(dest / 'completed.json')] = digest(dest / 'completed.json')
    gates = {task: all(s['oracle_relative_gain'] >= .01 for s in summaries if s['task'] == task)
             for task in TASKS}
    write(OUT / 'mechanism_audit.json', dict(passed=True, summaries=summaries, budgets=budgets,
        training_only_mechanism_gate=gates, core_efficacy_evaluated=False, hashes=hashes,
        report_source_hash=digest(Path(__file__))))
    save_csv(OUT / 'all_branches.csv', rows)
    save_csv(OUT / 'all_anchors.csv', anchors)
    save_csv(OUT / 'all_seeds.csv', summaries)
    lines = ['# 固定策略分支机制实验：训练数据审计', '',
        '这是训练场景中的单步干预诊断，不是已学习策略的验证结果或独立复现证据。', '',
        '沿用各自独立训练 15k 步的固定 H 终端价值（车辆 H25、倒立摆 H30），只改变锚点第一步 H，随后恢复固定 H。回报为完整有限回合累计代价，无熵项、折扣或学习式尾部估计。', '',
        '|任务|种子|锚点/跳过|安全 oracle 降幅|改善锚点|安全不满足候选|',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('|%s|%d|%d/%d|%.3f%%|%d|%d|' % (s['task'],s['seed'],s['usable_anchors'],
            s['skipped_anchors'],100*s['oracle_relative_gain'],s['improved_anchors'],s['unsafe_candidates']))
    lines += ['', '训练机制门槛：每个种子的安全 oracle 累计改善均须至少 1%。结果：' + json.dumps(gates), '',
        'Oracle 使用真实后续回报事后挑选 H，仅作为可利用信号的上界；部署策略无法直接使用这些结果。各锚点后缀相互重叠，不能当作独立样本做显著性检验。两个训练场景不足以支持泛化结论。', '',
        '所有分支均重新核算物理动力学、代价、约束、终止和观测。固定 H 分支与保存源轨迹的后缀逐项相同。冒烟阶段两个任务的分支均作精确重复。失败和跳过锚点全部保留。', '',
        '正式采集显式 step 调用 %d 次、reset 调用 %d 次；每个 reset 另含一次 H50 warmup，环境构造另有初始化，不混充训练更新。新增参数更新为 0。' %
        (sum(b['actual_step_calls'] for b in budgets),sum(b['actual_reset_calls'] for b in budgets)), '',
        '继承预算：固定 H 搜索 300k 步，加所选固定 H 的种子 1/2 共 60k 步。此前其他方法和本轮冒烟预算分别见现有总账及 smoke_* 的 completed.json。本轮并行采集时间只用于资源核算，不用于声称加速。', '',
        '重跑现有冻结实验：TF1 Python 执行 fixed_policy_branches.py --mode suite --smoke；--mode suite；再运行 fixed_policy_branches_report.py。已完成任务复用并校验散列，不重复采集。首次创建冻结目录时使用 --mode freeze（会记录当时已有的全部实验脚本；后来新增脚本不追加入历史散列清单）。', '']
    (OUT / 'mechanism_report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gates=gates, summaries=summaries), indent=2))


if __name__ == '__main__': main()
