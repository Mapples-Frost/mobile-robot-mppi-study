"""Cross-check witness probe coverage and configured equations; export all results."""
import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

from gated_horizon_training_witness import ROOT, ART, OUT, SOURCE, STATE_NAMES, DT, SUBSTEPS, STEPS, read, digest, rhs, write


def main():
    registration = read(OUT / 'registration.json')
    report = read(OUT / 'audit.json')
    assert report['passed'] and report['training_only']
    assert digest(OUT / 'registration.json') == report['registration_hash']
    assert digest(OUT / 'results.json') == report['results_hash']
    for name, value in registration['hashes'].items():
        assert digest(ROOT / name) == value
    # Check the previously audited source traces still match the angular study.
    angular = read(SOURCE / 'registration.json')
    for case in registration['cases']:
        assert digest(ROOT / case['trace']) == angular['hashes'][case['trace']]
    config = read(ART / 'configs/pendulum.json')
    model = config['plant']['model']
    assert model['states'] == config['mpc']['model']['states']
    exprs = {k:compile(v['rhs'], '<pinned pendulum config>', 'eval') for k,v in model['states'].items()}
    paths = []
    results = read(OUT / 'results.json')['results']
    assert len(results) == 810
    audited = {(r['seed'],r['case'],r['controller']):r for r in report['all_candidates']}
    assert len(audited) == 810
    expected = {(c['seed'],c['case'],p['id']) for c in registration['cases'] for p in registration['controllers']}
    assert set(audited) == expected
    detail = []
    source_rhs_max_error = 0.
    steps = 0
    full_state_pairs = 0
    for result in results:
        key = (result['seed'],result['case'],result['controller'])
        a = audited[key]
        path = OUT / result['path']
        paths.append(path)
        assert digest(path) == result['sha256']
        saved = read(path)
        trace = saved['rows']
        assert trace == [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        assert len(trace) == result['steps'] == a['steps']
        assert all(x['next_state'] == y['state'] for x,y in zip(trace,trace[1:]))
        original = next(c for c in registration['cases'] if (c['seed'],c['case']) == key[:2])
        assert trace[0]['state'] == original['initial']
        for index,row in enumerate(trace):
            assert row['step'] == index and -5 <= row['input'] <= 5
            assert abs(row['state'][0]) <= 1.5 and abs(row['state'][2]) <= math.pi/2
            scope = dict(model['parameters'], np=np, u1=row['input'], **dict(zip(STATE_NAMES,row['state'])))
            expected_rhs = np.array([float(eval(exprs[k], {'__builtins__':{}}, scope)) for k in STATE_NAMES])
            calculated = rhs(np.array(row['state']),row['input'])
            np.testing.assert_allclose(calculated,expected_rhs,rtol=1e-12,atol=1e-12)
            source_rhs_max_error=max(source_rhs_max_error,float(np.max(abs(calculated-expected_rhs))))
            steps += 1
        last = trace[-1]['next_state']
        assert result['full_horizon'] == (len(trace)==100)
        if len(trace)<100:
            assert abs(last[0])>1.5 or abs(last[2])>math.pi/2
        detail.append(dict(seed=key[0],case=key[1],controller=key[2],steps_to_termination=len(trace),
            termination_time_seconds=len(trace)*DT,baseline_steps=original['baseline_steps'],
            numerical_witness=a['numerical_witness'],min_sampled_margin=a['min_sampled_margin'],
            final_pos=last[0],final_v=last[1],final_theta=last[2],final_omega=last[3],
            angular_violation=abs(last[2])>math.pi/2,cart_violation=abs(last[0])>1.5,
            max_state_discrepancy=a['max_state_discrepancy'],trace=result['path']))
    state=read(OUT/'status.json');audit_state=read(OUT/'audit_status.json')
    assert state['complete'] and not state['active'] and audit_state['complete'] and not audit_state['active']
    assert state['step_attempts']==state['step_completed']==steps==audit_state['integrations_attempted']==audit_state['integrations_completed']==12331
    assert state['rk4_rhs_evaluations']==steps*4*SUBSTEPS
    assert len(detail)==810 and all(not r['numerical_witness'] for r in detail)
    csv_path=OUT/'all_candidates.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    case_rows=[]
    for case in registration['cases']:
        group=[r for r in detail if (r['seed'],r['case'])==(case['seed'],case['case'])]
        assert len(group)==54
        case_rows.append(dict(seed=case['seed'],case=case['case'],candidates=54,numerical_witnesses=sum(r['numerical_witness'] for r in group),
            baseline_steps_to_termination=case['baseline_steps'],min_steps_to_termination=min(r['steps_to_termination'] for r in group),
            max_steps_to_termination=max(r['steps_to_termination'] for r in group),
            angular_termination_candidates=sum(r['angular_violation'] for r in group),cart_termination_candidates=sum(r['cart_violation'] for r in group)))
    summary_path=OUT/'all_cases.csv'
    with summary_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    smoke=read(OUT/'smoke.json')
    budget=dict(training_steps=0,diagnostic_step_attempts=steps,diagnostic_completed_steps=steps,synthetic_smoke_steps=smoke['environment_steps'],
        total_new_simulated_steps=steps+smoke['environment_steps'],rk4_rhs_evaluations=state['rk4_rhs_evaluations']+smoke['rk4_rhs_evaluations'],
        independent_formal_integrations=audit_state['integrations_completed'],independent_smoke_integrations=smoke['independent_integration_calls'],
        independent_formal_rhs_evaluations=audit_state['rhs_evaluations'],independent_smoke_rhs_evaluations=None,
        independent_smoke_rhs_limit='Synthetic smoke did not persist solve_ivp nfev; unknown, not zero. No repeat performed to replace missing instrumentation.',
        config_rhs_state_checks=steps,reset_warmups=0,nlp_calls=0,policy_or_terminal_updates=0,validation_test_episodes=0,
        accounting='Separate diagnostic budget; do not combine its LQR controls or trajectories with frozen gate evaluation.')
    write(OUT/'budget.json',budget)
    inputs=[OUT/n for n in ('registration.json','results.json','audit.json','status.json','audit_status.json','smoke.json','all_candidates.csv','all_cases.csv','budget.json')]
    check=dict(passed=True,cases=15,candidate_controllers=54,conditions=810,steps_checked=steps,
        cfg_rhs_max_discrepancy=source_rhs_max_error,maximum_independent_state_discrepancy=report['totals']['max_state_discrepancy'],
        journal_rows_match=True,all_case_combinations_covered=True,training_only=True,
        hashes={p.relative_to(ROOT).as_posix():digest(p) for p in inputs},source_hash=digest(Path(__file__)))
    write(OUT/'coverage_check.json',check)
    lines=['# 饱和LQR有限集合的可行轨迹搜索：负结果', '',
        '对上一项角度诊断中起点尚未被充分条件判定的全部15个倒立摆失败训练回合，本次预先登记54个饱和LQR控制器并运行完整810组合。没有找到满足相同输入界及采样状态约束、持续100步的轨迹。它排除了这组具体反馈参数的可行见证，不能证明初态不可行，也不支持任意LQR或其他控制器均失败。', '',
        '该搜索属于事后机制诊断。控制器以直立和固定小车位置为目标，未跟踪论文随机参考；因此即使成功也只提供有限时域可行轨迹，不构成论文控制成本、学习式自适应H或实时计算优势。15个案例来自已有训练，不是新验证或独立测试。','',
        '## 登记与实现','',
        '每个控制器由同一配置的直立连续线性化构建。Q=diag(qpos,1,qtheta,1)，qpos∈{0.1,1,10}、qtheta∈{10,100,1000}，R∈{0.01,0.1,1}，共27组；分别跟踪小车零位置或计分初始位置，共54组。u=clip(−K(x−target),−5,5)，每0.04 s更新，区间内保持。线性化数值导数、Riccati残差与闭环特征值均在登记前检查。','',
        '从已保存的第一个计分状态开始，无新reset或预热。RK4每控制步20个子步；在首个计分状态越界后停止，保存该越界步。停止步数包含失败步，不等于全部这些步都安全。两组合成冒烟共20步通过，最大积分器差为%.3g。'%smoke['maximum_state_error'],'',
        '## 所有案例','',
        '|种子/场景|候选数|100步可行见证|基线终止步|候选最早终止步|候选最晚终止步|',
        '|---|---:|---:|---:|---:|---:|']
    for r in case_rows:
        lines.append('|{seed}/{case}|{candidates}|{numerical_witnesses}|{baseline_steps_to_termination}|{min_steps_to_termination}|{max_steps_to_termination}|'.format(**r))
    lines += ['', '全部810候选明细见[all_candidates.csv](all_candidates.csv)，全部15案例汇总见[all_cases.csv](all_cases.csv)，逐步控制与状态见traces/。没有因失败早停后取部分代价均值排名，也没有只保留最晚失败的控制器。', '',
        '## 核验与预算','',
        '全部12,331个实际诊断步用DOP853（rtol=atol=10⁻¹²）从相同初态重放已保存控制；与RK4最大状态差%.6g，在既定10⁻⁷容差内。每步21个采样状态检查包括端点，但这不是严格的连续时间约束证明。所有候选已在计分端点失败，因此本次不存在依赖稀疏采样而声称成功的问题。'%report['totals']['max_state_discrepancy'],'',
        '额外核对配置文件原始动力学表达式与实现rhs在全部12,331个记录状态/输入的一致性，最大差%.6g；组合唯一键、JSONL逐步日志、整轨迹、状态连续性、终止步与耐久调用计数均核对。报告见[coverage_check.json](coverage_check.json)。'%source_rhs_max_error,'',
        '新增诊断仿真12,331步，加合成冒烟20步，共12,351步。RK4共988,080次rhs求值；正式独立积分12,331次、rhs求值432,083次，冒烟另20次独立积分，其nfev未保存，明确记未知。没有reset、NLP求解、训练更新或验证/测试回合；[budget.json](budget.json)单列这项开销，不把它隐去或算作主验证。','',
        '## 后续限制','',
        '简单局部线性反馈在这些大角度、受输入/位置约束的起点失败，并未解决NLP初值与物理可行性的区分。完整原实验验证继续运行，不因这项负结果缩窄场景或改变冻结规则。若继续搜索，应采用能明确处理非线性动力学和位置/输入约束的方案，并完整保留尝试；仍不能由求解失败推出不可行。','',
        'WSL项目根目录可复查，已完成运行会核对散列后复用，不重复控制仿真：','',
        '    .venv/bin/python experiments/bohn2021_reproduction/gated_horizon_training_witness.py --mode run',
        '    .venv/bin/python experiments/bohn2021_reproduction/gated_horizon_training_witness.py --mode audit',
        '    .venv/bin/python experiments/bohn2021_reproduction/gated_horizon_training_witness_report.py','']
    (OUT/'report_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=case_rows,budget=budget,source_rhs_max_error=source_rhs_max_error),indent=2))


if __name__=='__main__':
    main()
