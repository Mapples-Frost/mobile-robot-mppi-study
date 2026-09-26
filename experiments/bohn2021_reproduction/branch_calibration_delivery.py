"""Build complete seed/scene tables and figures from independently audited data."""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24'
OUT = BASE / 'delivery'
ARMS = ('actor', 'raw_greedy', 'calibrated_greedy', 'fixed')
LABELS = ('Min-Q actor', 'Raw Q greedy', 'Calibrated Q', 'Fixed H')
COLORS = ('#0072B2', '#E69F00', '#CC79A7', '#009E73')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(p, require_pass=True):
    d = read(p)
    if require_pass: assert d['passed']
    for name, h in d['hashes'].items(): assert sha(Path(name)) == h, name
    return d


def table(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'plot_environment.json').write_text(json.dumps({'python':sys.version, 'executable':sys.executable,
        'platform':platform.platform(), 'numpy':np.__version__, 'matplotlib':matplotlib.__version__},indent=2)+'\n')
    for p, h in read(BASE / 'inputs_sha256.json').items(): assert sha(Path(p)) == h, p
    amendment = BASE / 'numeric_amendment.json'
    if amendment.exists():
        for p, h in read(amendment)['hashes'].items(): assert sha(Path(p)) == h, p
        provenance = read(BASE / 'numeric_training_audit_provenance.json')
        assert provenance['audit_hash'] == sha(BASE / 'training_audit.json')
        assert provenance['amendment_hash'] == sha(amendment)
    claim = read(BASE/'claim_requirements.json')
    assert claim['original_gate_required']
    train = checked(BASE / 'training_audit.json')
    gate = checked(BASE / 'validation_gate.json', False)
    assert gate['data_audit_passed']
    if not gate['passed']: assert not (BASE / 'evaluations/test').exists()
    rows, scene_rows = [], []
    for r in gate['rows']:
        t, s, a = r['task'], r['seed'], r['arm']
        p = BASE / 'evaluations/validation' / t / ('%s_s%d' % (a, s))
        summary = read(p / 'summary.json')
        eps = summary['episodes']
        success = sum(e['termination'] == ('goal' if t == 'vehicle' else 'steps') for e in eps)
        row = {'task': t, 'seed': s, 'arm': a, 'mean_cost': r['cost'],
               'mean_physical_cost': np.mean([e['performance_cost'] for e in eps]),
               'mean_h_proxy_cost': np.mean([e['computation_cost'] for e in eps]),
               'mean_constraint_cost': np.mean([e['constraint_cost'] for e in eps]),
               'successes': success, 'episodes': len(eps), 'constraint_stops': r['constraints'],
               'solver_failed_steps': r['solver_failures'], 'steps': r['steps'],
               'mean_h': sum(e['mean_horizon'] * e['steps'] for e in eps) / r['steps']}
        rows.append(row)
        for e in eps:
            scene_rows.append(dict(task=t, seed=s, arm=a, **e))
    table('validation_seeds.csv', rows)
    table('validation_episodes.csv', scene_rows)
    differences = []
    for r in gate['differences']:
        for j, value in enumerate(r['paired_scene_differences']):
            differences.append({'task': r['task'], 'seed': r['seed'], 'comparator': r['comparator'],
                                'case': j, 'calibrated_minus_comparator': value})
    table('paired_scene_differences.csv', differences)
    budget = {k: sum(r['budget'][k] for r in train['rows']) for k in train['rows'][0]['budget']}
    budget['critic_updates'] = sum(read(BASE / ('%s_s%d/fit.json' % (r['task'], r['seed'])))['steps'] for r in train['rows'])
    budget['attempted_step_calls'] = sum(r['attempted_calls']['step_calls'] for r in train['rows'])
    budget['attempted_reset_calls'] = sum(r['attempted_calls']['reset_calls'] for r in train['rows'])
    budget['validation_episodes'] = len(scene_rows)
    budget['validation_steps'] = sum(r['steps'] for r in rows)
    budget['validation_reset_calls'] = len(scene_rows)
    budget['smoke'] = checked(BASE / 'smoke_audit.json')['rows'][0]
    budget['inherited'] = read(BASE / 'protocol.json')['budget']
    budget['bank_generation'] = {'explicit_resets': 2 * (3 * 8 + 10 + 20), 'environments_constructed': 10,
        'limit': 'Each environment constructor may also reset. Initialization, audit replay and failed attempts are separate from collected transitions.'}
    attempts = []
    for p in sorted(BASE.rglob('attempt_*.json')):
        d = read(p)
        attempts.append({'path': str(p), 'reset_calls': d['reset_calls'], 'step_calls': d['step_calls'], 'sha256': sha(p)})
    for p in sorted((BASE / 'serial_timing').glob('r*/attempt.json')):
        d = read(p)
        attempts.append({'path': str(p), 'reset_calls': d['reset_attempts'], 'step_calls': d['step_attempts'], 'sha256': sha(p)})
    budget['all_instrumented_attempts'] = attempts
    budget['all_instrumented_explicit_calls'] = {k: sum(d[k] for d in attempts) for k in ('reset_calls', 'step_calls')}
    timing_attempts = [d for d in attempts if '/serial_timing/' in d['path']]
    budget['serial_timing_instrumented_calls'] = {k: sum(d[k] for d in timing_attempts) for k in ('reset_calls', 'step_calls')}
    budget['serial_timing_completed_conditions'] = len(list((BASE/'serial_timing').glob('r*/completed.json')))
    check = BASE / 'serial_timing/instrumentation_check.json'
    if check.exists(): budget['timing_instrumentation_smoke'] = read(check)
    (OUT / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
    for ti, t in enumerate(('vehicle', 'pendulum')):
        for ai, a in enumerate(ARMS):
            rr = sorted([r for r in rows if r['task'] == t and r['arm'] == a], key=lambda r:r['seed'])
            for k, r in enumerate(rr):
                x = ai + (k-1)*.13
                axs[ti, 0].scatter(x, r['mean_cost'], color=COLORS[ai], marker=('o','s','^')[k], s=45)
                axs[ti, 1].scatter(x, r['successes']/r['episodes'], color=COLORS[ai], marker=('o','s','^')[k], s=45)
                axs[ti, 2].scatter(x, r['solver_failed_steps'], color=COLORS[ai], marker=('o','s','^')[k], s=45)
        for ax in axs[ti]:
            ax.set_xticks(range(4), LABELS, rotation=18, ha='right')
            ax.grid(axis='y', alpha=.2)
        axs[ti, 0].set_ylabel(t.capitalize())
        if t == 'vehicle':
            axs[ti, 0].set_yscale('log')
            axs[ti, 0].set_ylabel('Vehicle (log cost axis)')
        else:
            axs[ti, 0].set_ylim(bottom=0)
        axs[ti, 2].set_ylim(bottom=-max(.5,.03*max(r['solver_failed_steps'] for r in rows if r['task']==t)))
        axs[ti, 1].set_ylim(-.03, 1.06)
    for ax, title in zip(axs[0], ('Mean total cost', 'Success fraction', 'Solver-failed steps')):
        ax.set_title(title)
    fig.suptitle('Validation: all three training seeds; circle=0, square=1, triangle=2\nPendulum success = survives full episode within state constraints')
    for ext in ('png', 'pdf'): fig.savefig(OUT / ('validation_all_seeds.' + ext), dpi=200)
    plt.close(fig)
    lines = ['# Bøhn 2021 分支回报校准：验证阶段交付', '', '## Material Passport', '',
        '- 范围：车辆与倒立摆；原作者核心代码重建之上的改进方法，非逐数字原论文复现。',
        '- 数据：原始仿真轨迹；数据审计通过不等于方法有效。',
        '- 本报告以固定协议、六组模型和全部验证原始结果为依据。', '',
        '**验证门槛：%s。** %s' % ('通过' if gate['passed'] else '未通过',
            '独立测试需另外完成审计。' if gate['passed'] else '对应独立测试未开启；不能宣称复现成功。'), '',
        '[全部种子验证图 PNG](validation_all_seeds.png) · [PDF](validation_all_seeds.pdf) · [逐种子 CSV](validation_seeds.csv) · [逐场景 CSV](validation_episodes.csv)', '',
        '校准只更新双 Q 网络，原 actor、终端多项式与 target-V 冻结。两次共享噪声后缀的平均 soft return 作为中心化监督标签；训练固定 1000 次更新，无验证选 checkpoint。', '',
        '本表 actor 指复用的 min-Q 改进模型的未改动 actor，不是最初作者 Q1 actor 方法；raw_greedy 和 calibrated_greedy 也均从该 min-Q 模型出发。原方法重建结果应查阅单独的 paper-grid 报告，不与本轮场景直接混表比较。', '',
        '|任务|种子|方法|总成本|物理成本|H 代理|成功/10|约束回合|求解失败步|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|{task}|{seed}|{arm}|{mean_cost:.3f}|{mean_physical_cost:.3f}|{mean_h_proxy_cost:.3f}|{successes}/10|{constraint_stops}|{solver_failed_steps}|'.format(**r))
    lines += ['', '车辆成功指到达；倒立摆成功指无状态越界地运行至时限，不能解释为严格稳态跟踪成功。全部逐场景成本、配对差及失败保留于 CSV。三个种子没有被场景重复数替代；不基于这点样本声称总体统计显著。', '',
        '## 预算与公平性', '',
        '沿用的固定 H 搜索为 5、10、…、50，各自独立学习终端价值，共 300,000 步；车辆 H25、倒立摆 H30 在旧验证集选定，额外种子共 60,000 步。原 actor 与 min-Q 各 90,000 步。这是当前比较保留模型预算，不是全历史总预算。', '',
        '新校准采集 %(source_steps)d 步原策略轨迹、%(prefix_steps)d 步前缀、%(suffix_steps)d 步后缀，共 %(rollouts)d 个分支，%(resets)d 次显式重置，%(critic_updates)d 次监督梯度更新。另有 %(validation_episodes)d 个验证回合、%(validation_steps)d 步。' % budget, '',
        '全部尝试计数、冒烟、生成场景、重置预热与继承预算见 budget.json。原始 attempt 文件可计入中断开销。额外分支仿真仅提供给校准方法，因此不能声称等训练计算预算优势。固定 H 不需要学习用于选 H 的 critic。', '',
        '## 解释边界', '',
        '训练拟合度不是闭环收益。分支标签只改首步 H、之后恢复原随机 actor，重复贪心部署改变了后续策略与状态分布；仅两次采样可能产生排序噪声。target-V 尾项仍是学习估计，非真实无穷期回报。', '',
        'H 线性成本只作为论文代理。并行验证日志中的 elapsed_s 包含记录开销，不作为加速证据。串行计时采用独立冻结协议，若尚未完成，不能补写耗时结论。', '',
        '## 复运行', '', '在 WSL 项目根目录使用固定 Python3.7：', '', '```bash',
        'PY=/home/mapples/.local/share/bohn2021-python37/bin/python',
        '$PY experiments/bohn2021_reproduction/branch_calibration_audit.py smoke --replay',
        '# 数值审计补丁保留原冻结源码；逐模型复用已存数据，缺失时才补跑：',
        'for TASK in vehicle pendulum; do',
        '  for SEED in 0 1 2; do',
        '    $PY experiments/bohn2021_reproduction/branch_calibration_resume.py train --task "$TASK" --seed "$SEED"',
        '  done',
        'done',
        '$PY experiments/bohn2021_reproduction/branch_calibration_resume.py audit --phase training',
        '$PY experiments/bohn2021_reproduction/branch_calibration_run.py evaluate --split validation',
        '$PY experiments/bohn2021_reproduction/branch_calibration_audit.py validation',
        '# 只在验证 passed=true 且审计通过后运行 evaluate --split test 与 audit.py test。',
        '# 无其他项目仿真进程时运行串行计时：',
        '$PY experiments/bohn2021_reproduction/branch_calibration_timing.py',
        '.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_timing_report.py',
        '.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_failure_diagnosis.py',
        '.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_claim_audit.py',
        '.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py', '```', '',
        '已完成运行会校验并复用，不要删除旧产物来制造独立重复。环境版本保存于 sources/runtime-freeze.txt；本轮模型、输入 SHA256、配置和协议均保留在实验目录。']
    if amendment.exists():
        lines += ['', '## 中断与修复披露', '',
            '车辆 seed0 与倒立摆 seed2 因独立 log-Jacobian 审计舍入顺序中断，问题 TensorFlow 采样记录重算逐值一致。补丁改用 TF1 的常量合并顺序，4e-5 容差未放宽；原始标签、轨迹、采样、拟合与效果门槛均未更改。全部旧分支复用，失败日志和尝试预算保留。numeric_amendment.json 固定补丁及诊断散列，numeric_training_audit_provenance.json 关联最终审计。',
            '故意改变对数概率、奖励或共享噪声的内存副本均被审计拒绝，检查记录见 numeric_corruption_check.json；未修改原始文件。']
    lines += ['', '## 用户目标的额外结论要求', '',
        '在任何新验证轨迹生成前，claim_requirements.json 已登记：原效果门槛通过仍不足以宣布目标完成。独立测试各训练种子的任务成功次数还必须不低于全部对照；原门槛报告成功次数但没有将其列入布尔判据。这个补充没有放宽原标准，也未改变模型或选模。',
        '独立加速结论必须另外具有全部独立测试场景的串行计时证据；只有验证计时则只报告描述性结果。作者原始完整配置缺失的限制继续保留。']
    ranking = BASE/'training_ranking_diagnosis.json'
    if ranking.exists():
        diagnosis = read(ranking)
        for p,h in diagnosis['hashes'].items(): assert sha(Path(p)) == h
        checked(BASE/'training_ranking_diagnosis_completed.json')
        rfig, raxs = plt.subplots(1, 2, figsize=(10, 4.7), layout='constrained')
        for ax, task in zip(raxs, ('vehicle', 'pendulum')):
            subset = [r for r in diagnosis['rows'] if r['task']==task]
            maximum = max(max(r['raw_label_regret'], r['calibrated_label_regret']) for r in subset)
            ax.plot([0, maximum], [0, maximum], color='#999999', linestyle='--', linewidth=1)
            for seed, color in enumerate(('#0072B2','#D55E00','#009E73')):
                selected = [r for r in subset if r['seed']==seed]
                ax.scatter([r['raw_label_regret'] for r in selected], [r['calibrated_label_regret'] for r in selected],
                           color=color, label='Seed %d'%seed, alpha=.75, s=32)
            ax.set_xscale('symlog', linthresh=.01); ax.set_yscale('symlog', linthresh=.01)
            ax.set_xlim(-.002, maximum*1.15); ax.set_ylim(-.002, maximum*1.15)
            ax.set_xlabel('Raw Q label regret (symlog)'); ax.set_ylabel('Calibrated Q label regret (symlog)')
            ax.set_title(task.capitalize()); ax.grid(alpha=.2); ax.legend(fontsize=8)
        rfig.suptitle('Training anchors only: lower than diagonal = closer to sampled best action\nTwo Monte Carlo returns per candidate; this is not generalization evidence')
        for ext in ('png','pdf'): rfig.savefig(OUT/('training_label_regret.'+ext),dpi=200)
        plt.close(rfig)
        lines += ['', '## 训练排序诊断（事后描述）', '',
            '|任务|种子|锚点数|原 Q 标签遗憾|校准后标签遗憾|两次采样最优 H 不一致|',
            '|---|---:|---:|---:|---:|---:|']
        for r in diagnosis['summary']:
            lines.append('|{task}|{seed}|{groups}|{raw_mean_label_regret:.4f}|{calibrated_mean_label_regret:.4f}|{replicate_best_h_disagreements}|'.format(**r))
        lines += ['', '标签遗憾只相对于同一训练锚点上两个采样回报的平均最优候选，并非真实最优策略遗憾；下降不能作为独立泛化证据。此诊断没有改变协议或重新选择模型。']
        lines += ['', '[训练锚点排序图 PNG](training_label_regret.png) · [PDF](training_label_regret.pdf)']
        lines += ['', '|任务|种子|原 actor 首步标签遗憾|固定 H 首步标签遗憾|两次采样中心化差异均值|',
            '|---|---:|---:|---:|---:|']
        for r in diagnosis['summary']:
            lines.append('|{task}|{seed}|{actor_mean_first_action_label_regret:.4f}|{fixed_mean_first_action_label_regret:.4f}|{mean_abs_centered_between_repeat_difference:.4f}|'.format(**r))
        lines += ['', '这里的固定 H 只强制当前第一步，后续仍是原随机 actor；不能代替完整固定 H 控制器。两次采样差异是描述量，不是标准误或置信区间。']
    timing = BASE/'serial_timing/timing_audit.json'
    if timing.exists():
        tr = read(timing)
        assert tr['passed']
        for p, h in tr['input_hashes'].items(): assert sha(Path(p)) == h, p
        assert budget['serial_timing_completed_conditions'] == 48
        assert budget['serial_timing_instrumented_calls']['step_calls'] == 2*budget['validation_steps']
        assert budget['serial_timing_instrumented_calls']['reset_calls'] == 2*budget['validation_episodes']
        lines += ['', '## 实测计时', '',
            '全部 48 个串行条件已完成，两次重复、全部种子和场景，逐轨迹与效果验证一致。另计 480 次显式重置、41,870 次仿真步；没有增加训练样本或独立场景数。完整数值与环境限制见 [串行计时中文报告](../serial_timing/report_CN.md)，逐步原始耗时和每种子表保留。验证计时不替代独立效果确认。']
        lines += ['', '[实测耗时—控制成本图 PNG](../serial_timing/measured_tradeoff.png) · [PDF](../serial_timing/measured_tradeoff.pdf) · [逐种子耗时 CSV](../serial_timing/seed_timings.csv)']
    failure = BASE/'failure_coverage_diagnosis.json'
    if failure.exists():
        fd = read(failure)
        assert fd['validation_gate_sha256'] == sha(BASE/'validation_gate.json')
        for p, h in fd['hashes'].items(): assert sha(Path(p)) == h, p
        lines += ['', '## 完整失败诊断', '',
            '全部 240 回合的训练锚点覆盖距离、H 范围、首次求解失败和场景配对差见 [失败诊断](../failure_coverage_CN.md)。它是事后描述，没有删场景、重新训练或调整门槛；距离不构成分布偏移的因果证明。']
    (OUT / 'report_CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    inputs = [BASE/p for p in ('protocol.json', 'inputs_sha256.json', 'training_audit.json',
        'validation_gate.json', 'claim_requirements.json', 'numeric_amendment.json',
        'numeric_training_audit_provenance.json', 'training_ranking_diagnosis.json',
        'training_ranking_diagnosis_completed.json', 'failure_coverage_diagnosis.json',
        'claim_audit.json', 'serial_timing/timing_audit.json') if (BASE/p).exists()]
    inputs.append(Path(__file__))
    (OUT / 'manifest.json').write_text(json.dumps({'validation_gate_passed': gate['passed'],
        'input_hashes': {str(p): sha(p) for p in inputs},
        'output_hashes': {str(p): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'}}, indent=2)+'\n')
    print(OUT / 'report_CN.md')


if __name__ == '__main__': main()
