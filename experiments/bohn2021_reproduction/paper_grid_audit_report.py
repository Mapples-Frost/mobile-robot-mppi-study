"""Strict coverage/accounting audit and complete paper-grid reconstruction report.

No simulator is used. Validation selects H; test-minimum is a labelled envelope.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import zipfile

from paper_grid_recover import ROOT, ART, OUT, OLD, REC, JOBS, name, read, write, verify

REPORT = ART / 'report/paper_exact_grid_2026-09-23'


def scalar(x):
    while isinstance(x, list):
        assert len(x) == 1
        x = x[0]
    return float(x)


def close(a, b, label):
    assert math.isfinite(a) and math.isfinite(b), label
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6), (label, a, b)


def physics(task, row, case, t):
    s = row['state']
    u = {k: scalar(v) for k, v in row['input'].items()}
    def ref(n):
        return case['tvp'][n][t + 2]['true'][0]
    if task == 'pendulum':
        cost = (.5 * .8 * s['v']**2 + .2 * .25 * s['v'] * s['omega'] * math.cos(s['theta'])
                + (2 / 3) * .2 * .25**2 * s['omega']**2 - .2 * 9.81 * .25 * math.cos(s['theta'])
                + 10 * (s['pos'] - ref('pos_r'))**2 + .1 * u['u1']**2)
        violation = abs(s['pos']) > 1.5 or abs(s['theta']) > math.pi / 2
        input_excess = max(0., abs(u['u1']) - 5)
    else:
        cost = (s['x'] - ref('trajectory_x'))**2 + (s['y'] - ref('trajectory_y'))**2
        violation = any(math.hypot(s['x'] - ref('obj_%d_x' % k), s['y'] - ref('obj_%d_y' % k))
                        <= ref('obj_%d_r' % k) for k in range(3))
        input_excess = max(0., -u['u_s'], u['u_s'] - 5, abs(u['u_omega']) - 4)
    return cost, violation, input_excess


def model_parameter_hash(path):
    # Hash the saved parameter payload, not timestamp-dependent zip metadata.
    with zipfile.ZipFile(path) as z:
        assert read_zip_data(z)['num_timesteps'] == 15000
        return hashlib.sha256(z.read('parameters')).hexdigest()


def read_zip_data(z):
    return json.loads(z.read('data'))


def audit_evaluation(folder, spec, mode, split):
    ev = folder / mode
    summary = read(ev / 'summary.json')
    cases = read(ART / 'configs' / (spec['task'] + '_' + split + '_bank.json'))['cases']
    n_expected = 20 if split == 'holdout' else 10
    assert len(cases) == len(summary['episodes']) == n_expected
    if split == 'holdout':
        completion = read(ev / 'completed.json')
        assert completion['frozen'] and completion['terminal_value'] == (mode == 'holdout_value')
        assert completion['model_hash'] == read(folder / 'completed.json')['final_hash']
    episode_values, checked_steps = [], 0
    input_excess_count, max_input_excess = 0, 0.
    for j, (episode, case) in enumerate(zip(summary['episodes'], cases)):
        trace = read(ev / ('trace_%02d.json' % j))
        assert episode['episode'] == j and len(trace) == episode['steps'] and trace
        assert episode['solver_failure_steps'] == sum(not r['solver_success'] for r in trace)
        sums = dict(performance=0., compute=0., constraint=0.)
        T = 100 if spec['task'] == 'pendulum' else 150
        assert len(trace) <= T
        for t, row in enumerate(trace):
            h = row['horizon']
            assert h in range(1, 51)
            if spec['fixed_horizon'] is not None:
                assert h == spec['fixed_horizon']
            cost, violated, excess = physics(spec['task'], row, case, t)
            close(cost, row['performance'], (folder.name, mode, j, t, 'physical cost'))
            close(row['compute'], h * (.003 if spec['task'] == 'pendulum' else .001), 'H cost')
            penalty = (10 if spec['task'] == 'pendulum' else 2) * (T - (t + 1)) if violated else 0.
            close(row['constraint'], penalty, (folder.name, mode, j, t, 'constraint penalty'))
            if violated:
                assert t == len(trace) - 1 and episode['termination'] == 'constraint'
            if t == len(trace) - 1 and episode['termination'] == 'constraint':
                assert violated
            if excess > 1e-5:
                input_excess_count += 1
            max_input_excess = max(max_input_excess, excess)
            close(-row['reward'], cost + row['compute'] + row['constraint'], 'reward total')
            for key in sums:
                sums[key] += row[key]
        for k, key in [('performance', 'performance_cost'), ('compute', 'computation_cost'), ('constraint', 'constraint_cost')]:
            close(sums[k], episode[key], (folder.name, j, key))
        close(sum(sums.values()), episode['total_cost'], 'episode cost')
        close(mean(r['horizon'] for r in trace), episode['mean_horizon'], 'mean H')
        episode_values.append(sum(sums.values()))
        checked_steps += len(trace)
    close(mean(episode_values), summary['mean_total_cost'], 'summary cost')
    assert summary['constraint_episodes'] == sum(e['termination'] == 'constraint' for e in summary['episodes'])
    assert summary['goal_episodes'] == sum(e['termination'] == 'goal' for e in summary['episodes'])
    return {'run': folder.name, 'mode': mode, 'episodes': n_expected, 'steps': checked_steps,
            'input_bound_excess_steps': input_excess_count, 'max_input_bound_excess': max_input_excess}


def row(folder, mode):
    spec = read(folder / 'manifest.json')
    s = read(folder / mode / 'summary.json')
    return {'run': folder.name, 'task': spec['task'], 'seed': spec['seed'],
            'fixed_horizon': spec['fixed_horizon'], 'mode': mode,
            'mean_total_cost': s['mean_total_cost'],
            'performance_cost': mean(e['performance_cost'] for e in s['episodes']),
            'H_cost': mean(e['computation_cost'] for e in s['episodes']),
            'constraint_cost': mean(e['constraint_cost'] for e in s['episodes']),
            'constraint_episodes': s['constraint_episodes'], 'goal_episodes': s['goal_episodes'],
            'solver_failure_steps': sum(e['solver_failure_steps'] for e in s['episodes']),
            'steps': sum(e['steps'] for e in s['episodes']),
            'mean_H': mean(e['mean_horizon'] for e in s['episodes']),
            'scene_costs': [e['total_cost'] for e in s['episodes']]}


def main():
    verify()
    folders = [OUT / name(j) for j in JOBS]
    folders += sorted(p for p in OLD.iterdir() if p.is_dir() and (p / 'completed.json').exists())
    assert len(folders) == 26
    audits, rows, manifests = [], [], []
    for folder in folders:
        spec, done = read(folder / 'manifest.json'), read(folder / 'completed.json')
        assert done['steps'] == 15000 and done['updates'] == 14745
        assert done['weights_changed'] and done['evaluation_frozen']
        adaptation = spec['adaptations']
        assert adaptation == {'aligned': True, 'scaled_obs': True, 'ent_coef': '1.0',
                              'no_online_value': False, 'batch_size': 256, 'buffer_size': 1000000}
        for field, path in [('config_sha256', ART / 'configs' / (spec['task'] + '.json')),
                            ('test_bank_sha256', ART / 'configs' / (spec['task'] + '_validation_bank.json'))]:
            assert spec[field] == hashlib.sha256(path.read_bytes()).hexdigest()
        model_parameter_hash(folder / 'model.zip')
        for mode, split in [('eval_value', 'validation'), ('eval_no_value', 'validation'),
                            ('holdout_value', 'holdout'), ('holdout_no_value', 'holdout')]:
            audits.append(audit_evaluation(folder, spec, mode, split))
            rows.append(row(folder, mode))
        manifests.append(spec)
    for task in ['pendulum', 'vehicle']:
        subset = [s for s in manifests if s['task'] == task]
        assert sorted(s['seed'] for s in subset if s['fixed_horizon'] is None) == [0, 1, 2]
        assert sorted(s['fixed_horizon'] for s in subset if s['fixed_horizon'] is not None) == list(range(5, 51, 5))
    inventory = read(REC / 'inventory.json')
    interrupted = [e for e in inventory if e['state'] == 'interrupted']
    for e in interrupted:
        for rel, digest in e['files'].items():
            path = REC / 'interrupted' / rel
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
    prefix_checks = []
    for e in interrupted:
        before, after = REC / 'interrupted' / e['name'], OUT / e['name']
        old_episodes = read(before / 'training_episodes.json') if (before / 'training_episodes.json').exists() else []
        new_episodes = read(after / 'training_episodes.json')
        assert old_episodes == new_episodes[:len(old_episodes)], e['name']
        check = {'name': e['name'], 'training_episode_prefix_exact': True, 'prefix_episodes': len(old_episodes)}
        if (before / 'model.zip').exists():
            check['final_parameter_payload_exact'] = model_parameter_hash(before / 'model.zip') == model_parameter_hash(after / 'model.zip')
            assert check['final_parameter_payload_exact'], check
        prefix_checks.append(check)
    budget = {'inherited_complete_training_steps': 18 * 15000,
              'new_recovery_training_steps': 8 * 15000,
              'retained_total_training_steps': 26 * 15000,
              'interrupted_observed_training_steps': sum(e['saved_progress'].get('steps', 0) for e in interrupted),
              'retained_validation_episodes': 520, 'retained_holdout_episodes': 1040,
              'retained_validation_steps': sum(a['steps'] for a in audits if a['mode'].startswith('eval_')),
              'retained_holdout_steps': sum(a['steps'] for a in audits if a['mode'].startswith('holdout_')),
              'new_holdout_episodes': 560,
              'scope': 'This reconstruction comparison only, not all Bohn studies. Interrupted unflushed work and historical reset overhead are not exhaustively known. Retained episodes are not independent training replicates.'}
    comparisons = []
    for task in ['vehicle', 'pendulum']:
        validation = [r for r in rows if r['task'] == task and r['mode'] == 'eval_value' and r['fixed_horizon'] is not None]
        selected = min(validation, key=lambda r: (r['mean_total_cost'], r['fixed_horizon']))
        val_h = selected['fixed_horizon']
        for mode in ['holdout_value', 'holdout_no_value']:
            subset = [r for r in rows if r['task'] == task and r['mode'] == mode]
            fixed = [r for r in subset if r['fixed_horizon'] is not None]
            rl = sorted([r for r in subset if r['fixed_horizon'] is None], key=lambda r: r['seed'])
            baseline = next(r for r in fixed if r['fixed_horizon'] == val_h)
            envelope = min(fixed, key=lambda r: r['mean_total_cost'])
            delta = [r['mean_total_cost'] - baseline['mean_total_cost'] for r in rl]
            comparisons.append({'task': task, 'mode': mode, 'validation_selected_H': val_h,
                                'fixed_holdout_cost': baseline['mean_total_cost'],
                                'rl_mean': mean(r['mean_total_cost'] for r in rl),
                                'rl_sd_across_seeds': stdev(r['mean_total_cost'] for r in rl),
                                'paired_seed_cost_differences': delta,
                                'paired_scene_differences_by_seed': [[a-b for a,b in zip(r['scene_costs'], baseline['scene_costs'])] for r in rl],
                                'all_rl_seeds_lower_cost': all(d < 0 for d in delta),
                                'test_minimum_H_descriptive': envelope['fixed_horizon'],
                                'test_minimum_cost_descriptive': envelope['mean_total_cost']})
    audit = {'passed': True, 'models': 26, 'validation_and_holdout_episode_conditions': 1560,
             'checks': ['complete ten-H grid and three RL seeds per task', 'unchanged core/config/bank hashes',
                        '15000 model timesteps and 14745 updates', 'frozen evaluation model hash',
                        'independent physical/H/failure costs, every step', 'physical constraint termination',
                        'all failures retained', 'interruption archive hashes', 'restart episode prefixes'],
             'bounds': 'Input excess beyond 1e-5 is reported as a feasibility result, not hidden by cost audit.',
             'input_bound_excess_steps': sum(a['input_bound_excess_steps'] for a in audits),
             'evaluations': audits, 'recovery_prefix_checks': prefix_checks, 'budget': budget,
             'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT / 'audit_2026-09-24.json', audit)
    write(REPORT / 'results.json', {'rows': rows, 'comparisons': comparisons, 'budget': budget})
    lines = ['# Bøhn 2021：完整固定时域网格重建', '',
             '已完成两任务各十个固定 H 与三个 RL 种子，共 26 个模型。每模型 15,000 训练步；固定 H 各自独立学习终端价值，RL 与终端估计器联合训练。',
             '保持作者 SAC 核心、连续输出取整 H、32 步终端学习、actor 32×32、gamma=rho=0.97、batch 256、replay 1e6、固定熵系数 1，以及正确的任务成本权重。初始分布、观测处理、缺失作者实验配置等重建假设仍需披露。', '',
             '## 同一保留集上的比较', '',
             '成本越低越好。主列固定 H 由原验证集平均总成本选择；括号内的测试最低 H 仅描述完整网格，不当作独立选择的基线。历史保留集早已观察过，本轮是补齐已登记比较，不是全新确认实验。', '',
             '|任务|终端项|验证选 H|固定成本|RL 均值±种子SD|三个 RL−固定差值|测试最低 H/成本（描述性）|',
             '|---|---|---:|---:|---:|---|---|']
    for c in comparisons:
        lines.append('|%s|%s|%d|%.3f|%.3f ± %.3f|%s|H%d / %.3f|' %
                     (c['task'], '有' if c['mode'] == 'holdout_value' else '无', c['validation_selected_H'],
                      c['fixed_holdout_cost'], c['rl_mean'], c['rl_sd_across_seeds'],
                      ', '.join('%+.3f' % v for v in c['paired_seed_cost_differences']),
                      c['test_minimum_H_descriptive'], c['test_minimum_cost_descriptive']))
    lines += ['', '无终端项为同一已训练模型清零 MPC 终端值的冻结干预，不等价于从头训练无终端方法。', '',
              '## 全部有终端模型', '',
              '|任务|模型|种子|总成本|物理成本|H成本|约束成本|约束终止/20|到达/20|求解失败/步|平均H|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    values = sorted([r for r in rows if r['mode'] == 'holdout_value'],
                    key=lambda r: (r['task'], r['fixed_horizon'] is None, r['fixed_horizon'] or 0, r['seed']))
    for r in values:
        lines.append('|%s|%s|%d|%.3f|%.3f|%.3f|%.3f|%d|%d|%d/%d|%.2f|' %
                     (r['task'], 'RL' if r['fixed_horizon'] is None else 'H%d' % r['fixed_horizon'], r['seed'],
                      r['mean_total_cost'], r['performance_cost'], r['H_cost'], r['constraint_cost'],
                      r['constraint_episodes'], r['goal_episodes'], r['solver_failure_steps'], r['steps'], r['mean_H']))
    lines += ['', '## 审计、恢复与限制', '',
              '全部 520 个验证和 1,040 个保留评估条件完成逐步成本与约束核对；旧模型与冻结配置哈希保持一致。归档的中断数据逐文件哈希核对通过，重跑已保存训练回合前缀核对通过。',
              '新增恢复训练 120,000 步；保留 26 模型合计 390,000 步。中断训练可观察下界 %d 步另计；未刷盘工作和旧 reset 开销不能冒充精确总量。' % budget['interrupted_observed_training_steps'],
              '输入界限超出 1e-5 的步数：%d；这是可行性结果，独立于成本核算是否通过。' % audit['input_bound_excess_steps'],
              '每个固定 H 只有 seed0，RL 三个训练种子；每任务仅 20 个配对场景。不能将多条件评估视为 1,040 个独立样本。并行墙钟耗时不构成实时加速证据，H 是论文计算代理。',
              '作者原始实验配置、测试集及完整训练工程未恢复，因此即使某个局部结果较好，也不能宣称逐数值复现。完整负结果、全部网格和无终端结果均保存在 results.json。']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(REC / 'final_audit_complete.json', {'passed': True, 'report': str((REPORT / 'report.md').relative_to(ROOT))})
    print(json.dumps({'audit_passed': True, 'comparisons': [{k:v for k,v in c.items() if k != 'paired_scene_differences_by_seed'} for c in comparisons], 'budget': budget}, indent=2))


def check():
    case = {'tvp': {'pos_r': [{'true': [0.]}] * 3}}
    row0 = {'state': dict(pos=0., v=0., theta=0., omega=0.), 'input': {'u1': [[0.]]}}
    cost, violated, excess = physics('pendulum', row0, case, 0)
    close(cost, -.4905, 'equilibrium energy reference')
    assert not violated and excess == 0
    row0['state']['pos'] = 1.6
    assert physics('pendulum', row0, case, 0)[1]
    row0['input']['u1'] = [[6.]]
    assert physics('pendulum', row0, case, 0)[2] == 1
    for task in ['pendulum', 'vehicle']:
        folder = OLD / (task + '_rl_s0')
        result = audit_evaluation(folder, read(folder / 'manifest.json'), 'holdout_value', 'holdout')
        assert result['episodes'] == 20
    print('Independent equilibrium, constraint and actuator-bound checks passed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    check() if args.check else main()
