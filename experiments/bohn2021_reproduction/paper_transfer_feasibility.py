"""Validation-only angular certificate and observed actuator-bound diagnosis.

Read existing trajectories only; no MPC, retraining, or holdout access.
"""
import math
from paper_transfer_common import ART, ROOT, OUT, DOMAINS, read, write, sha, verify
from paper_transfer_diagnose import DEST, NAMES


def main():
    verify()
    config_path = ART / 'configs/pendulum.json'
    config = read(config_path)
    model = config['plant']['model']
    assert model['states']['omega']['rhs'] == '(M*g*np.sin(theta)-np.cos(theta)*(u1+m*l*omega**2*np.sin(theta)))/((4/3)*M*l-m*l*np.cos(theta)**2)'
    params = model['parameters']
    M, m, length, g = [params[k] for k in ['M', 'm', 'l', 'g']]
    bounds = {(c['name'], c['constraint_type']): c['value']
              for c in config['mpc']['constraints']}
    umax = bounds['u1', 'upper']
    assert bounds['u1', 'lower'] == -umax
    assert bounds['theta', 'upper'] == math.pi / 2
    assert bounds['theta', 'lower'] == -math.pi / 2
    assert (4 / 3) * M > m and length > 0
    threshold = math.atan(umax / (M * g))

    def certificate(state):
        theta, omega = state['theta'], state['omega']
        return threshold < abs(theta) < math.pi / 2 and theta * omega >= 0

    def min_outward_acceleration_at_zero_speed(theta):
        a = abs(theta)
        return (M * g * math.sin(a) - umax * math.cos(a)) / (
            (4 / 3) * M * length - m * length * math.cos(a)**2)

    bank_path = OUT / 'validation_bank.json'
    bank = read(bank_path)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [config_path, bank_path]}
    rows = []
    for domain in DOMAINS:
        for scene in bank['domains'][domain]:
            for name in NAMES:
                path = OUT / 'evaluation/validation' / domain / name / ('scene_%02d.json' % scene['id'])
                data = read(path)
                hashes[str(path.relative_to(ROOT))] = sha(path)
                trace = data['trace']
                # checked_step stores the pre-control mpc_state in alphabetic state order.
                pre = dict(zip(['omega', 'pos', 'theta', 'v'], trace[0]['terminal_input'][:4]))
                scaled = [pre['pos'] / 1.5, pre['v'] / 5, pre['theta'] / (math.pi / 2), pre['omega'] / 10]
                assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(scaled, data['initial'][:4]))
                max_input = max(abs(r['input']['u1']) for r in trace)
                certified = certificate(pre)
                if certified:
                    assert min_outward_acceleration_at_zero_speed(pre['theta']) > 0
                rows.append({'domain': domain, 'scene': scene['id'], 'name': name,
                    'bank_state': scene['case']['state'], 'post_warmup_state': pre,
                    'bank_certificate': certificate(scene['case']['state']),
                    'post_warmup_certificate': certified,
                    'min_outward_acceleration_at_zero_speed': min_outward_acceleration_at_zero_speed(pre['theta']),
                    'max_abs_input': max_input, 'input_excess_steps_gt_1e_5': sum(abs(r['input']['u1']) > umax + 1e-5 for r in trace),
                    'physical_failure': data['summary']['physical_failure'],
                    'steps': len(trace), 'first_solver_failure_step': next((i + 1 for i, r in enumerate(trace) if not r['solver_success']), None),
                    'last_state': trace[-1]['state']})
    certified_rows = [r for r in rows if r['post_warmup_certificate']]
    extra_failures = [r for r in rows if r['physical_failure'] and not r['post_warmup_certificate']]
    result = {'scope': 'Post-hoc descriptive validation diagnosis of the same seven policies; all original cases retained. No holdout trajectory access.',
        'threshold_rad': threshold, 'threshold_deg': math.degrees(threshold),
        'certificate': 'Within |theta| < pi/2, if |theta| > atan(umax/(M*g)) and theta*omega >= 0, the omega=0 boundary points outward for every |u|<=umax. The state cannot return to theta=0 without first leaving this angular safety strip.',
        'proof_boundary': 'sign(theta)*omega_dot at omega=0 is bounded below by (M*g*sin(abs(theta))-umax*cos(abs(theta)))/((4/3)*M*l-m*l*cos(theta)^2), which is strictly positive above the threshold. The denominator is positive. Continuous dynamics therefore cannot cross from outward to inward angular velocity in this region.',
        'limits': ['Sufficient condition only; an uncertified state is not certified recoverable.',
                   'No assertion about unconstrained full-rotation swing-up or the exact time of termination.',
                   'The continuous-model certificate assumes bounded actuation. Stored controlled steps are checked at tolerance 1e-5; reset warmup input is not recorded here. The post-warmup state is checked separately.',
                   'Solver failure is not itself a certificate. This calculation does not attribute other failures to a unique cause.',
                   'Only four validation scenes per domain; repeated policies/domains do not create independent initial states.'],
        'episodes_checked': len(rows), 'certified_episode_conditions': len(certified_rows),
        'all_certified_conditions_physically_failed': all(r['physical_failure'] for r in certified_rows),
        'input_excess_steps_gt_1e_5': sum(r['input_excess_steps_gt_1e_5'] for r in rows),
        'max_abs_input_all_checked': max(r['max_abs_input'] for r in rows),
        'uncertified_failure_conditions': extra_failures, 'rows': rows, 'input_sha256': hashes}
    write(DEST / 'feasibility.json', result)
    lines = ['# 迁移验证集：受限输入下的角度可恢复性补充', '',
        '这是对已有验证轨迹的事后诊断，没有新增仿真、训练或读取 holdout。分析范围与迁移诊断相同：5 个域、7 个策略、每域 4 个场景，共 140 个回合条件。', '',
        '配置为 M=0.8、m=0.2、l=0.25、g=9.81，控制力约束 |u|≤5，角度安全范围 |θ|<π/2。', '',
        '在角速度 ω=0 边界，最小向外角加速度为', '',
        '`[M g sin(|θ|) − 5 cos(|θ|)] / [(4/3) M l − m l cos²(θ)]`。', '',
        '因此，当 %.6f rad（%.4f°）<|θ|<π/2 且 θω≥0 时，所有允许输入都使该边界指向外侧；连续模型不能在保持角度安全的同时回到直立。它是充分条件，不是完整可行域判定，也不证明某个有限步数内必然终止。' % (threshold, math.degrees(threshold)), '',
        '## 轨迹核对', '',
        '宽初始分布的 scene 3（0 起编号）满足该条件，重置预热后的实际状态也满足。三个 broad 域中，所检查的七个策略均在第 9 步约束终止，共 21 个回合条件。它们来自同一个配对初始状态，不是 21 个独立样本。', '',
        '140 个回合的最大 |u| 为 %.12g；超过 5+1e-5 的控制步数为 %d。预热输入没有保存在这些文件中，故从单独核验的预热后状态应用证书。证书不替代求解器诊断，也不能证明未满足条件的状态可恢复。' % (result['max_abs_input_all_checked'], result['input_excess_steps_gt_1e_5']), '',
        '## 未被该证书解释的额外失败', '',
        '|域|策略|场景|终止步|首次求解失败步|', '|---|---|---:|---:|---:|']
    for r in extra_failures:
        lines.append('|%s|%s|%d|%d|%s|' % (r['domain'], r['name'], r['scene'], r['steps'], r['first_solver_failure_step']))
    lines += ['', '这些额外失败仍需与策略、时域决策及数值求解行为一起研究；不能仅凭此证书确定原因。所有失败场景保留在主结果和成本均值中，不删除共同失败来重算一个更有利的主指标。', '',
        '原验证诊断见 [迁移诊断](bohn2021_transfer_diagnosis_2026-09-24.md)；逐条件状态、输入、源文件散列见 `results/transfer_diagnosis_2026-09-24/feasibility.json`。']
    report = ROOT / 'docs/reports/bohn2021_transfer_feasibility_2026-09-24.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print({'report': str(report), 'episodes': len(rows), 'certified_conditions': len(certified_rows),
           'uncertified_failures': len(extra_failures), 'input_excess_steps': result['input_excess_steps_gt_1e_5']})


if __name__ == '__main__':
    main()
