"""Descriptive transfer diagnosis from the frozen VALIDATION split only.

No policy changes, simulator calls, training updates or holdout-based selection.
"""
import json
import statistics
from pathlib import Path
from paper_transfer_common import OUT, ROOT, DOMAINS, read, write, sha, verify

DEST = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/transfer_diagnosis_2026-09-24'
NAMES = ['fixed_25', 'fixed_30', 'rule_baseline', 'value_teacher',
         'value_s0', 'value_s1', 'value_s2']


def main():
    verify()
    bank = read(OUT / 'validation_bank.json')
    source_hashes = {}
    cache = {}
    cells = {}
    failures = []
    for domain in DOMAINS:
        cells[domain] = {}
        for name in NAMES:
            episodes = []
            for scene in bank['domains'][domain]:
                p = OUT / 'evaluation/validation' / domain / name / ('scene_%02d.json' % scene['id'])
                data = read(p)
                source_hashes[str(p.relative_to(ROOT))] = sha(p)
                trace = data['trace']
                cache[(domain, name, scene['id'])] = trace
                s = data['summary']
                assert len(trace) == s['steps']
                assert abs(sum(r['cost'] for r in trace) - s['raw_cost']) < 1e-6
                assert abs(s['physical_cost'] + s['H_cost'] + s['failure_penalty'] +
                           s['unexecuted_offset'] - s['adjusted_cost']) < 1e-6
                first_bad = next((i + 1 for i, r in enumerate(trace) if not r['solver_success']), None)
                item = dict(s, first_solver_failure_step=first_bad)
                episodes.append(item)
                if s['physical_failure'] or s['solver_failures']:
                    failures.append({'domain': domain, 'name': name, 'scene': scene['id'],
                                     'initial_state': scene['case']['state'],
                                     'post_warmup_obs': data['initial'][:4],
                                     'first_solver_failure_step': first_bad,
                                     'last_step': len(trace), 'termination': trace[-1]['termination'],
                                     'solver_failures': s['solver_failures'],
                                     'last_state': trace[-1]['state']})
            components = {key: statistics.mean(e[key] for e in episodes)
                          for key in ['adjusted_cost', 'physical_cost', 'H_cost',
                                      'failure_penalty', 'unexecuted_offset', 'mean_H']}
            cells[domain][name] = dict(components, physical_failures=sum(e['physical_failure'] for e in episodes),
                                      solver_failures=sum(e['solver_failures'] for e in episodes),
                                      steps=sum(e['steps'] for e in episodes), episodes=episodes)
    # Use the same H30 in every cell so baseline selection is not a factor.
    contrasts = []
    for name in NAMES[2:]:
        gaps = {d: cells[d][name]['adjusted_cost'] - cells[d]['fixed_30']['adjusted_cost'] for d in DOMAINS}
        contrasts.append({'name': name, 'cost_gap_to_fixed30': gaps,
                          'reference_effect_on_gap_near_600': gaps['redraw_near_600'] - gaps['plateau_near_600'],
                          'reference_effect_on_gap_broad_600': gaps['redraw_broad_600'] - gaps['plateau_broad_600'],
                          'initial_distribution_effect_on_gap_plateau': gaps['plateau_broad_600'] - gaps['plateau_near_600'],
                          'initial_distribution_effect_on_gap_redraw': gaps['redraw_broad_600'] - gaps['redraw_near_600']})
    prefix = []
    for name in NAMES:
        for scene in bank['domains']['redraw_broad_100']:
            short = cache[('redraw_broad_100', name, scene['id'])]
            long = cache[('redraw_broad_600', name, scene['id'])][:100]
            n = min(len(short), len(long))
            physical_delta = sum(r['performance'] + r['compute'] for r in short) - sum(r['performance'] + r['compute'] for r in long)
            prefix.append({'name': name, 'scene': scene['id'], 'short_steps': len(short), 'long_prefix_steps': len(long),
                           'H_disagreement_steps': sum(short[i]['horizon'] != long[i]['horizon'] for i in range(n)),
                           'physical_plus_H_cost_delta_short_minus_long_prefix': physical_delta,
                           'warning': 'Failure penalties excluded because they depend on duration; unequal survival is explicitly retained.'})
    out = {'scope': 'Frozen validation trajectories only; descriptive mechanism diagnosis, not independent confirmation.',
           'validation_scenes_per_domain': 4, 'new_training_updates': 0, 'new_simulator_steps': 0,
           'cells': cells, 'paired_factor_contrasts': contrasts, 'duration_prefix': prefix, 'failures': failures,
           'limitations': ['Four paired validation scenes; no significance claim.',
                          'Near-equilibrium initial positions follow the initial reference; reference contrast holds relative error, not absolute position, fixed.',
                          'Broad-state failures also affect fixed H; these traces do not prove initial states are uncontrollable.',
                          'Duration changes normalized time input and remaining-step failure cost; prefix comparison does not isolate either alone.',
                          'The original author reconstruction grid is unfinished at registration; finish it before proposing a new teacher-prior training intervention.'],
           'input_sha256': source_hashes}
    write(DEST / 'diagnosis.json', out)
    lines = ['# 论文任务迁移诊断：验证集证据', '',
             '仅使用冻结 validation 的每域 4 个场景，不读取 holdout 轨迹、不新增训练或仿真。以下是描述性诊断，不是新确认实验。', '',
             '|分布|方法|平均成本|物理项|H项|失败罚项|未执行常数项|物理失败|求解失败/步|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        for n in NAMES:
            c = cells[d][n]
            lines.append('|%s|%s|%.3f|%.3f|%.3f|%.3f|%.3f|%d/4|%d/%d|' %
                         (d, n, c['adjusted_cost'], c['physical_cost'], c['H_cost'], c['failure_penalty'],
                          c['unexecuted_offset'], c['physical_failures'], c['solver_failures'], c['steps']))
    lines += ['', '## 可确定的范围', '',
              '1. 频繁参考重绘带来独立于求解失败的性能退化：near/600 中全部策略均无失败，value 三种子在 plateau 优于固定 H30，在 redraw 则全部更差。冻结价值教师也退化；单纯保持旧教师不能保证迁移成功。',
              '2. 扩大初始状态范围增加失败：固定 H30 与规则也失败，value seed2 额外失败。失败罚项与未执行常数项单列，不把提前终止当作节省计算。不能由共同失败推断不可控。',
              '3. 100 步条件同时改变时间特征与终止罚项，不能用 100/600 总成本直接归因。JSON 保存相同初始状态和参考前缀的逐策略前 100 步比较；该比较仍是诊断。',
              '4. 只有四个验证场景；三训练种子才是学习重复单位。near 状态按初始参考平移，因此参考因素并非严格固定绝对位置的干预。', '',
              '## 后续顺序', '',
              '先收尾原作者 SAC + 联合终端学习的 paper_exact_grid 对照，完整呈现 H=5,10,…,50 与所有 RL 种子。迁移失败的混合教师扩展不能代替这条复现线。',
              '若随后研究扩展重训，需预登记新的训练分布及独立验证/测试种子，并将参考变化、初始状态与时间编码分开检验；不能依据当前 holdout 反复调参。']
    report = ROOT / 'docs/reports/bohn2021_transfer_diagnosis_2026-09-24.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(report), 'episodes_checked': len(source_hashes),
                      'validation_value_gaps': [c for c in contrasts if c['name'].startswith('value_s')],
                      'duration_prefix': prefix}, indent=2))


if __name__ == '__main__':
    main()
