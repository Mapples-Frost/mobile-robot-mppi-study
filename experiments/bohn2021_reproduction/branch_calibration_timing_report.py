"""Independently check timing traces against efficacy traces and summarize all seeds."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24'
OUT = BASE/'serial_timing'
ARMS = ('actor', 'raw_greedy', 'calibrated_greedy', 'fixed')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(p):
    d = read(p)
    assert d.get('passed', True)
    for name, h in d['hashes'].items(): assert sha(Path(name)) == h, name
    return d


def main():
    checked(OUT/'registration.json')
    complete = checked(OUT/'completed.json')
    assert complete['conditions'] == 48
    seed_rows, episode_rows, repeat_rows, gathered, conditions = [], [], [], {}, set()
    for path in sorted(OUT.glob('r*/completed.json')):
        checked(path)
        summary = read(path.parent/'summary.json')
        t, s, a, rep = [summary[k] for k in ('task','seed','arm','repeat')]
        assert (t,s,a,rep) not in conditions
        conditions.add((t,s,a,rep))
        key = (t,s,a)
        data = gathered.setdefault(key, {'latencies': [], 'policy': [], 'controller': [], 'costs': [], 'resets': [], 'episodes': [], 'repeat_means': []})
        condition_latencies = []
        reference = BASE/'evaluations/validation'/t/('%s_s%d' % (a,s))
        assert len(summary['episodes']) == 10
        assert {e['case'] for e in summary['episodes']} == set(range(10))
        for e in summary['episodes']:
            j = e['case']
            trace = read(path.parent/('trace_%02d.json' % j))
            old = read(reference/('trace_%02d.json' % j))
            assert len(trace) == len(old) == e['steps']
            for row, ref in zip(trace, old):
                assert {k:v for k,v in row.items() if k != 'timing'} == {k:v for k,v in ref.items() if k != 'elapsed_s'}
                times = row['timing']
                assert all(np.isfinite(v) and v >= 0 for v in times.values())
                assert times['environment_s'] >= times['controller_s'] > 0
                np.testing.assert_allclose(times['decision_s'], times['selection_s'] + times['controller_s'], atol=1e-12)
                data['latencies'].append(times['decision_s'])
                data['policy'].append(times['selection_s'])
                data['controller'].append(times['controller_s'])
            values = np.array([r['timing']['decision_s'] for r in trace])
            condition_latencies.extend(values.tolist())
            assert e['termination'] == trace[-1]['termination']
            assert e['solver_failures'] == sum(not r['solver_success'] for r in trace)
            assert e['deadline_exceed_steps'] == sum(values > (.1 if t=='vehicle' else .04))
            np.testing.assert_allclose(e['total_cost'], sum(-r['reward'] for r in trace), atol=1e-7)
            for name, value in [('decision_total_s', values.sum()), ('decision_mean_s', values.mean()),
                                ('decision_median_s', np.median(values)), ('decision_p95_s', np.percentile(values,95))]:
                np.testing.assert_allclose(e[name], value, atol=1e-12)
            data['costs'].append(e['total_cost']); data['resets'].append(e['reset_s']); data['episodes'].append(e)
            episode_rows.append(dict(task=t, seed=s, arm=a, repeat=rep, **e))
        mean_ms = float(1000*np.mean(condition_latencies))
        data['repeat_means'].append(mean_ms)
        repeat_rows.append({'task':t,'seed':s,'arm':a,'repeat':rep,'steps':len(condition_latencies),
            'mean_decision_ms':mean_ms,'p95_decision_ms':float(1000*np.percentile(condition_latencies,95))})
    assert len(gathered) == 24
    assert conditions == {(t,s,a,r) for t in ('vehicle','pendulum') for s in range(3) for a in ARMS for r in range(2)}
    for (t,s,a), d in sorted(gathered.items()):
        x = np.array(d['latencies'])
        assert len(d['episodes']) == 20
        seed_rows.append({'task':t,'seed':s,'arm':a,'timing_repetitions':2,'unique_scenes':10,
            'steps_including_repeats':len(x),'mean_decision_ms':float(1000*x.mean()),
            'median_decision_ms':float(1000*np.median(x)), 'p95_decision_ms':float(1000*np.percentile(x,95)),
            'mean_policy_ms':float(1000*np.mean(d['policy'])), 'mean_controller_ms':float(1000*np.mean(d['controller'])),
            'mean_episode_decision_s':float(x.sum()/20), 'mean_reset_s':float(np.mean(d['resets'])),
            'repeat_mean_decision_ms_min':min(d['repeat_means']), 'repeat_mean_decision_ms_max':max(d['repeat_means']),
            'repeat_mean_max_min_ratio':max(d['repeat_means'])/min(d['repeat_means']),
            'deadline_exceed_fraction':float(np.mean(x>(.1 if t=='vehicle' else .04))),
            'mean_control_cost':float(np.mean(d['costs'])),
            'successes_unique_scenes':sum(e['termination']==('goal' if t=='vehicle' else 'steps') for e in d['episodes'])//2,
            'solver_failed_steps_per_repeat':sum(e['solver_failures'] for e in d['episodes'])//2})
    lookup = {(r['task'],r['seed'],r['arm']):r for r in seed_rows}
    for r in seed_rows:
        fixed = lookup[r['task'],r['seed'],'fixed']
        r['mean_decision_time_ratio_vs_fixed'] = r['mean_decision_ms']/fixed['mean_decision_ms']
        r['mean_episode_time_ratio_vs_fixed'] = r['mean_episode_decision_s']/fixed['mean_episode_decision_s']
    episode_lookup = {(e['task'],e['seed'],e['arm'],e['repeat'],e['case']):e for e in episode_rows}
    assert len(episode_lookup) == 480
    paired_rows = []
    for e in episode_rows:
        if e['arm'] == 'fixed': continue
        fixed = episode_lookup[e['task'],e['seed'],'fixed',e['repeat'],e['case']]
        paired_rows.append({'task':e['task'],'seed':e['seed'],'arm':e['arm'],'repeat':e['repeat'],'case':e['case'],
            'decision_total_ratio_vs_fixed':e['decision_total_s']/fixed['decision_total_s'],
            'decision_mean_ratio_vs_fixed':e['decision_mean_s']/fixed['decision_mean_s'],
            'cost_difference_vs_fixed':e['total_cost']-fixed['total_cost'],
            'termination':e['termination'],'fixed_termination':fixed['termination'],
            'steps':e['steps'],'fixed_steps':fixed['steps'],
            'solver_failed_steps':e['solver_failures'],'fixed_solver_failed_steps':fixed['solver_failures']})
    for name, rows in [('seed_timings.csv',seed_rows),('repeat_timings.csv',repeat_rows),
                       ('episode_timings.csv',episode_rows),('paired_episode_timings.csv',paired_rows)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    colors=('#0072B2','#E69F00','#CC79A7','#009E73')
    labels=('Min-Q actor','Raw Q greedy','Calibrated Q','Fixed H')
    for ax,t in zip(axs,('vehicle','pendulum')):
        for a,c,label in zip(ARMS,colors,labels):
            for s,marker in enumerate(('o','s','^')):
                r=lookup[t,s,a]
                ax.errorbar(r['mean_decision_ms'],r['mean_control_cost'],
                    xerr=[[r['mean_decision_ms']-r['repeat_mean_decision_ms_min']],
                          [r['repeat_mean_decision_ms_max']-r['mean_decision_ms']]],
                    fmt='none',ecolor=c,alpha=.4,linewidth=1,capsize=2)
                ax.scatter(r['mean_decision_ms'],r['mean_control_cost'],color=c,marker=marker,s=55,label=label if s==0 else None)
        ax.set_title(t.capitalize()); ax.set_xlabel('Measured mean decision time (ms)')
        ax.set_ylabel('Mean validation control cost')
        if t == 'vehicle':
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Validation tradeoff: circle=seed 0, square=1, triangle=2\nHorizontal bars: range of two timing means (not confidence intervals)')
    for ext in ('png','pdf'): fig.savefig(OUT/('measured_tradeoff.'+ext),dpi=200)
    plt.close(fig)
    lines=['# 串行实测计算代价', '',
        '完整 48 个条件（两次计时重复）逐轨迹复核通过。仅为验证集描述，不能称新的独立加速证据；重复计时不增加独立训练种子数。', '',
        '主耗时 = H 选择 + controller.get_action，包含求解与预测提取，不含物理仿真、奖励审计、磁盘写入或 reset。全部失败步保留；提前失败的短回合不能解释为成功完成任务更快。', '',
        'actor 是 min-Q 改进模型的未改动 actor；本表不含最初作者 Q1 actor 方法。场景配对的每步和每回合耗时比、终止原因、成本差及失败数见 paired_episode_timings.csv。', '',
        '|任务|种子|方法|均值 ms|p95 ms|策略 ms|周期超时率|总控制成本|成功/10|相对固定每步耗时|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        lines.append('|{task}|{seed}|{arm}|{mean_decision_ms:.3f}|{p95_decision_ms:.3f}|{mean_policy_ms:.3f}|{deadline_exceed_fraction:.2%}|{mean_control_cost:.3f}|{successes_unique_scenes}/10|{mean_decision_time_ratio_vs_fixed:.3f}|'.format(**r))
    lines += ['', '各条件两次计时均值的最大/最小比最高为 %.3f；仅两次重复不足以精确估计测量不确定性，几个百分点的均值差不应被解释为可靠加速。图中水平线是两次均值范围，不是置信区间。控制成本包含物理项、H 代理项与约束项。' % max(r['repeat_mean_max_min_ratio'] for r in seed_rows)]
    lines += ['', '完整逐步/逐场景原始耗时、不同分量、每回合耗时和全部种子 CSV 均保留。两次重复各自的均值及 p95 在 repeat_timings.csv；seed_timings.csv 另给重复均值范围及最大/最小比，不能将重复差异隐藏在汇总均值中。周期为车辆 100 ms、倒立摆 40 ms；这里只统计已定义控制计算段，不能将低超时率解释为整个机器人系统的实时保证。',
        '各方法沿各自闭环轨迹运行，所以时延差异也包含状态/求解难度差异；不是固定同一状态下 H 的因果时延曲线。WSL 与宿主调度未完全隔离。主性能结论仍服从效果协议的验证/独立测试门槛。']
    (OUT/'report_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write = {'passed':True,'conditions':48,'model_episode_repetitions':len(episode_rows),'unique_scenes_per_task':10,
        'exact_trace_audit':True,'rows':seed_rows,'input_hashes':{str(p):sha(p) for p in
            [*OUT.glob('r*/completed.json'),OUT/'registration.json',OUT/'completed.json',Path(__file__)]}}
    (OUT/'timing_audit.json').write_text(json.dumps(write,indent=2)+'\n')
    print({'timing_audit_passed':True,'conditions':48,'model_episode_repetitions':len(episode_rows)})


if __name__=='__main__': main()
