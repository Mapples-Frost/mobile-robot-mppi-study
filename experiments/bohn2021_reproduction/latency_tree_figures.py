"""All-seed figure preparation, preview, and reviewed export after serial work.

Commands intentionally separate machine layout checks from actual visual review.
No code here computes an acceptance gate or unlocks a sealed test set.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from latency_tree_figure_data import (
    OUT, TASKS, collect, collect_training, check, csv_write, frozen_json,
    frozen_text, idle, read, register, sha, sources, verify,
)

COLORS = ('#0072B2', '#D55E00', '#009E73')
MARKERS = ('o', 's', '^')
VISUAL_ITEMS = ('glyphs', 'clipping', 'overlap', 'panel_alignment',
                'panel_spacing', 'color_and_grayscale', 'all_data_visible', 'cross_panel_consistency')
SKILL_FILES = ('profile_data.py', 'setup_style.py', 'visual_qa.py',
               'layout_tools.py', 'export_figure.py', 'check_figure.py')


def destination(split):
    return OUT / (split + '_figures')


def run_logged(command, path):
    value = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           encoding='utf-8', errors='replace')
    frozen_text(path, value.stdout)
    frozen_text(Path(str(path) + '.stderr'), value.stderr)
    assert value.returncode == 0, ('Helper failed', command, value.returncode, str(path))


def skill_hashes(folder):
    return {str(folder / name): sha(folder / name) for name in SKILL_FILES}


def prepare(split, skill):
    checks = read(OUT / 'figure_checks.json')
    assert checks['passed'] and checks['source_hashes'] == sources()
    data, training = collect(split), collect_training()
    dest = destination(split)
    dest.mkdir(exist_ok=True)
    for name in ('comparisons', 'pairs', 'episodes', 'timing'):
        csv_write(dest / (name + '.csv'), data[name])
    csv_write(dest / 'training_candidates.csv', training['candidates'])
    frozen_json(dest / 'figure_data.json', data)
    frozen_json(dest / 'training_data.json', training)
    groups = {
        'comparisons': ('task', 'method', 'comparator'),
        'pairs': ('task', 'comparator', 'seed'),
        'episodes': ('task', 'method', 'seed', 'h'),
        'timing': ('task', 'method', 'seed', 'h', 'timing_repeat'),
        'training_candidates': ('task', 'seed', 'generation'),
    }
    profiles = []
    for name, columns in groups.items():
        command = [sys.executable, str(skill / 'profile_data.py'), str(dest / (name + '.csv'))]
        for column in columns:
            command += ['--group', column]
        profile = dest / (name + '_profile.md')
        run_logged(command, profile)
        profiles += [profile, Path(str(profile) + '.stderr')]
    files = [dest / 'figure_data.json', dest / 'training_data.json']
    files += [dest / (name + '.csv') for name in groups] + profiles
    manifest = dict(passed_preparation=True, split=split, n_cases=data['n_cases'],
                    effect_passed=data['effect_passed'], main_comparisons=12,
                    diagnostic_comparisons=21, training_candidates=288,
                    source_hashes=sources(), skill_hashes=skill_hashes(skill),
                    input_hashes=dict(data['hashes'], **training['hashes'],
                                      **{str(OUT / 'figure_checks.json'): sha(OUT / 'figure_checks.json')}),
                    output_hashes={str(p): sha(p) for p in files},
                    test_accessed=split == 'test', goal_complete=False,
                    profiling_review_pending=True, visual_review_pending=True)
    frozen_json(dest / 'data_manifest.json', manifest)
    frozen_json(dest / 'profile_review_template.json', dict(
        passed=False, data_manifest_sha256=sha(dest / 'data_manifest.json'), notes='',
        all_cases_and_seeds_retained=None, no_outlier_exclusions=None))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('hashes')}, indent=2))


def load_prepared(split, skill):
    idle()
    dest = destination(split)
    manifest = read(dest / 'data_manifest.json')
    assert manifest['passed_preparation'] and manifest['split'] == split
    for field in ('source_hashes', 'skill_hashes', 'input_hashes', 'output_hashes'):
        verify(manifest[field])
    assert manifest['source_hashes'] == sources() and manifest['skill_hashes'] == skill_hashes(skill)
    # A reviewer records what the profiles show before choosing the final plots.
    review = read(dest / 'profile_review.json')
    assert review['passed'] and review['data_manifest_sha256'] == sha(dest / 'data_manifest.json')
    assert isinstance(review['notes'], str) and review['notes'].strip()
    assert review['all_cases_and_seeds_retained'] and review['no_outlier_exclusions']
    return read(dest / 'figure_data.json'), read(dest / 'training_data.json')


def plot_style(skill):
    sys.path.insert(0, str(skill))
    import matplotlib
    matplotlib.use('Agg')
    from setup_style import setup_style
    setup_style(journal='general', lang='en', use_sciplots=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 9,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'figure.constrained_layout.use': True})
    return plt


def limits(values, include=()):
    values = [float(v) for v in list(values) + list(include) if v is not None]
    assert values and all(math.isfinite(v) for v in values)
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = .08 * span if span else max(.1, .08 * abs(lo))
    return lo - pad, hi + pad


def notes(data):
    return (f"{data['split'].capitalize()} | {data['n_cases']} paired scenes per task | "
            "all 3 fitted seeds | selected adaptive policy")


def style_axis(ax):
    ax.grid(axis='x', color='.88', linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def forest(plt, data, metric):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    rows = [r for r in data['comparisons'] if r['main_gate']]
    for i, task in enumerate(TASKS):
        taskrows = [r for r in rows if r['task'] == task]
        prefix = 'cost' if metric == 'cost' else 'time'
        values = [r[prefix + '_' + part + '_pct'] for r in taskrows for part in ('mean', 'lower', 'upper')]
        if metric == 'time':
            values += [r['time_repeat%d_pct' % repeat] for r in taskrows for repeat in range(2)]
        bound = limits(values, (0, -3, 2) if metric == 'cost' else (0, -10))
        for j, comparator in enumerate(('independent', 'matched')):
            ax = axes[i, j]
            for row in [r for r in taskrows if r['comparator'] == comparator]:
                seed = row['seed']
                mid, low, high = [row[prefix + '_' + part + '_pct'] for part in ('mean', 'lower', 'upper')]
                if mid is None:
                    ax.text(.5, seed, 'Undefined: fixed mean cost = 0', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=7)
                    continue
                # Draw endpoints directly: percentile CIs can exclude estimates.
                ax.hlines(seed, low, high, color=COLORS[seed], linewidth=1.4)
                ax.scatter([low, high], [seed, seed], marker='|', color=COLORS[seed], s=35)
                ax.scatter([mid], [seed], marker=MARKERS[seed], color=COLORS[seed], s=32, zorder=4)
                if metric == 'time':
                    ax.scatter([row['time_repeat0_pct'], row['time_repeat1_pct']],
                               [seed - .16, seed + .16], marker='x', color=COLORS[seed], s=22, zorder=3)
            ax.axvline(0, color='.3', linewidth=.7)
            ax.axvline(-3 if metric == 'cost' else -10, color='.3', linestyle='--', linewidth=.8)
            if metric == 'cost':
                ax.axvline(2, color='.55', linestyle=':', linewidth=.9)
            ax.set(xlim=bound, ylim=(-.5, 2.5), yticks=[0, 1, 2],
                   yticklabels=['Seed 0', 'Seed 1', 'Seed 2'],
                   title=task.capitalize() + ' / ' + comparator + ' fixed',
                   xlabel=('Mean cost change (% of |fixed mean|)' if metric == 'cost'
                           else 'Decision latency change (%)'))
            ax.invert_yaxis()
            style_axis(ax)
    fig.suptitle(notes(data), fontsize=9)
    return fig


def paired(plt, data, task):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.6))
    rows = [r for r in data['pairs'] if r['task'] == task]
    bound = limits([r['cost_difference'] for r in rows], (0,))
    for seed in range(3):
        for j, comparator in enumerate(('independent', 'matched')):
            selected = [r for r in rows if r['seed'] == seed and r['comparator'] == comparator]
            assert [r['case'] for r in selected] == list(range(data['n_cases']))
            ax = axes[seed, j]
            ax.scatter([r['case'] for r in selected], [r['cost_difference'] for r in selected],
                       color=COLORS[seed], marker=MARKERS[seed], s=12, alpha=.8)
            ax.axhline(0, color='.4', linewidth=.7)
            ax.set(xlabel='Scene index (not elapsed time)', ylabel='Cost difference (adaptive - fixed)',
                   ylim=bound, title='Seed %d / %s fixed' % (seed, comparator))
            style_axis(ax)
    fig.suptitle(task.capitalize() + ' | all paired scenes | ' + data['split'], fontsize=10)
    return fig


def physical_cost(plt, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, task in zip(axes, TASKS):
        rows = sorted([r for r in data['comparisons'] if r['main_gate'] and r['task'] == task],
                      key=lambda r: (r['comparator'], r['seed']))
        labels = [('Independent' if r['comparator'] == 'independent' else 'Matched') + ' / s%d' % r['seed'] for r in rows]
        for y, row in enumerate(rows):
            value = row['physical_cost_change_pct']
            if value is None:
                ax.text(.5, y, 'Undefined: zero reference mean', transform=ax.get_yaxis_transform(), ha='center', fontsize=7)
            else:
                ax.scatter([value], [y], color=COLORS[row['seed']], marker=MARKERS[row['seed']], s=28)
        ax.axvline(0, color='.4', linewidth=.7)
        ax.axvline(2, color='.4', linewidth=.8, linestyle=':')
        ax.set(yticks=range(6), yticklabels=labels, ylim=(5.6, -.6), title=task.capitalize(),
               xlabel='Physical + constraint cost change (%)',
               xlim=limits([r['physical_cost_change_pct'] for r in rows], (0, 2)))
        style_axis(ax)
    fig.suptitle('Physical + constraint cost | all fitted seeds | +2% noninferiority boundary', fontsize=9)
    return fig


def safety(plt, data, task):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fields = [('success_loss_pp', 'Success loss (percentage points)'),
              ('constraint_increase_pp', 'Constraint-episode increase (pp)'),
              ('initial_failure_increase_pp', 'Initial failed-step increase (pp)'),
              ('final_failure_increase_pp', 'Final failed-step increase (pp)')]
    rows = sorted([r for r in data['comparisons'] if r['main_gate'] and r['task'] == task],
                  key=lambda r: (r['comparator'], r['seed']))
    labels = [('Independent' if r['comparator'] == 'independent' else 'Matched') + ' / s%d' % r['seed'] for r in rows]
    for ax, (field, title) in zip(axes.flat, fields):
        for y, row in enumerate(rows):
            ax.scatter([row[field]], [y], color=COLORS[row['seed']], marker=MARKERS[row['seed']], s=28)
        ax.axvline(0, color='.3', linewidth=.8)
        ax.set(yticks=range(6), yticklabels=labels, ylim=(5.6, -.6), xlabel=title,
               xlim=limits([r[field] for r in rows], (0,)))
        style_axis(ax)
    fig.suptitle(task.capitalize() + ' | safety differences | positive means worse', fontsize=10)
    return fig


def adaptation(plt, data, metric):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    rows = [r for r in data['episodes'] if r['method'] == 'adaptive']
    for i, task in enumerate(TASKS):
        taskrows = [r for r in rows if r['task'] == task]
        bound = (0, max(1, max(r['switches'] for r in taskrows)) * 1.08) if metric == 'switches' else (4, 51)
        for seed in range(3):
            ax = axes[i, seed]
            selected = [r for r in taskrows if r['seed'] == seed]
            assert len(selected) == data['n_cases']
            ax.scatter([r['case'] for r in selected], [r[metric] for r in selected],
                       color=COLORS[seed], marker=MARKERS[seed], s=11, alpha=.8, clip_on=False)
            ax.set(title=task.capitalize() + ' / seed %d' % seed,
                   xlabel='Scene index', ylabel='Within-episode H switches' if metric == 'switches' else 'Mean horizon (steps)',
                   ylim=bound)
            if metric == 'switches':
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            style_axis(ax)
    fig.suptitle(notes(data), fontsize=9)
    return fig


def diagnostics(plt, data, method):
    rows = sorted([r for r in data['comparisons'] if r['method'] == method],
                  key=lambda r: (r['comparator'], r['seed']))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.0 if method == 'combined' else 4.9))
    labels = [r['comparator'].capitalize() + ' / s%d' % r['seed'] for r in rows]
    for ax, prefix in zip(axes, ('cost', 'time')):
        for y, row in enumerate(rows):
            mid, low, high = [row[prefix + '_' + part + '_pct'] for part in ('mean', 'lower', 'upper')]
            if mid is None:
                ax.text(.5, y, 'Undefined: zero fixed mean', transform=ax.get_yaxis_transform(), ha='center', fontsize=7)
                continue
            ax.hlines(y, low, high, color=COLORS[row['seed']], linewidth=1.2)
            ax.scatter([mid], [y], color=COLORS[row['seed']], marker=MARKERS[row['seed']], s=25)
        bound = limits([r[prefix + '_' + part + '_pct'] for r in rows for part in ('mean', 'lower', 'upper')], (0,))
        ax.axvline(0, color='.4', linewidth=.7)
        ax.set(yticks=range(len(rows)), yticklabels=labels, ylim=(len(rows) - .4, -.6), xlim=bound,
               xlabel='Cost change (% of |reference mean|)' if prefix == 'cost' else 'Decision latency change (%)')
        style_axis(ax)
    fig.suptitle(('C5 rule' if method == 'certificate' else 'Adaptive + C5') +
                 ' | pendulum diagnostics | excluded from main gate', fontsize=10)
    return fig


def training_plot(plt, training):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    from matplotlib.lines import Line2D
    for i, task in enumerate(TASKS):
        allrows = [r for r in training['candidates'] if r['task'] == task]
        xbound = limits([100 * r['cost_change'] for r in allrows], (0,))
        ybound = limits([100 * (r['time_ratio'] - 1) for r in allrows], (0,))
        for seed in range(3):
            ax = axes[i, seed]
            rows = [r for r in allrows if r['seed'] == seed]
            assert len(rows) == 48
            for eligible, marker, color in ((True, 'o', COLORS[seed]), (False, 'x', '.5')):
                selected = [r for r in rows if r['eligible'] == eligible]
                ax.scatter([100 * r['cost_change'] for r in selected],
                           [100 * (r['time_ratio'] - 1) for r in selected],
                           marker=marker, color=color, s=18, alpha=.7)
            selected = [r for r in rows if r['selected']]
            ax.scatter([100 * r['cost_change'] for r in selected],
                       [100 * (r['time_ratio'] - 1) for r in selected],
                       marker='s', facecolors='none', edgecolors='black', s=65, linewidth=1.)
            policy = next(r for r in training['policies'] if (r['task'], r['seed']) == (task, seed))
            ax.set(title='%s s%d / %s' % (task.capitalize(), seed, policy['selected']),
                   xlabel='Training cost change (%)', ylabel='Training latency change (%)',
                   xlim=xbound, ylim=ybound)
            ax.axhline(0, color='.6', linewidth=.6)
            ax.axvline(0, color='.6', linewidth=.6)
            style_axis(ax)
    handles = [Line2D([], [], marker='o', color='.2', linestyle='none', label='Eligible'),
               Line2D([], [], marker='x', color='.5', linestyle='none', label='Ineligible'),
               Line2D([], [], marker='s', markerfacecolor='none', color='black', linestyle='none', label='Selected')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('All 288 training candidates | selected in-sample | no independent efficacy claim', fontsize=9)
    return fig


def fixed_grid(plt, data, task):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    base = 25 if task == 'vehicle' else 30
    raw = [r for r in data['episodes'] if r['task'] == task]
    by_arm = {}
    for row in raw:
        key = row['method'], row['seed'], row['h']
        by_arm.setdefault(key, []).append(row)
    points = []
    for label, family, seeds in (('independent', 'grid', (0,)), ('matched', 'matched', (0, 1, 2))):
        for seed in seeds:
            for h in range(5, 51, 5):
                key = ('primary' if h == base else family), seed, h
                selected = by_arm[key]
                assert len(selected) == data['n_cases']
                points.append(dict(label=label, seed=seed, h=h,
                                   cost=math.fsum(r['total_cost'] for r in selected) / len(selected)))
    bound = limits([r['cost'] for r in points], (0,))
    for ax, label in zip(axes, ('independent', 'matched')):
        rows = [r for r in points if r['label'] == label]
        for seed in sorted({r['seed'] for r in rows}):
            selected = [r for r in rows if r['seed'] == seed]
            ax.scatter([r['h'] for r in selected], [r['cost'] for r in selected],
                       marker=MARKERS[seed], color=COLORS[seed], s=26, label='Seed %d' % seed)
        h = data['nominations'][task][label + '_h']
        ax.axvline(h, color='.45', linestyle=':', linewidth=.8)
        ax.set(title=label.capitalize() + ' fixed / nominated H=%d' % h,
               xlabel='Fixed prediction horizon H (steps)', ylabel='Mean total cost (raw objective)',
               ylim=bound, xticks=list(range(5, 51, 5)))
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=3, frameon=False)
        style_axis(ax)
    fig.suptitle(task.capitalize() + ' | complete validation H grids | descriptive selection data', fontsize=9)
    return fig


def definitions(data):
    n = data['n_cases']
    common = (f'每任务全部{n}个配对场景、三个已拟合种子。区间为原登记10,000次配对场景bootstrap的95%百分位区间，'
              '条件化于这些模型；不把训练种子、控制步或两次计时当新增独立样本。未额外计算p值或添加显著性星号。')
    result = {
        'cost_intervals': common + 'adaptive包括预先允许的固定回退，不能把回退记作成功学出自适应。'
                          '区间为绝对成本差区间除以观测固定平均成本绝对值；分母固定，不是重新bootstrap的比值。'
                          '虚线为-3%改善门槛，点线为+2%非劣界。零分母标未定义。图中阈值不能替代全部门槛。',
        'latency_intervals': common + '延迟为总决策时间除以总控制步；两次串行重复均计入。叉号为两次单独重复，虚线为-10%。'
                             '区间为延迟比值的配对bootstrap结果转换为百分比变化。',
        'physical_cost': '所有主比较的物理成本加约束成本平均值相对变化，点线为+2%非劣门槛。无置信区间；零分母标未定义并保留原值。',
        'vehicle_paired_costs': '车辆全部场景成本差；每点为一个场景，横轴只是场景编号。无剔除、连线或局部缩放。',
        'pendulum_paired_costs': '倒立摆全部场景成本差；每点为一个场景，横轴只是场景编号。无剔除、连线或局部缩放。',
        'vehicle_safety': '车辆成功损失、约束回合增加、初始/最终失败步率增加；均以百分点计。正数表示劣化。无置信区间；精确计数在CSV。',
        'pendulum_safety': '倒立摆成功损失、约束回合增加、初始/最终失败步率增加；均以百分点计。正数表示劣化。无置信区间；精确计数在CSV。',
        'horizon_switches': '每个场景的回合内H切换次数。每步选择和跨回合选择须区分，零值保留。此图不代表加速。',
        'mean_horizon': '每个场景实际执行控制步的平均H，全部失败场景保留。平均H是代理量，不能作为实测延迟证据。',
        'c5_diagnostics': common + '手工C5相对学习树和两固定比较器的全部9项诊断比较，不计入主学习效果。',
        'combined_diagnostics': common + '学习树+C5相对树、C5和两固定比较器的全部12项诊断比较，不计入主学习效果。',
        'training_candidates': '两任务×三种子×四代×12候选共288项全部保留（重复点可能重叠）。各代独立固定参考；'
                               '圈/叉表示训练资格，空心方框标最终候选，固定回退在标题标明。训练内选择偏倚，不能作为独立验证或加速结论。',
    }
    if data['split'] == 'validation':
        for task in TASKS:
            result[task + '_fixed_grid'] = ('全部H5/10/…/50固定网格；独立终端完整网格采用seed0，共用终端完整网格采用三个种子。'
                                          '每点为全部64场景均值，无连线或置信区间；点线标按已登记安全规则提名的H。'
                                          '补充seed1/2独立终端仅评估提名H，不能声称它们也完成独立全网格。')
    return result


def figures(plt, data, training):
    yield 'cost_intervals', forest(plt, data, 'cost')
    yield 'latency_intervals', forest(plt, data, 'time')
    yield 'physical_cost', physical_cost(plt, data)
    for task in TASKS:
        yield task + '_paired_costs', paired(plt, data, task)
        yield task + '_safety', safety(plt, data, task)
        if data['split'] == 'validation':
            yield task + '_fixed_grid', fixed_grid(plt, data, task)
    yield 'horizon_switches', adaptation(plt, data, 'switches')
    yield 'mean_horizon', adaptation(plt, data, 'mean_horizon')
    yield 'c5_diagnostics', diagnostics(plt, data, 'certificate')
    yield 'combined_diagnostics', diagnostics(plt, data, 'combined')
    yield 'training_candidates', training_plot(plt, training)


def render(split, skill, revision, export=False):
    data, training = load_prepared(split, skill)
    dest = destination(split)
    folder = dest / ('preview_%02d' % revision)
    assert revision > 0
    review = None
    if export:
        preview = read(folder / 'preview_manifest.json')
        review = read(folder / 'visual_review.json')
        assert review['passed'] and review['preview_manifest_sha256'] == sha(folder / 'preview_manifest.json')
        assert review['reviewer'] and review['notes']
        assert set(review['figures']) == set(preview['figures'])
        for row in review['figures'].values():
            assert all(row[k] is True for k in VISUAL_ITEMS)
        verify(preview['hashes'])
        assert preview['source_hashes'] == sources()
        assert preview['data_manifest_sha256'] == sha(dest / 'data_manifest.json')
        assert preview['profile_review_sha256'] == sha(dest / 'profile_review.json')
        assert preview['skill_hashes'] == skill_hashes(skill)
        target = dest / ('final_%02d' % revision)
        assert not target.exists(), 'Inspect existing final export before retrying'
    else:
        target = folder
        assert not folder.exists(), 'Use a new preview revision, preserve old QA history'
    target.mkdir()
    plt = plot_style(skill)
    from layout_tools import finalize_figure, add_panel_labels
    from visual_qa import audit_layout
    from export_figure import export_figure
    from PIL import Image
    entries, hashes = {}, {}
    for name, fig in figures(plt, data, training):
        finalize_figure(fig, verbose=False)
        add_panel_labels(fig, style='nature', x_offset_pt=-13, y_offset_pt=5)
        issues = audit_layout(fig)
        assert not any(level == 'FAIL' for level, _ in issues), (name, issues)
        size = [float(v) for v in fig.get_size_inches()]
        if export:
            # The reviewed preview and export come from exactly the same source/data.
            paths = export_figure(fig, str(target / name), formats=['pdf', 'svg', 'png'],
                                  size_inches=tuple(size), dpi=300, tight=False, grayscale_preview=False)
            gray = target / (name + '_grayscale.png')
            with Image.open(str(target / (name + '.png'))) as color:
                color.convert('L').save(str(gray), dpi=(300, 300))
            paths.append(str(gray))
            run_logged([sys.executable, str(skill / 'check_figure.py'), *paths, '--min-dpi', '300',
                        '--width-in', str(size[0]), '--height-in', str(size[1]), '--strict'],
                       target / (name + '_compliance.txt'))
            for suffix in ('_compliance.txt', '_compliance.txt.stderr'):
                p = target / (name + suffix)
                hashes[str(p)] = sha(p)
        else:
            color, gray = target / (name + '.png'), target / (name + '_grayscale.png')
            fig.savefig(str(color), dpi=150, bbox_inches=None)
            with Image.open(str(color)) as image:
                image.convert('L').save(str(gray), dpi=(150, 150))
            paths = [str(color), str(gray)]
        entries[name] = dict(size_inches=size, layout_issues=issues,
                             caption=definitions(data)[name], files=paths)
        hashes.update({str(p): sha(p) for p in map(Path, paths)})
        plt.close(fig)
    caption = '\n\n'.join('## ' + name + '\n\n' + value for name, value in definitions(data).items()) + '\n'
    frozen_text(target / 'captions_CN.md', caption)
    hashes[str(target / 'captions_CN.md')] = sha(target / 'captions_CN.md')
    record = dict(split=split, revision=revision, figures=entries, hashes=hashes,
                  source_hashes=sources(), skill_hashes=skill_hashes(skill),
                  data_manifest_sha256=sha(dest / 'data_manifest.json'),
                  profile_review_sha256=sha(dest / 'profile_review.json'),
                  visual_review_passed=export, goal_complete=False,
                  effect_passed=data['effect_passed'], test_accessed=split == 'test')
    if export:
        record['visual_review_sha256'] = sha(folder / 'visual_review.json')
        frozen_json(target / 'export_manifest.json', record)
    else:
        frozen_json(folder / 'preview_manifest.json', record)
        template = dict(passed=False, preview_manifest_sha256=sha(folder / 'preview_manifest.json'),
                        reviewer='', notes='', figures={name: {k: None for k in VISUAL_ITEMS} for name in entries})
        frozen_json(folder / 'visual_review_template.json', template)
    print(json.dumps(dict(figures=len(entries), folder=str(target), exported=export,
                          actual_visual_review_pending=not export, goal_complete=False), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('register', 'check', 'prepare', 'preview', 'export'), required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--skill-scripts', type=Path,
                        default=Path('/mnt/c/Users/lenovo/.codex/skills/scipilot-figure-skill/scripts'))
    args = parser.parse_args()
    if args.mode == 'register':
        print(json.dumps(register(), indent=2))
    elif args.mode == 'check':
        result = check()
        result['source_hashes'] = sources()
        frozen_json(OUT / 'figure_checks.json', result)
        print(json.dumps(result, indent=2))
    elif args.mode == 'prepare':
        prepare(args.split, args.skill_scripts)
    else:
        render(args.split, args.skill_scripts, args.revision, args.mode == 'export')


if __name__ == '__main__':
    main()
