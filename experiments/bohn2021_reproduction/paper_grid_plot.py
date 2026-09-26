"""Plot all audited grid outcomes; no selection, simulation or training."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/paper_exact_grid_2026-09-23'
REPORT = ART / 'report/paper_exact_grid_2026-09-23'


def main():
    audit = json.loads((OUT / 'audit_2026-09-24.json').read_text())
    assert audit['passed'] and audit['models'] == 26
    result_path = REPORT / 'results.json'
    data = json.loads(result_path.read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    colors = ['#D55E00', '#009E73', '#CC79A7']
    markers = ['^', 's', 'X']
    for i, task in enumerate(['pendulum', 'vehicle']):
        for j, mode in enumerate(['holdout_value', 'holdout_no_value']):
            ax = axes[i, j]
            rows = [r for r in data['rows'] if r['task'] == task and r['mode'] == mode]
            fixed = sorted([r for r in rows if r['fixed_horizon'] is not None], key=lambda r: r['fixed_horizon'])
            rl = sorted([r for r in rows if r['fixed_horizon'] is None], key=lambda r: r['seed'])
            assert [r['fixed_horizon'] for r in fixed] == list(range(5, 51, 5))
            assert [r['seed'] for r in rl] == [0, 1, 2]
            comp = next(c for c in data['comparisons'] if c['task'] == task and c['mode'] == mode)
            chosen = next(r for r in fixed if r['fixed_horizon'] == comp['validation_selected_H'])
            ax.plot([r['fixed_horizon'] for r in fixed], [r['mean_total_cost'] for r in fixed],
                    '-o', color='#0072B2', markersize=4.5, linewidth=1.7, zorder=2)
            ax.scatter(chosen['fixed_horizon'], chosen['mean_total_cost'], marker='D',
                       s=90, facecolors='none', edgecolors='#111111', linewidths=1.6, zorder=4)
            for r, color, marker in zip(rl, colors, markers):
                ax.scatter(r['mean_H'], r['mean_total_cost'], marker=marker, color=color,
                           s=65, edgecolors='white', linewidths=.7, zorder=5)
            ax.set_xlim(2.5, 52.5)
            ax.set_xticks(range(5, 51, 5))
            ax.set_title(('Pendulum' if task == 'pendulum' else 'Vehicle') +
                         (' | learned terminal value' if j == 0 else ' | terminal value zeroed'), loc='left', fontweight='bold')
            if task == 'vehicle':
                assert all(r['mean_total_cost'] > 0 for r in rows)
                ax.set_yscale('log')
                ax.set_ylabel('Mean episode cost (log scale)')
            else:
                ax.set_ylim(bottom=min(0, min(r['mean_total_cost'] for r in rows) * 1.1))
                ax.set_ylabel('Mean episode cost')
            ax.set_xlabel('Fixed H / episode-mean H for RL')
            ax.grid(axis='y', which='major', color='#D9DEE3', linewidth=.7)
            ax.set_axisbelow(True)
    handles = [Line2D([], [], color='#0072B2', marker='o', label='Fixed H, seed 0'),
               Line2D([], [], color='#111111', marker='D', markerfacecolor='none', linestyle='none', label='Validation-selected H')]
    handles += [Line2D([], [], color=c, marker=m, linestyle='none', label='RL seed %d' % s)
                for s, (c, m) in enumerate(zip(colors, markers))]
    fig.suptitle('Author-core reconstruction: complete horizon grid', fontsize=15, fontweight='bold', y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .951), ncol=5, frameon=False)
    fig.text(.06, .025, '20 shared holdout scenes per task; lower cost is better. All seeds and failures retained.\n'
             'Historical holdout already exposed. H is a cost proxy, not measured runtime. Fixed H has one training seed.', fontsize=9, color='#48515A')
    fig.subplots_adjust(left=.085, right=.985, top=.855, bottom=.145, wspace=.27, hspace=.4)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(REPORT / ('complete_grid.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    provenance = {'source': str(result_path.relative_to(ROOT)),
                  'source_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'plotted_models_per_mode': 26, 'modes': ['holdout_value', 'holdout_no_value'],
                  'fixed_selection': 'Mean validation cost with terminal value; identical selected H retained for zero-terminal intervention.',
                  'rl_x_axis': 'Unweighted mean of each episode mean H, as in audited summaries.',
                  'error_bars': 'None: all three training seeds shown individually; fixed H has one seed.'}
    (REPORT / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (REPORT / 'figures.md').write_text('# 完整网格图\n\n![全部固定时域与 RL 种子](complete_grid.png)\n\n'
        '两任务、两种终端设置、全部 26 个模型均展示。RL 点的横坐标是先按回合求均值再平均的 H，固定策略横坐标为其固定 H；不把它当作实测计算时间。车辆纵轴为对数尺度。\n\n'
        '黑色空心菱形标注有终端项验证集选择的 H，无终端项干预仍沿用该 H。RL 三个训练种子分别绘点，没有把 20 个测试场景当作训练种子的重复，也没有用小样本误差带暗示统计显著性。\n\n'
        '[完整数值、成本分解、失败数和限制](report.md) · [PDF](complete_grid.pdf) · [SVG](complete_grid.svg)\n', encoding='utf-8')
    print(str(REPORT / 'complete_grid.png'))


if __name__ == '__main__':
    main()
