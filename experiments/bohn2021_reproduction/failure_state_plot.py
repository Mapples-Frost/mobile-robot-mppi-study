"""All-seed descriptive diagnostic plots after complete timing and review.

The figure is not a confirmation test. Previews require recorded visual review
before vector export. Runs on bundled Windows Python or a plotting environment.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
DEST = OUT / 'figures'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name):
    return ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def guard():
    finish = read(OUT / 'finish_status.json')
    timing = read(OUT / 'timing_status.json')
    assert finish['complete'] and not finish['active'] and timing['complete'] and not timing['active']
    proc = Path('/proc') if Path('/proc').exists() else Path('//wsl.localhost/Ubuntu-20.04/proc')
    assert not (proc / str(finish['pid']) / 'cmdline').exists()
    assert read(OUT / 'delivery_review/report.json')['passed']
    assert read(OUT / 'timing_delivery/report.json')['passed']
    for p, h in finish['output_hashes'].items():
        assert sha(resolve(p)) == h


def prepare(skill):
    guard()
    names = ('delivery/report.json', 'delivery/all_comparisons.csv',
             'timing_delivery/report.json', 'timing_delivery/all_comparisons.csv',
             'delivery_review/report.json', 'audit_full.json')
    inputs = {n: sha(OUT / n) for n in names}
    expected = dict(source_sha256=sha(Path(__file__)), inputs=inputs, diagnostic_only=True,
        question='Across all three trained terminals, does the hand-coded failure-state rule trade control cost for measured time, and do longer vehicle horizons help consistently?',
        choice='Two task-specific point plots. Pendulum: total/physical costs and mean/repeat latency. Vehicle: total and physical costs, with prerequisite failures marked.',
        alternative='Per-case paired scatter remains available through all_paired_cases.csv; no pooled seed mean or confidence interval is shown.',
        inference='Post-hoc exposed training cases,24 per condition. Three terminals and two serial timing repeats; timing repeats are not extra scenes. No significance or learned-policy claim.',
        dimensions_inches=[7.2, 5.4], min_font_pt=7.5, dpi=300)
    DEST.mkdir(exist_ok=True)
    reg = DEST / 'registration.json'
    if reg.exists():
        assert read(reg) == expected
    else:
        write(reg, expected)
    report = read(OUT / 'delivery/report.json')
    timing = read(OUT / 'timing_delivery/report.json')
    assert len(report['comparisons']) == 18 and len(timing['comparisons']) == 9
    for task in ('pendulum', 'vehicle'):
        rows = []
        for c in report['comparisons']:
            if c['task'] != task:
                continue
            row = dict(seed='S%d' % c['seed'], policy=c['policy'], total_cost_change_percent=c['total_change_percent'],
                       physical_cost_change_percent=c['physical_change_percent'], safe=c['safe'], physical_noninferior=c['physical_noninferior'])
            if task == 'pendulum':
                matches = [t for t in timing['comparisons'] if (t['seed'], t['policy']) == (c['seed'], c['policy'])]
                assert len(matches) == 1
                t = matches[0]
                row.update(time_change_percent=t['mean_time_change_percent'], repeat0_percent=t['repeat0_change_percent'], repeat1_percent=t['repeat1_change_percent'])
            rows.append(row)
        assert len(rows) == 9
        path = DEST / (task + '.csv')
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with (DEST / (task + '_profile.md')).open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, str(skill / 'profile_data.py'), str(path), '--group', 'policy'], stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(DEST / 'input_audit.json', dict(passed=True, registration_sha256=sha(reg),
        inputs=inputs, task_csv_hashes={t: sha(DEST / (t + '.csv')) for t in ('pendulum', 'vehicle')},
        note='Seed and policy are categorical. Each plotted point summarizes24 paired scenes; the three terminal seeds are retained separately. Aggregate columns are correlated summaries, not independent samples.'))
    print('Prepared18 all-seed rows and two data profiles')


def render(skill, export=False):
    guard()
    spec = read(DEST / 'registration.json')
    assert spec['source_sha256'] == sha(Path(__file__))
    for p, h in spec['inputs'].items():
        assert sha(OUT / p) == h
    inp = read(DEST / 'input_audit.json')
    sys.path.insert(0, str(skill))
    from visual_qa import audit_layout
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from PIL import Image
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    outcome = {}
    for task in ('pendulum', 'vehicle'):
        assert sha(DEST / (task + '.csv')) == inp['task_csv_hashes'][task]
        with (DEST / (task + '.csv')).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        labels = [r['seed'] + ' / ' + r['policy'].replace('certificate', 'C').replace('fixed', 'H').replace('selected', 'gate')
                  + (' !' if r['safe'] == 'False' or r['physical_noninferior'] == 'False' else '') for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4), sharey=True)
        values = [[], []]
        for i, r in enumerate(rows):
            total = float(r['total_cost_change_percent']); physical = float(r['physical_cost_change_percent'])
            if task == 'pendulum':
                axes[0].scatter(total, i-.09, marker='s', s=27, color='#0072B2', zorder=4)
                axes[0].scatter(physical, i+.09, marker='o', s=29, facecolors='none', edgecolors='#D55E00', zorder=4)
                values[0].extend((total, physical))
                for key, marker, offset, filled in (('time_change_percent', 'o', 0, True), ('repeat0_percent', '<', -.15, False), ('repeat1_percent', '>', .15, False)):
                    value = float(r[key]); values[1].append(value)
                    axes[1].scatter(value, i+offset, marker=marker, s=27, facecolors='#333333' if filled else 'none', edgecolors='#333333', zorder=4)
            else:
                for j, v in enumerate((total, physical)):
                    axes[j].scatter(v, i, s=29, marker='s' if j == 0 else 'o',
                        facecolors='#0072B2' if j == 0 else 'none', edgecolors='#0072B2' if j == 0 else '#D55E00', zorder=4)
                    values[j].append(v)
        for j, ax in enumerate(axes):
            ax.set_yticks(range(9)); ax.set_yticklabels(labels)
            ax.set_ylim(8.5, -.6)
            ax.axvline(0, color='#555555', linewidth=.8, linestyle='--')
            ax.grid(axis='x', color='#dddddd', linewidth=.6)
            lo, hi = min(values[j]+[0.]), max(values[j]+[0.]); span=max(hi-lo, .2)
            ax.set_xlim(lo-.1*span, hi+.1*span)
            ax.xaxis.set_major_locator(MaxNLocator(5))
        axes[0].set_xlabel('Change in cost (%)')
        if task == 'pendulum':
            axes[0].set_title('a   Total and physical costs', fontsize=10, loc='left')
            axes[1].set_title('b   Measured decision time', fontsize=10, loc='left')
            axes[1].set_xlabel('Change in time per step (%)')
            handles = [Line2D([], [], color='#0072B2', marker='s', linestyle='None', label='Total cost'),
                       Line2D([], [], color='#D55E00', marker='o', markerfacecolor='none', linestyle='None', label='Physical + constraint cost'),
                       Line2D([], [], color='#333333', marker='o', linestyle='None', label='Mean ratio'),
                       Line2D([], [], color='#333333', marker='<', markerfacecolor='none', linestyle='None', label='Repeat1'),
                       Line2D([], [], color='#333333', marker='>', markerfacecolor='none', linestyle='None', label='Repeat2')]
            foot='C5/C10/C15: hand-coded short-H rules; reference: same-terminal H30.\n24 exposed training scenes per seed; two timing repeats do not add scenes.\nNegative changes favor the rule. Descriptive only; no confidence intervals or learned-policy claim.'
        else:
            axes[0].set_title('a   Total cost', fontsize=10, loc='left')
            axes[1].set_title('b   Physical + constraint cost', fontsize=10, loc='left')
            axes[1].set_xlabel('Change in cost (%)')
            handles = []
            foot='Reference: same-terminal H25; gate: inherited policy selected on these training cases.\n24 exposed training scenes per seed; all three terminal seeds retained.\n! Safety or physical-cost premise fails. Negative changes favor the candidate; no efficacy claim.'
        fig.suptitle(task.capitalize() + ': complete training-case diagnosis', fontsize=11, y=.975)
        if handles:
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.13), ncol=3, frameon=False, fontsize=7.5)
        fig.text(.5, .025, foot, ha='center', fontsize=7.5, linespacing=1.5)
        fig.tight_layout(rect=(.01, .24 if handles else .18, 1, .93), w_pad=1.4)
        preview = DEST / (task + '_preview.png')
        fig.savefig(preview, dpi=180)
        with Image.open(preview) as im:
            im.convert('L').save(DEST / (task + '_grayscale.png'))
        issues = audit_layout(fig)
        outcome[task] = dict(issues=issues, preview_sha256=sha(preview), grayscale_sha256=sha(DEST / (task + '_grayscale.png')),
            plotted_rows=9, point_markers=sum(len(c.get_offsets()) for ax in axes for c in ax.collections))
        if export:
            visual = read(DEST / 'visual_review.json')
            assert visual['passed'] and visual['previews'][task] == sha(preview)
            assert not issues, issues
            for suffix in ('pdf', 'svg'):
                fig.savefig(DEST / (task + '.' + suffix))
            fig.savefig(DEST / (task + '.png'), dpi=300)
        plt.close(fig)
    write(DEST / 'layout_audit.json', dict(source_sha256=sha(Path(__file__)), tasks=outcome, exported=export))
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('prepare', 'render'), required=True)
    ap.add_argument('--skill-scripts', type=Path, required=True)
    ap.add_argument('--export', action='store_true')
    a=ap.parse_args()
    if a.mode == 'prepare':
        prepare(a.skill_scripts)
    else:
        render(a.skill_scripts, a.export)
