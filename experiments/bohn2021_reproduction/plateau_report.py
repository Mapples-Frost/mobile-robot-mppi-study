"""Independently recompute pilot trajectories and produce a complete report."""
import hashlib
import json

import numpy as np
from plateau_screen import OUT, ROOT, STEPS, OFFSET, COARSE, RULES, verify_registration
from run import write


def read(path):
    return json.loads(path.read_text())


def aggregate(rows):
    keys = ['total_cost', 'shifted_total_cost', 'shifted_performance_cost',
            'computation_cost', 'constraint_cost', 'tracking_rmse', 'mean_horizon']
    return dict({key: float(np.mean([r[key] for r in rows])) for key in keys},
        constraint_episodes=sum(r['termination'] == 'constraint' for r in rows),
        solver_failure_steps=sum(r['solver_failure_steps'] for r in rows),
        episodes=len(rows))


def audit():
    verify_registration()
    selection = read(OUT / 'selection.json')
    expected_validation = set(['fixed_%d' % h for h in COARSE] + RULES +
        ['fixed_%d' % h for h in read(OUT / 'refinement.json')['additional_fixed_h']])
    expected_holdout = {selection['fixed'], selection['switch'], 'fixed_30'}
    checks = {'episodes': 0, 'transitions': 0, 'files_sha256': {}, 'splits': {}}
    all_rows = {}
    for split, expected in [('validation', expected_validation), ('holdout', expected_holdout)]:
        bank = read(OUT / (split + '_bank.json'))['scenes']
        starts = {}
        paths = sorted((OUT / split).glob('*/summary.json'))
        assert {p.parent.name for p in paths} == expected
        all_rows[split] = {}
        for path in paths:
            arm = path.parent.name
            saved = read(path)
            rows = []
            assert len(saved['episodes']) == len(bank)
            for scene in bank:
                episode_path = path.parent / ('episode_%02d.json' % scene['id'])
                ep = read(episode_path)
                summary, trace = ep['summary'], ep['trace']
                assert summary == saved['episodes'][scene['id']]
                initial = summary['initial_state']
                if scene['id'] in starts:
                    assert initial == starts[scene['id']]
                else:
                    starts[scene['id']] = initial
                assert len(trace) == summary['steps']
                assert len(trace) == STEPS or trace[-1]['termination'] == 'constraint'
                totals = np.zeros(3)
                errors, horizons = [], []
                phases = {name: np.zeros(4) for name in ['plateau', 'transition']}
                before = initial
                for t, row in enumerate(trace):
                    clock = t + 1
                    assert row['clock'] == clock and row['next_clock'] == clock + 1
                    refs = scene['case']['tvp']['pos_r']
                    ref = refs[clock + 1]['true'][0]
                    assert row['reference'] == ref
                    if arm.startswith('fixed_'):
                        h, reason = int(arm.split('_')[1]), 'fixed'
                    else:
                        _, short, long = arm.split('_')
                        short, long = int(short), int(long)
                        current_ref = refs[clock]['true'][0]
                        upcoming = any(abs(refs[k]['true'][0] - current_ref) > 1e-9
                                       for k in range(clock + 1, clock + long + 1))
                        unsettled = (abs(before['pos'] - current_ref) > .03 or
                            abs(before['v']) > .08 or abs(before['theta']) > .04 or
                            abs(before['omega']) > .12)
                        h = long if upcoming or unsettled else short
                        reason = 'preview' if upcoming else ('recovery' if unsettled else 'settled')
                    assert row['horizon'] == h and row['reason'] == reason
                    state, u = row['state'], row['input']['u1']
                    assert all(np.isfinite(v) for v in state.values()) and abs(u) <= 5 + 1e-5
                    x, v, theta, omega = [state[key] for key in ['pos', 'v', 'theta', 'omega']]
                    perf = (.4*v*v + .05*v*omega*np.cos(theta) +
                        (2/3)*.2*.25**2*omega*omega - OFFSET*np.cos(theta) +
                        10*(x-ref)**2 + .1*u*u)
                    violation = abs(x) > 1.5 or abs(theta) > np.pi/2
                    penalty = 10*(STEPS-t-1) if violation else 0.
                    np.testing.assert_allclose([row['performance'], row['compute'], row['constraint']],
                        [perf, .003*h, penalty], rtol=1e-10, atol=1e-8)
                    assert np.isclose(row['cost'], perf + .003*h + penalty, atol=1e-8)
                    if violation:
                        assert t == len(trace)-1 and row['termination'] == 'constraint'
                    totals += [perf, .003*h, penalty]
                    errors.append((x-ref)**2)
                    horizons.append(h)
                    phase = 'transition' if any(s-50 <= clock < s+100 for s in scene['switch_clocks']) else 'plateau'
                    assert phase == row['phase']
                    phases[phase] += [1, perf+OFFSET, .003*h, penalty]
                    before = state
                np.testing.assert_allclose(totals,
                    [summary['performance_cost'], summary['computation_cost'], summary['constraint_cost']], atol=1e-7)
                np.testing.assert_allclose([sum(totals), sum(totals)+OFFSET*STEPS,
                    totals[0]+OFFSET*len(trace), np.sqrt(np.mean(errors)), np.mean(horizons)],
                    [summary['total_cost'], summary['shifted_total_cost'], summary['shifted_performance_cost'],
                     summary['tracking_rmse'], summary['mean_horizon']], atol=1e-7)
                assert summary['solver_failure_steps'] == sum(not r['solver_success'] for r in trace)
                for phase, values in phases.items():
                    np.testing.assert_allclose(values, [summary['phase_costs'][phase][key] for key in
                        ['steps', 'shifted_performance_cost', 'computation_cost', 'constraint_cost']], atol=1e-7)
                checks['episodes'] += 1
                checks['transitions'] += len(trace)
                checks['files_sha256'][str(episode_path.relative_to(ROOT))] = hashlib.sha256(episode_path.read_bytes()).hexdigest()
                rows.append(summary)
            stats = aggregate(rows)
            assert np.isclose(stats['total_cost'], saved['mean_total_cost'])
            assert np.isclose(stats['shifted_total_cost'], saved['mean_shifted_total_cost'])
            all_rows[split][arm] = {'summary': stats, 'episodes': rows}
        checks['splits'][split] = {'arms': len(paths), 'matched_starts': len(starts)}
    for arm, expected_hash in selection['validation_summaries_sha256'].items():
        assert hashlib.sha256((OUT/'validation'/arm/'summary.json').read_bytes()).hexdigest() == expected_hash
    for prefix, key in [('fixed_', 'fixed'), ('switch_', 'switch')]:
        best = min((name for name in all_rows['validation'] if name.startswith(prefix)),
                   key=lambda name: (all_rows['validation'][name]['summary']['total_cost'], name))
        assert best == selection[key]
    checks['verified'] = ['registered sources and banks', 'all registered arms and scenes',
        'matched reset starts', 'physical cost and constraints', 'reference clocks',
        'switch decisions use only allowed preview and current state', 'phase decomposition',
        'summary aggregation', 'selection uses validation only']
    write(OUT / 'audit.json', checks)
    return all_rows, selection, checks


def figures(rows, selection, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fixed = sorted((int(name.split('_')[1]), r['summary']['shifted_total_cost'])
                   for name, r in rows['validation'].items() if name.startswith('fixed_'))
    axes[0, 0].plot([r[0] for r in fixed], [r[1] for r in fixed], 'o-', color='#356FA2', ms=4, label='Fixed H (mean of 3 scenes)')
    switch_cost = rows['validation'][selection['switch']]['summary']['shifted_total_cost']
    axes[0, 0].axhline(switch_cost, color='#BB493F', ls='--', label=selection['switch'])
    axes[0, 0].set(xlabel='Fixed horizon H', ylabel='Offset-adjusted total cost', yscale='log', title='A. Validation screening (lower is better)')
    axes[0, 0].legend(fontsize=8)
    colors = {selection['fixed']: '#356FA2', selection['switch']: '#BB493F', 'fixed_30': '#53895C'}
    for name, r in rows['holdout'].items():
        axes[0, 1].scatter(np.arange(1, 7), [v['shifted_total_cost'] for v in r['episodes']],
                         label=name, color=colors[name], s=35)
    fixed_rows = rows['holdout'][selection['fixed']]['episodes']
    switch_rows = rows['holdout'][selection['switch']]['episodes']
    for i, (a, b) in enumerate(zip(fixed_rows, switch_rows)):
        axes[0, 1].plot([i+1, i+1], [a['shifted_total_cost'], b['shifted_total_cost']], color='0.65', zorder=0)
    axes[0, 1].set(xlabel='Holdout scene (0.4, 0.6, 0.8 m; repeated)', ylabel='Offset-adjusted total cost', title='B. All six holdout scenes')
    axes[0, 1].legend(fontsize=8)
    # The first middle-amplitude holdout scene is fixed in advance, not selected by result.
    for name in [selection['fixed'], selection['switch']]:
        ep = read(OUT/'holdout'/name/'episode_01.json')
        trace = ep['trace']
        time_s = [r['next_clock']*.04 for r in trace]
        axes[1, 0].plot(time_s, [r['state']['pos'] for r in trace], color=colors[name], label=name)
        axes[1, 1].step([r['clock']*.04 for r in trace], [r['horizon'] for r in trace], where='post', color=colors[name], label=name)
    axes[1, 0].step(time_s, [r['reference'] for r in trace], where='post', color='0.2', ls='--', label='Reference')
    axes[1, 0].set(xlabel='Time (s)', ylabel='Cart position (m)', title='C. Prespecified holdout scene 2')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(xlabel='Time (s)', ylabel='Executed H', ylim=(0, 52), title='D. Horizon choices in the same scene')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.savefig(dest/'overview.png', dpi=170)
    fig.savefig(dest/'overview.pdf')
    plt.close(fig)


def main():
    rows, selection, checks = audit()
    dest = OUT.parents[1] / 'report/plateau_screen'
    dest.mkdir(parents=True, exist_ok=True)
    fixed = rows['holdout'][selection['fixed']]['summary']
    switch = rows['holdout'][selection['switch']]['summary']
    paired = []
    for a, b in zip(rows['holdout'][selection['fixed']]['episodes'], rows['holdout'][selection['switch']]['episodes']):
        paired.append({'scene': a['scene'], 'amplitude': a['amplitude'],
            'fixed_minus_switch': a['total_cost']-b['total_cost'],
            'performance_saving': a['performance_cost']-b['performance_cost'],
            'computation_saving': a['computation_cost']-b['computation_cost'],
            'constraint_saving': a['constraint_cost']-b['constraint_cost']})
    phase_savings = {}
    for phase in ['plateau', 'transition']:
        phase_savings[phase] = {}
        for key in ['shifted_performance_cost', 'computation_cost', 'constraint_cost']:
            phase_savings[phase][key] = float(np.mean([
                a['phase_costs'][phase][key] - b['phase_costs'][phase][key]
                for a, b in zip(rows['holdout'][selection['fixed']]['episodes'],
                                rows['holdout'][selection['switch']]['episodes'])]))
    analysis = {'selection': selection, 'results': rows, 'paired_holdout': paired,
        'phase_savings_fixed_minus_switch': phase_savings,
        'mean_fixed_minus_switch': fixed['total_cost']-switch['total_cost'],
        'shifted_cost_reduction_percent': 100*(fixed['shifted_total_cost']-switch['shifted_total_cost'])/fixed['shifted_total_cost'],
        'audit': {'episodes': checks['episodes'], 'transitions': checks['transitions']}}
    write(OUT/'analysis.json', analysis)
    text = ['# Pendulum Plateau and Preview Pilot', '',
        'Exploratory task extension; no RL training. Costs are lower-is-better.', '',
        'Each scene has 600 steps (24 seconds), two reference reversals, common 50-step preview, and the same frozen Riccati terminal. Three validation scenes cover amplitudes 0.4/0.6/0.8 m; six new holdout scenes repeat these amplitudes twice.', '',
        'Fixed H uses a coarse grid plus every integer within +/-4 of the best coarse H. This is the best tested fixed policy, not a certified global optimum. One fixed policy and one switching rule are selected globally on validation only.', '',
        'Adjusted cost = raw cost + 0.4905*600. This common constant preserves rankings, including failed episodes, and removes the equilibrium potential offset for full-length trajectories. Percentages depend on this stated normalization; raw reward percentages are not reported.', '',
        '## All Validation Arms', '', '| Arm | Adjusted cost | Physical cost above equilibrium | H cost | Failures | Mean H |', '|---|---:|---:|---:|---:|---:|']
    for name, r in sorted(rows['validation'].items()):
        s = r['summary']
        text.append('| %s | %.4f | %.4f | %.4f | %d/%d | %.2f |' % (name, s['shifted_total_cost'], s['shifted_performance_cost'], s['computation_cost'], s['constraint_episodes'], s['episodes'], s['mean_horizon']))
    text.extend(['', '## Holdout', '', '| Arm | Raw cost | Adjusted cost | Physical cost above equilibrium | H cost | Failures | Mean H |', '|---|---:|---:|---:|---:|---:|---:|'])
    for name, r in sorted(rows['holdout'].items()):
        s = r['summary']
        text.append('| %s | %.4f | %.4f | %.4f | %.4f | %d/%d | %.2f |' % (name, s['total_cost'], s['shifted_total_cost'], s['shifted_performance_cost'], s['computation_cost'], s['constraint_episodes'], s['episodes'], s['mean_horizon']))
    text.extend(['', 'Selected fixed: `%s`; selected rule: `%s`.' % (selection['fixed'], selection['switch']), '',
        'Mean fixed-minus-switch cost: %.6f; adjusted-cost reduction: %.3f%%.' % (analysis['mean_fixed_minus_switch'], analysis['shifted_cost_reduction_percent']), '',
        '| Scene | Amplitude | Fixed minus switch | Physical saving | H saving | Constraint saving |', '|---|---:|---:|---:|---:|---:|'])
    for r in paired:
        text.append('| %d | %.1f | %.6f | %.6f | %.6f | %.6f |' % tuple(r[k] for k in ['scene', 'amplitude', 'fixed_minus_switch', 'performance_saving', 'computation_saving', 'constraint_saving']))
    text.extend(['', 'Mean phase decomposition, fixed minus switch (positive is a saving):', '',
        '| Phase | Physical saving | H saving | Constraint saving |', '|---|---:|---:|---:|'])
    for phase, values in phase_savings.items():
        text.append('| %s | %.6f | %.6f | %.6f |' % (phase, values['shifted_performance_cost'],
            values['computation_cost'], values['constraint_cost']))
    text.extend(['', '## Verification and Scope', '',
        '%d episodes / %d physical transitions independently recomputed. Reference clocks, identical reset states, input bounds, constraints, allowed preview, rule actions, source hashes and validation selection were checked.' % (checks['episodes'], checks['transitions']), '',
        'Frozen terminal equality was asserted after every runtime step. No actual CPU acceleration is claimed. There are no learned-policy seeds; six holdout scenes are a small exploratory check, not evidence of broad statistical generalization. Extending episodes retains the existing failure penalty of 10 times remaining steps; failures and cost components remain visible.', '',
        'The original source generator redraws the reference every 25 steps deterministically (given redraw_probability=0.04); randomness is in reference values. This pilot introduces randomized reversal times explicitly.', '',
        '![Pilot overview](overview.png)', ''])
    (dest/'report.md').write_text('\n'.join(text))
    figures(rows, selection, dest)
    print(json.dumps({'selection': {k: selection[k] for k in ['fixed', 'switch']},
        'holdout': {k: v['summary'] for k, v in rows['holdout'].items()},
        'paired': paired, 'audit': analysis['audit'], 'report': str(dest/'report.md')}, indent=2))


if __name__ == '__main__':
    main()
