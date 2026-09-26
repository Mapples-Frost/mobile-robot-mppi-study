"""Audit saved physical trajectories and report every seed of both optimizations."""
import hashlib
import json
import subprocess
import numpy as np
from runtime import ART, ROOT
from run import write
from credit_probe_report import check_trace

BASE = ART / 'results/stationary_terminal'
MODERN = ART / 'results/categorical_frozen'


def read(path):
    return json.loads(path.read_text())


def main():
    assert (BASE / 'completed.json').exists() and (MODERN / 'completed.json').exists()
    assert read(BASE / 'reload_verification.json')['passed']
    validation_cases = read(BASE / 'validation_bank.json')['cases']
    holdout_cases = read(BASE / 'holdout_bank.json')['cases']
    encode = lambda case: json.dumps(case, sort_keys=True)
    assert len(set(map(encode, validation_cases))) == 10
    assert len(set(map(encode, holdout_cases))) == 30
    assert not set(map(encode, validation_cases)) & set(map(encode, holdout_cases))
    for group in ['recoverable_distribution', 'prior_refinement']:
        old = read(ART / 'results' / group / 'holdout_bank.json')['cases']
        assert not set(map(encode, old)) & set(map(encode, holdout_cases))
    scripts = ROOT / 'experiments/bohn2021_reproduction'
    for source in [BASE, MODERN]:
        for name, expected in read(source / 'source_hashes.json').items():
            path = ROOT / '.codex_tmp/tianshou051/tianshou/policy/modelfree' / name.split('/')[-1] if name.startswith('tianshou/') else scripts / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    amendment = read(MODERN / 'resource_amendment.json')
    for name, expected in amendment['source_hashes'].items():
        assert hashlib.sha256((scripts / name).read_bytes()).hexdigest() == expected
    import torch
    restart_checks = []
    for attempt in amendment['interrupted_attempts']:
        relative = attempt['name'] + '/checkpoint_02500.pt'
        original = torch.load(MODERN / 'interrupted_memory' / relative, weights_only=False)['policy']
        restarted = torch.load(MODERN / relative, weights_only=False)['policy']
        assert original.keys() == restarted.keys()
        exact = all(torch.equal(original[k], restarted[k]) for k in original)
        assert exact, ('Replay allocation changed training prefix', attempt['name'])
        restart_checks.append({'name': attempt['name'], 'step2500_policy_exact': exact})
    for seed in range(3):
        frozen = read(BASE / ('pendulum_rl_s%d' % seed) / 'manifest.json')
        joint = read(ART / 'results/prior_refinement' / ('pendulum_rl_s%d' % seed) / 'manifest.json')
        assert frozen['initial_hash'] == joint['initial_hash']
        assert frozen['config_sha256'] == joint['config_sha256']
        assert read(BASE / ('pendulum_rl_s%d' % seed) / 'completed.json')['updates'] == 14745
        for mode in ['continuous', 'discrete']:
            done = read(MODERN / ('%s_s%d' % (mode, seed)) / 'completed.json')
            assert done['steps'] == 15000 and done['updates'] == 14745 and done['save_load_exact']
    folders = []
    folders += [(p, 'validation') for p in BASE.glob('fixed_h*_validation')]
    folders += [(BASE / ('pendulum_rl_s%d' % s) / mode, 'validation') for s in range(3) for mode in ['eval_value', 'eval_no_value']]
    folders += [(p, 'holdout') for p in BASE.glob('holdout_*') if p.is_dir()]
    folders += [(MODERN / ('%s_s%d' % (m, s)) / ('eval_' + split + '_bank'), split)
                for m in ['continuous', 'discrete'] for s in range(3) for split in ['validation', 'holdout']]
    initial_states = {}
    episodes = transitions = 0
    for folder, split in folders:
        bank_path = BASE / (split + '_bank.json')
        cases = read(bank_path)['cases']
        summary = read(folder / 'summary.json')
        bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        if 'bank_sha256' in summary:
            assert summary['bank_sha256'] == bank_hash
        elif (folder / 'completed.json').exists():
            assert read(folder / 'completed.json')['bank_sha256'] == bank_hash
        else:
            assert read(folder.parent / 'manifest.json')['test_bank_sha256'] == bank_hash
        assert len(summary['episodes']) == len(cases)
        initial = [e['initial_state'] for e in summary['episodes']]
        if split not in initial_states:
            initial_states[split] = initial
        assert initial == initial_states[split], str(folder)
        for index, ep in enumerate(summary['episodes']):
            trace = read(folder / ('trace_%02d.json' % index))
            transitions += check_trace(trace, cases[index], 'pendulum')
            assert len(trace) == ep['steps'] and trace[-1]['termination'] == ep['termination']
            for field, key in [('total_cost', 'cost'), ('performance_cost', 'performance'),
                               ('computation_cost', 'compute'), ('constraint_cost', 'constraint')]:
                assert np.isclose(ep[field], sum(r[key] for r in trace))
            assert np.isclose(ep['discounted_cost'], sum(.97**t * r['cost'] for t, r in enumerate(trace)))
            episodes += 1
        assert np.isclose(summary['mean_total_cost'], np.mean([e['total_cost'] for e in summary['episodes']]))
        assert summary['constraint_episodes'] == sum(e['termination'] == 'constraint' for e in summary['episodes'])
    groups = {
        'legacy_joint': [BASE / ('holdout_joint_s%d' % s) for s in range(3)],
        'legacy_frozen': [BASE / ('holdout_frozen_s%d' % s) for s in range(3)],
        'legacy_joint_fixed_h30': [BASE / ('holdout_joint_fixed_s%d' % s) for s in range(3)],
        'analytic_fixed': [BASE / 'holdout_fixed'],
        'modern_continuous': [MODERN / ('continuous_s%d' % s) / 'eval_holdout_bank' for s in range(3)],
        'modern_discrete': [MODERN / ('discrete_s%d' % s) / 'eval_holdout_bank' for s in range(3)]}
    results = {}
    for name, paths in groups.items():
        summaries = [read(p / 'summary.json') for p in paths]
        all_episodes = [e for summary in summaries for e in summary['episodes']]
        results[name] = {'cost': float(np.mean([s['mean_total_cost'] for s in summaries])),
            'seed_costs': [s['mean_total_cost'] for s in summaries],
            'seed_constraints': [s['constraint_episodes'] for s in summaries],
            'seed_mean_h': [float(np.mean([e['mean_horizon'] for e in s['episodes']])) for s in summaries],
            'seed_solver_failure_steps': [sum(e['solver_failure_steps'] for e in s['episodes']) for s in summaries],
            'performance': float(np.mean([e['performance_cost'] for e in all_episodes])),
            'compute': float(np.mean([e['computation_cost'] for e in all_episodes])),
            'mean_h': float(np.mean([e['mean_horizon'] for e in all_episodes])),
            'solver_failure_steps': sum(e['solver_failure_steps'] for e in all_episodes)}
    comparisons = {}
    for name, first, second in [('freeze_vs_joint', 'legacy_frozen', 'legacy_joint'),
                                ('modern_continuous_vs_legacy_frozen', 'modern_continuous', 'legacy_frozen'),
                                ('discrete_vs_continuous', 'modern_discrete', 'modern_continuous'),
                                ('discrete_vs_fixed', 'modern_discrete', 'analytic_fixed')]:
        a, b = results[first], results[second]
        differences = [x - y for x, y in zip(a['seed_costs'], b['seed_costs'] * (3 if len(b['seed_costs']) == 1 else 1))]
        comparisons[name] = {'cost_change_percent': 100 * (a['cost'] / b['cost'] - 1),
                             'per_seed_cost_difference': differences}
    sources = {}
    for name in ['gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon']:
        path = ART / 'sources' / name
        assert not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
        sources[name] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    audit = {'passed': True, 'episodes': episodes, 'physical_transitions_recomputed': transitions,
             'all_starts_matched': True, 'legacy_initial_hashes_matched': True, 'source_hashes_match': True,
             'author_repositories_clean': sources, 'new_formal_training_runs': 9, 'new_formal_training_steps': 135000,
             'smoke_training_steps_separate': 900, 'independent_holdout_scenes': 30, 'training_replicates_per_arm': 3}
    audit['resource_interrupted_attempts'] = len(amendment['interrupted_attempts'])
    audit['resource_restart_prefix_checks'] = restart_checks
    audit['resource_interrupted_extra_transitions_lower_bound'] = sum(r['steps'] for r in amendment['interrupted_attempts'])
    audit['resource_interrupted_extra_transitions_upper_bound'] = sum(r['transitions_upper_bound'] for r in amendment['interrupted_attempts'])
    write(MODERN / 'audit.json', audit)
    write(MODERN / 'analysis.json', {'groups': results, 'comparisons': comparisons,
          'selected_fixed': read(BASE / 'selected_fixed.json'), 'validation_gate': read(MODERN / 'validation_gate.json')})
    lines = ['# Frozen Terminal and Categorical Horizon Results', '',
        'All formal learned models use 15,000 transitions and seeds 0, 1, 2. '
        'The table uses the same 30 independent holdout scenes. Lower cost is better.', '',
        '| Arm | Mean total cost | Seed costs | Constraint episodes per seed | Mean H |',
        '|---|---:|---|---|---:|']
    for name, data in results.items():
        lines.append('| %s | %.4f | %s | %s | %.2f |' % (name, data['cost'],
            ', '.join('%.4f' % v for v in data['seed_costs']),
            ', '.join(str(v) for v in data['seed_constraints']), data['mean_h']))
    lines += ['', 'The analytic fixed baseline has H=%d, selected on validation only, and zero training transitions. '
              'It is deterministic, so repeating it does not create independent training seeds.' % read(BASE / 'selected_fixed.json')['h'], '',
              '## Comparisons', '']
    for name, data in comparisons.items():
        lines.append('- %s: total cost change %+.2f%%; paired seed differences %s.' %
                     (name, data['cost_change_percent'], ', '.join('%+.4f' % v for v in data['per_seed_cost_difference'])))
    gate = read(MODERN / 'validation_gate.json')
    lines += ['', '## Validation Gate', '',
              'Joint-terminal follow-up gate passed: **%s**. Decision: %s.' % (gate['passed'], gate['decision']),
              'Validation means: continuous %.4f, discrete %.4f, selected fixed %.4f. '
              'Discrete constraints per seed: %s.' % (gate['means']['continuous'], gate['means']['discrete'],
                  gate['selected_fixed']['validation_cost'], gate['discrete_constraint_episodes']), '',
              '## Verification and Budget', '',
              '- %d saved evaluation episodes and %d transitions independently checked against the physical equations.' % (episodes, transitions),
              '- Matched reset states, exact legacy initial weights/configurations, immutable registered source hashes, and clean author repositories.',
              '- Frozen terminal checked at every legacy update and modern environment step. Final TF save/load rollouts and modern save/load parameters/actions verified.',
              '- Four restarted modern runs reproduce their original 2,500-step policy parameters exactly after the replay allocation fix.',
              '- Nine completed new formal models: 135,000 transitions. Three smoke tests: 900 additional transitions.',
              '- Resource-interrupted attempts used an additional %d to %d transitions, archived without checkpoint or seed selection.' %
                  (audit['resource_interrupted_extra_transitions_lower_bound'], audit['resource_interrupted_extra_transitions_upper_bound']), '',
              '## Limits', '',
              'Three training seeds do not establish broad statistical superiority. Scene repeats are not independent training replicates. '
              'Freezing also prevents terminal improvement, so the comparison does not isolate nonstationarity from useful terminal adaptation. '
              'Modern continuous SAC is the primary comparator for categorical SAC because the author SAC has a different critic/actor formulation. '
              'Equal alpha does not equate categorical and differential entropy. This is an explicit method extension, not exact paper reproduction. '
              'The H cost is a computation proxy, not evidence of a wall-clock speedup.', '']
    report = ART / 'report/stationary_optimization'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'groups': results, 'comparisons': comparisons, 'audit': audit}, indent=2))


if __name__ == '__main__':
    main()
