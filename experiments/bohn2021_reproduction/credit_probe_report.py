"""Audit all saved diagnosis trajectories and summarize without significance claims."""
import hashlib
import json

import numpy as np

from runtime import ART, ROOT
from run import write

OUT = ART / 'results/horizon_credit_probe'


def read(path):
    return json.loads(path.read_text())


def check_trace(trace, case, task, offset=0):
    for k, row in enumerate(trace):
        t = offset + k
        s, u, h = row['state'], row['input'], row['horizon']
        ref = lambda key: case['tvp'][key][t + 2]['true'][0]
        if task == 'pendulum':
            perf = (.4 * s['v']**2 + .05 * s['v'] * s['omega'] * np.cos(s['theta'])
                    + (2 / 3) * .2 * .25**2 * s['omega']**2 - .2 * 9.81 * .25 * np.cos(s['theta'])
                    + 10 * (s['pos'] - ref('pos_r'))**2 + .1 * u['u1']**2)
            violation = abs(s['pos']) > 1.5 or abs(s['theta']) > np.pi / 2
            penalty = 10 * (100 - t - 1) if violation else 0.
            compute = .003 * h
            assert abs(u['u1']) <= 5 + 1e-5
        else:
            perf = (s['x'] - ref('trajectory_x'))**2 + (s['y'] - ref('trajectory_y'))**2
            violation = any(np.hypot(s['x'] - ref('obj_%d_x' % j), s['y'] - ref('obj_%d_y' % j)) <= ref('obj_%d_r' % j) for j in range(3))
            penalty = 2 * (150 - t - 1) if violation else 0.
            compute = .001 * h
            assert -1e-5 <= u['u_s'] <= 5 + 1e-5 and abs(u['u_omega']) <= 4 + 1e-5
        assert 1 <= h <= 50 and int(h) == h
        assert np.isclose(row['performance'], perf, atol=1e-8, rtol=1e-8)
        assert np.isclose(row['compute'], compute)
        assert np.isclose(row['constraint'], penalty)
        assert np.isclose(row['cost'], perf + compute + penalty)
        if violation:
            assert k == len(trace) - 1 and row['termination'] == 'constraint'
    return len(trace)


def main():
    assert read(OUT / 'completed.json')['complete']
    protocol = read(OUT / 'protocol.json')
    assert hashlib.sha256((ROOT / 'experiments/bohn2021_reproduction/horizon_credit_probe.py').read_bytes()).hexdigest() == protocol['script_sha256']
    models = list(OUT.glob('*_s*/summary.json'))
    assert len(models) == 6
    branches = transitions = 0
    rows = []
    for path in sorted(models):
        model = read(path)
        assert model['complete'] and model['frozen']
        source = ART / 'results' / model['group']
        bank = read(source / 'validation_bank.json')['cases']
        assert len(model['results']) == 4
        for d in model['results']:
            a, q = str(d['picks']['actor']), str(d['picks']['q1'])
            anchor = path.parent / ('case%d_t%d' % (d['case'], d['t']))
            expected = {'det_h%s.json' % h for h in d['deterministic']}
            expected |= {'stoch_h%s_r%d.json' % (h, r) for h in d['stochastic'] for r in range(4)}
            assert {p.name for p in anchor.glob('*.json') if p.name != 'summary.json'} == expected
            initial = None
            for name in sorted(expected):
                b = read(anchor / name)
                if initial is None:
                    initial = b['initial_observation']
                assert initial == b['initial_observation']
                tr = b['trace']
                transitions += check_trace(tr, bank[d['case']], 'pendulum', d['t'])
                branches += 1
                assert len(tr) == b['steps'] and tr[-1]['termination'] == b['termination']
                assert np.isclose(b['cost'], sum(r['cost'] for r in tr))
                assert np.isclose(b['discounted_cost'], sum(.97**k * r['cost'] for k, r in enumerate(tr)))
                assert tr[0]['continuation_log_probability'] == 0.
                entropy = sum(.97**k * (-.01 * r['continuation_log_probability']) for k, r in enumerate(tr))
                assert np.isclose(b['soft_return_finite'], -b['discounted_cost'] / .6 + entropy)
                assert np.isclose(b['soft_return_with_target_tail'], b['soft_return_finite'] + b['discounted_target_tail'])
            deterministic = d['deterministic']
            differences = {key: [d['stochastic'][q][i][key] - d['stochastic'][a][i][key] for i in range(4)]
                           for key in ['soft_return_finite', 'soft_return_with_target_tail']}
            rows.append({'group': model['group'], 'seed': model['seed'], 'case': d['case'], 't': d['t'],
                         'picks': d['picks'], 'q1_minus_actor_discounted_cost': deterministic[q]['discounted_cost'] - deterministic[a]['discounted_cost'],
                         'minq_minus_actor_discounted_cost': deterministic[str(d['picks']['min_q'])]['discounted_cost'] - deterministic[a]['discounted_cost'],
                         'q1_minus_actor_soft_return_samples': differences,
                         'critic_q1_gain': d['q1'][int(q) - 1] - d['q1'][int(a) - 1]})
    full = ART / 'results/critic_execution_probe'
    assert read(full / 'summary.json')['complete']
    full_rows = []
    for model in read(full / 'summary.json')['results']:
        group, seed = model['group'], model['seed']
        source = ART / 'results' / group
        bank = read(source / 'validation_bank.json')['cases']
        original = read(source / ('pendulum_rl_s%d' % seed) / 'eval_value/summary.json')
        initial = [e['initial_state'] for e in original['episodes']]
        row = {'group': group, 'seed': seed, 'actor_cost': original['mean_total_cost'], 'actor_constraints': original['constraint_episodes']}
        for mode in ['q1', 'min_q']:
            target = full / ('%s_s%d' % (group, seed)) / mode
            summary = read(target / 'summary.json')
            assert read(target / 'completed.json')['frozen']
            assert initial == [e['initial_state'] for e in summary['episodes']]
            for i, ep in enumerate(summary['episodes']):
                tr = read(target / ('trace_%02d.json' % i))
                transitions += check_trace(tr, bank[i], 'pendulum')
                assert np.isclose(ep['total_cost'], sum(r['cost'] for r in tr))
                assert len(tr) == ep['steps']
            row[mode + '_cost'] = summary['mean_total_cost']
            row[mode + '_constraints'] = summary['constraint_episodes']
        full_rows.append(row)
    vehicle = ART / 'results/vehicle_stall_probe'
    v = read(vehicle / 'summary.json')
    assert v['complete'] and len(v['results']) == 6
    bank = read(ART / 'results/vehicle_entropy/holdout_bank.json')['cases']
    for case in v['results']:
        assert case['frozen'] and case['original_suffix_exact']
        for mode in ['original', 'force10']:
            result = read(vehicle / ('case%02d' % case['case']) / (mode + '.json'))
            transitions += check_trace(result['trace'], bank[case['case']], 'vehicle', 130)
            assert np.isclose(result['cost'], sum(r['cost'] for r in result['trace']))
    objectives = read(OUT / 'actor_objective.json')
    assert objectives['complete'] and objectives['frozen'] and len(objectives['rows']) == 24
    summary = {}
    for group in ['recoverable_distribution', 'prior_refinement']:
        selected = [r for r in rows if r['group'] == group]
        different = [r for r in selected if r['picks']['actor'] != r['picks']['q1']]
        complete = [r for r in full_rows if r['group'] == group]
        summary[group] = {
            'anchors': len(selected), 'actor_q1_disagreements': len(different),
            'q1_single_action_better': sum(r['q1_minus_actor_discounted_cost'] < -1e-5 for r in different),
            'q1_single_action_worse': sum(r['q1_minus_actor_discounted_cost'] > 1e-5 for r in different),
            'q1_soft_finite_mean_better': sum(np.mean(r['q1_minus_actor_soft_return_samples']['soft_return_finite']) > 0 for r in different),
            'q1_soft_tail_mean_better': sum(np.mean(r['q1_minus_actor_soft_return_samples']['soft_return_with_target_tail']) > 0 for r in different),
            'full_validation': {mode: {'mean_cost': float(np.mean([r[mode + '_cost'] for r in complete])),
                                       'seed_costs': [r[mode + '_cost'] for r in complete],
                                       'constraints_per_seed': [r[mode + '_constraints'] for r in complete]}
                                for mode in ['actor', 'q1', 'min_q']}}
    result = {'complete': True, 'summary': summary, 'anchors': rows, 'full_validation': full_rows,
              'actor_objective_gain_over_001': sum(r['same_std_objective_gain'] > .01 for r in objectives['rows']),
              'vehicle': v, 'scope': 'Post-hoc validation and selected-failure diagnosis; no population significance or new holdout superiority claim.'}
    write(OUT / 'analysis.json', result)
    scripts = ['horizon_credit_probe.py', 'actor_objective_audit.py', 'vehicle_stall_probe.py', 'critic_execution_probe.py', 'credit_probe_report.py']
    write(OUT / 'audit.json', {'passed': True, 'new_training_runs': 0, 'models': 6, 'anchors': 24,
                              'branch_rollouts': branches, 'full_validation_evaluations': 120, 'vehicle_suffix_rollouts': 12,
                              'physical_transitions_recomputed': transitions,
                              'original_actor_suffixes_replayed_exactly': 24, 'vehicle_original_suffixes_replayed_exactly': 6,
                              'source_sha256': {n: hashlib.sha256((ROOT / 'experiments/bohn2021_reproduction' / n).read_bytes()).hexdigest() for n in scripts}})
    print(json.dumps(read(OUT / 'analysis.json')['summary'], indent=2))
    print(json.dumps(read(OUT / 'audit.json'), indent=2))


if __name__ == '__main__':
    main()
