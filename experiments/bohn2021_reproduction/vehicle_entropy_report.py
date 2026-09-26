"""Finish the vehicle entropy study using all declared frozen conditions."""
import hashlib
import json

import numpy as np

from runtime import ART, ROOT
from run import write

OUT = ART / 'results/vehicle_entropy'
DEST = ART / 'report/vehicle_entropy'


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(OUT / 'training_completed.json')['complete']
    completed = read(OUT / 'evaluation_completed.json')
    assert completed['complete']
    jobs = completed['jobs']
    expected = {'vehicle_alpha%s_s%d' % (a, s) for a in ['0p1', '0p01'] for s in range(3)}
    expected |= {'vehicle_rl_s%d' % s for s in range(3)}
    expected |= {'vehicle_fixed_h10_s%d' % s for s in range(3)}
    assert len(jobs) == 24
    assert {(j['name'], j['mode']) for j in jobs} == {(n, m) for n in expected for m in ['value', 'no_value']}
    for path, digest in read(OUT / 'source_hashes.json').items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    bank = read(OUT / 'holdout_bank.json')['cases']
    key = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    old = []
    for p in list((ART / 'configs').glob('vehicle_*bank.json')) + [
        ART / 'results/optimized/vehicle_holdout_bank.json',
        ART / 'results/mechanism_probe/vehicle_bank.json',
    ]:
        old.extend(read(p)['cases'])
    assert len(bank) == len(set(map(key, bank))) == 20
    assert not set(map(key, bank)) & set(map(key, old))
    rows, starts, transitions = [], None, 0
    for job in jobs:
        assert job['exit_code'] == 0
        name, mode = job['name'], job['mode']
        folder = OUT / name
        if not (folder / 'manifest.json').exists():
            folder = ART / 'results/optimized' / name
        spec, trained = read(folder / 'manifest.json'), read(folder / 'completed.json')
        assert trained['steps'] == 15000 and trained['updates'] == 14745
        assert spec['config_sha256'] == hashlib.sha256((ART / 'configs/vehicle.json').read_bytes()).hexdigest()
        assert spec['test_bank_sha256'] == hashlib.sha256((ART / 'configs/vehicle_validation_bank.json').read_bytes()).hexdigest()
        ev = OUT / 'evaluations' / name / mode
        summary, frozen = read(ev / 'summary.json'), read(ev / 'completed.json')
        assert frozen['frozen'] and frozen['weights_sha256'] == trained['final_hash']
        assert summary['terminal_value'] == (mode == 'value')
        eps = summary['episodes']
        assert len(eps) == 20 and summary['physical_cost_and_bounds_verified']
        initial = [e['initial_state'] for e in eps]
        if starts is None:
            starts = initial
        assert initial == starts
        for i, ep in enumerate(eps):
            trace = read(ev / ('trace_%02d.json' % i))
            assert len(trace) == ep['steps']
            assert trace[-1]['termination'] == ep['termination']
            for t, row in enumerate(trace):
                # Reset consumes one physical warmup step; logged next state is t+2.
                ref = lambda k: bank[i]['tvp'][k][t + 2]['true'][0]
                s, u = row['state'], row['input']
                performance = (s['x'] - ref('trajectory_x'))**2 + (s['y'] - ref('trajectory_y'))**2
                violation = any(np.hypot(s['x'] - ref('obj_%d_x' % j), s['y'] - ref('obj_%d_y' % j)) <= ref('obj_%d_r' % j) for j in range(3))
                penalty = 2 * (150 - t - 1) if violation else 0.
                assert 1 <= row['horizon'] <= 50 and row['horizon'] == int(row['horizon'])
                assert -1e-5 <= u['u_s'] <= 5 + 1e-5 and abs(u['u_omega']) <= 4 + 1e-5
                assert row['solver_calls'] == 3
                assert np.isclose(performance, row['performance'], rtol=1e-8, atol=1e-8)
                assert np.isclose(row['compute'], .001 * row['horizon'])
                assert np.isclose(row['constraint'], penalty)
                assert np.isclose(row['cost'], performance + .001 * row['horizon'] + penalty)
                if violation:
                    assert t == len(trace) - 1 and row['termination'] == 'constraint'
            for field, source in [('total_cost', 'cost'), ('performance_cost', 'performance'), ('computation_cost', 'compute'), ('constraint_cost', 'constraint')]:
                assert np.isclose(ep[field], sum(r[source] for r in trace))
            assert np.isclose(ep['discounted_cost'], sum(.97**t * r['cost'] for t, r in enumerate(trace)))
            transitions += len(trace)
        assert np.isclose(summary['mean_total_cost'], np.mean([e['total_cost'] for e in eps]))
        group = 'fixed10' if spec['fixed_horizon'] is not None else 'alpha' + str(float(spec['adaptations']['ent_coef']))
        rows.append({'name': name, 'group': group, 'mode': mode, 'seed': spec['seed'],
                     'cost': summary['mean_total_cost'], 'goals': sum(e['termination'] == 'goal' for e in eps),
                     'constraints': sum(e['termination'] == 'constraint' for e in eps),
                     'mean_h': float(np.mean([e['mean_horizon'] for e in eps])),
                     'performance': float(np.mean([e['performance_cost'] for e in eps])),
                     'compute': float(np.mean([e['computation_cost'] for e in eps])),
                     'costs': [e['total_cost'] for e in eps]})
    comparison = {}
    for mode in ['value', 'no_value']:
        fixed = np.array([r['costs'] for r in rows if r['mode'] == mode and r['group'] == 'fixed10'])
        assert fixed.shape == (3, 20)
        comparison[mode] = {}
        for group in ['alpha1.0', 'alpha0.1', 'alpha0.01']:
            selected = [r for r in rows if r['mode'] == mode and r['group'] == group]
            costs = np.array([r['costs'] for r in selected])
            assert costs.shape == (3, 20)
            comparison[mode][group] = {
                'mean': float(costs.mean()), 'seed_means': costs.mean(axis=1).tolist(),
                'fixed_mean': float(fixed.mean()), 'improvement_pct': float(100 * (fixed.mean() - costs.mean()) / abs(fixed.mean())),
                'better_scenes': int(np.sum(costs.mean(axis=0) < fixed.mean(axis=0))),
                'goals': [r['goals'] for r in selected], 'constraints': [r['constraints'] for r in selected],
            }
    DEST.mkdir(exist_ok=True)
    lines = ['# Vehicle entropy refinement', '',
             'All final seeds and both reduced entropy settings are reported. Each new model has 15,000 training transitions. The fixed H10 comparator was selected on the original validation bank.',
             'The 480 frozen evaluations share 20 independent holdout scenes (seed 26091711). This is a disclosed hyperparameter extension, not an exact paper reproduction. Timing under parallel execution is not a speed benchmark.', '',
             '|Terminal|Entropy|RL cost|Fixed H10 cost|RL improvement|Better scenes|Goals per seed|',
             '|---|---:|---:|---:|---:|---:|---|']
    for mode, groups in comparison.items():
        for group, row in groups.items():
            lines.append('|%s|%s|%.4f|%.4f|%.2f%%|%d/20|%s|' % (mode, group[5:], row['mean'], row['fixed_mean'], row['improvement_pct'], row['better_scenes'], row['goals']))
    lines += ['', 'Positive improvement means lower RL cost. Zero-terminal results are frozen interventions, not separately trained agents.', '',
              '|Model|Terminal|Cost|Performance|H cost|Mean H|Goals|Constraints|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|%s|%s|%.4f|%.4f|%.4f|%.2f|%d|%d|' % (r['name'], r['mode'], r['cost'], r['performance'], r['compute'], r['mean_h'], r['goals'], r['constraints']))
    write(OUT / 'summary.json', {'rows': rows, 'comparison': comparison})
    write(OUT / 'audit.json', {'passed': True, 'new_training_runs': 6, 'new_training_transitions': 90000,
                              'evaluated_models': 12, 'episodes': 480, 'independent_scenes': 20,
                              'transitions_recomputed': transitions, 'physical_costs_recomputed': True,
                              'source_hashes_match': True, 'frozen': True, 'starts_match': True})
    (DEST / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
