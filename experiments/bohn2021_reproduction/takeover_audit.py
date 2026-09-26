"""Independently recompute pendulum traces and inventory the resumed studies."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from runtime import ART, ROOT
from run import write


def read(path):
    return json.loads(path.read_text())


def audit_pendulum(group):
    source = ART / 'results' / group
    jobs = read(source / 'training_completed.json')['jobs']
    cases = read(source / 'holdout_bank.json')['cases']
    selected = read(source / 'selected_fixed.json')['h']
    expected = {'pendulum_rl_s%d' % s for s in range(3)}
    expected |= {'pendulum_fixed_h%d_s0' % h for h in range(5, 51, 5)}
    expected |= {'pendulum_fixed_h%d_s%d' % (selected, s) for s in [1, 2]}
    assert len(jobs) == 15 and {j['name'] for j in jobs} == expected
    evaluations = read(source / 'evaluation_completed.json')['jobs']
    assert len(evaluations) == 30 and all(j['exit_code'] == 0 for j in evaluations)
    assert {(j['name'], j['mode']) for j in evaluations} == {(n, m) for n in expected for m in ['value', 'no_value']}
    episodes, transitions = 0, 0
    for job in jobs:
        assert job['exit_code'] == 0
        folder = source / job['name']
        for mode in ['value', 'no_value']:
            ev = folder / ('holdout_' + mode)
            summary = read(ev / 'summary.json')
            for i, ep in enumerate(summary['episodes']):
                trace = read(ev / ('trace_%02d.json' % i))
                for t, row in enumerate(trace):
                    s, u = row['state'], row['input']['u1']
                    reference = cases[i]['tvp']['pos_r'][t + 2]['true'][0]
                    perf = (.4 * s['v']**2 + .05 * s['v'] * s['omega'] * np.cos(s['theta'])
                            + (2 / 3) * .2 * .25**2 * s['omega']**2 - .2 * 9.81 * .25 * np.cos(s['theta'])
                            + 10 * (s['pos'] - reference)**2 + .1 * u**2)
                    violation = abs(s['pos']) > 1.5 or abs(s['theta']) > np.pi / 2
                    penalty = 10 * (100 - t - 1) if violation else 0.
                    assert abs(u) <= 5 + 1e-5
                    assert 1 <= row['horizon'] <= 50 and row['horizon'] == int(row['horizon'])
                    assert row['solver_calls'] == 1
                    assert np.isclose(row['performance'], perf, rtol=1e-8, atol=1e-8)
                    assert np.isclose(row['compute'], .003 * row['horizon'])
                    assert np.isclose(row['constraint'], penalty)
                    assert np.isclose(row['cost'], perf + .003 * row['horizon'] + penalty)
                    if violation:
                        assert t == len(trace) - 1 and row['termination'] == 'constraint'
                assert len(trace) == ep['steps'] and trace[-1]['termination'] == ep['termination']
                assert np.isclose(ep['discounted_cost'], sum(.97**t * r['cost'] for t, r in enumerate(trace)))
                for field, key in [('total_cost', 'cost'), ('performance_cost', 'performance'), ('computation_cost', 'compute'), ('constraint_cost', 'constraint')]:
                    assert np.isclose(ep[field], sum(r[key] for r in trace))
                episodes += 1
                transitions += len(trace)
    assert episodes == 900
    rows = read(source / 'summary.json')['rows']
    rl = [r for r in rows if r['mode'] == 'value' and r['h'] is None]
    fixed = [r for r in rows if r['mode'] == 'value' and r['h'] == selected]
    decomposition = {k: float(np.mean([r[k] for r in rl]) - np.mean([r[k] for r in fixed])) for k in ['cost', 'performance', 'compute', 'mean_h']}
    assert np.isclose(decomposition['cost'], decomposition['performance'] + decomposition['compute'])
    return {'passed': True, 'episodes': episodes, 'physical_transitions_recomputed': transitions,
            'rl_minus_fixed': decomposition, 'comparison': read(source / 'summary.json')['comparison']}


def main():
    out = ART / 'results/takeover_2026-09-18'
    out.mkdir(exist_ok=True)
    groups = {g: audit_pendulum(g) for g in ['recoverable_distribution', 'prior_refinement']}
    vehicle = read(ART / 'results/vehicle_entropy/audit.json')
    assert vehicle['passed']
    groups['vehicle_entropy'] = {'audit': vehicle, 'comparison': read(ART / 'results/vehicle_entropy/summary.json')['comparison']}
    repositories = {}
    for name in ['gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon']:
        folder = ART / 'sources' / name
        status = subprocess.check_output(['git', '-C', str(folder), 'status', '--porcelain'], text=True)
        assert not status.strip(), (name, status)
        repositories[name] = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
    evidence = [ART / 'results' / g / name for g in groups for name in ['summary.json', 'audit.json']]
    interventions = {}
    for group in ['recoverable_distribution', 'prior_refinement']:
        path = ART / 'results/takeover_intervention' / group / 'summary.json'
        result = read(path)
        assert result['complete'] and len(result['rows']) == 3
        assert all(r['replay_exact'] for r in result['rows'])
        interventions[group] = result
        evidence.append(path)
    scripts = ['vehicle_entropy_report.py', 'takeover_intervention.py', 'takeover_audit.py']
    evidence += [ROOT / 'experiments/bohn2021_reproduction' / n for n in scripts]
    record = {
        'complete': True, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source_thread': '01a0ace3-6eec-7fc3-b541-9ad884b3bb8f',
        'new_training_this_takeover': 0, 'previously_finished_training_runs_audited': 36,
        'previously_finished_training_transitions_audited': 540000,
        'holdout_evaluations_audited': 2280, 'independent_holdout_scenes': 80,
        'new_validation_interventions': 60, 'exact_case_replays': 6,
        'groups': groups, 'interventions': interventions, 'author_repositories_clean': repositories,
        'evidence_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        'scope': 'Evidence-backed handoff and follow-up diagnosis. No claim of reproducing the paper advantage.',
    }
    write(out / 'audit.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ['groups', 'interventions', 'evidence_sha256']}, indent=2))


if __name__ == '__main__':
    main()
