"""Post-hoc sufficient angular certificate for every fixed pendulum training case.

Standard library only. No simulation, validation/test access, policy change, or
sample exclusion. Register first, then analyze; registration is immutable.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/gated_horizon_search_2026-09-25'
DEST = OUT / 'training_angular_diagnosis'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
INPUT_TOLERANCE = 1e-5
ANGLE_MARGIN = 1e-10
EXPECTED_RHS = '(M*g*np.sin(theta)-np.cos(theta)*(u1+m*l*omega**2*np.sin(theta)))/((4/3)*M*l-m*l*np.cos(theta)**2)'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name):
    return ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def model():
    config = read(ART / 'configs/pendulum.json')
    plant, mpc = config['plant']['model'], config['mpc']['model']
    assert plant['type'] == mpc['type'] == 'continuous'
    assert plant['parameters'] == mpc['parameters']
    assert plant['states']['theta']['rhs'] == mpc['states']['theta']['rhs'] == 'omega'
    assert plant['states']['omega']['rhs'] == mpc['states']['omega']['rhs'] == EXPECTED_RHS
    p = plant['parameters']
    assert all(p[k] > 0 for k in ('M', 'm', 'l', 'g'))
    bounds = {(r['name'], r['constraint_type']): r['value'] for r in config['mpc']['constraints']}
    force = bounds['u1', 'upper']
    assert force > 0 and bounds['u1', 'lower'] == -force
    angular_bound = bounds['theta', 'upper']
    assert bounds['theta', 'lower'] == -angular_bound
    assert math.isclose(angular_bound, math.pi / 2, rel_tol=0, abs_tol=1e-12)
    assert (4 / 3 * p['M'] - p['m']) * p['l'] > 0
    return dict(parameters=p, force_bound=force, input_tolerance=INPUT_TOLERANCE,
                angle_margin=ANGLE_MARGIN, angular_bound=angular_bound,
                nominal_threshold=math.atan(force / (p['M'] * p['g'])),
                robust_threshold=math.atan((force + INPUT_TOLERANCE) / (p['M'] * p['g'])),
                position_lower=bounds['pos', 'lower'], position_upper=bounds['pos', 'upper'],
                dt=config['plant']['params']['t_step'])


def certificate(state, threshold, angular_bound):
    theta, omega = float(state['theta']), float(state['omega'])
    assert math.isfinite(theta) and math.isfinite(omega)
    return threshold + ANGLE_MARGIN < abs(theta) < angular_bound - ANGLE_MARGIN and theta * omega >= 0


def check_sources():
    paths = {Path(__file__).resolve(), ART / 'configs/pendulum.json',
             ROOT / 'experiments/bohn2021_reproduction/initial_feasibility_audit.py',
             OUT / 'audit_train.json', OUT / 'inputs_sha256.json',
             OUT / 'selected_training_diagnostics/report.json'}
    report = read(OUT / 'selected_training_diagnostics/report.json')
    assert report['passed'] and report['training_only'] and report['completed_fits'] == 6
    assert report['pending'] == [] and read(OUT / 'audit_train.json')['passed']
    cases = []
    for seed in range(3):
        audit = OUT / 'selected_training_diagnostics' / ('pendulum_s%d' % seed) / 'audit.json'
        expected = [h for p, h in report['hashes'].items() if resolve(p) == audit]
        assert expected == [digest(audit)]
        paths.add(audit)
        for name, expected_hash in read(audit)['hashes'].items():
            path = resolve(name)
            assert digest(path) == expected_hash, str(path)
            paths.add(path)
        folder = OUT / 'train' / ('pendulum_s%d' % seed) / 'fixed'
        completion = folder / 'completed.json'
        paths.add(completion)
        completed_hashes = {resolve(p): h for p, h in read(completion)['hashes'].items()}
        for path, expected_hash in completed_hashes.items():
            assert digest(path) == expected_hash, str(path)
            paths.add(path)
        expected_traces = {folder / ('r0_trace_%02d.json' % case) for case in range(24)}
        assert set(folder.glob('r0_trace_*.json')) == expected_traces
        for case in range(24):
            trace = folder / ('r0_trace_%02d.json' % case)
            assert trace in completed_hashes
            cases.append(dict(seed=seed, case=case, trace=relative(trace)))
    return {relative(p): digest(p) for p in sorted(paths)}, cases


def register():
    DEST.mkdir(exist_ok=True)
    target = DEST / 'registration.json'
    hashes, cases = check_sources()
    value = dict(registered_utc=datetime.now(timezone.utc).isoformat(), training_only=True,
        design='Post-hoc descriptive diagnosis registered after exposure to training outcomes; not a confirmatory preregistration.',
        coverage='All 72 fixed-H30 baseline training episodes, three seeds, 24 cases each; no exclusions.',
        state_definition='First scored previous_state after the existing unscored reset/warm-up; not the random generator state.',
        rule='Strictly above atan((umax+1e-5)/(M*g)), strictly below pi/2, outward or zero angular velocity. Also report nominal force-bound sensitivity.',
        source_model=model(), cases=cases, hashes=hashes,
        limits=['Sufficient angular non-recoverability while remaining angular-safe, not a complete viability classification.',
                'No finite-time failure bound, no claim all NLP failures have this cause.',
                'Certificate after step zero does not establish that failure was unavoidable at the episode start.',
                'Uncertified states are unresolved, not declared recoverable.',
                'No change to current policy, distributions, metrics, baseline, validation/test gates, or retained failures.'],
        extra_control_simulation_steps=0, extra_training_steps=0)
    if target.exists():
        existing = read(target)
        assert existing['hashes'] == hashes and existing['cases'] == cases and existing['source_model'] == value['source_model']
        print(json.dumps(dict(reused=True, registration=relative(target), cases=len(cases))))
        return
    write_new(target, value)
    print(json.dumps(dict(registered=True, registration=relative(target), cases=len(cases), sources=len(hashes))))


def scalar(value):
    while isinstance(value, list):
        assert len(value) == 1
        value = value[0]
    value = float(value)
    assert math.isfinite(value)
    return value


def boundary_checks(m):
    t = m['robust_threshold']
    def cert(theta, omega):
        return certificate(dict(theta=theta, omega=omega), t, m['angular_bound'])
    assert not cert(t, 0) and not cert(t + ANGLE_MARGIN / 2, 0)
    assert cert(t + 1e-4, 0) and cert(-t - 1e-4, 0)
    assert cert(t + 1e-4, .1) and cert(-t - 1e-4, -.1)
    assert not cert(t + 1e-4, -.1) and not cert(-t - 1e-4, .1)
    assert not cert(math.pi / 2, 0) and not cert(0, 1)
    p = m['parameters']
    for sign in (-1, 1):
        for theta_abs in (t + 1e-4, (t + math.pi / 2) / 2, math.pi / 2 - 1e-4):
            theta = sign * theta_abs
            for force in (-(m['force_bound'] + INPUT_TOLERANCE), 0, m['force_bound'] + INPUT_TOLERANCE):
                accel = (p['M'] * p['g'] * math.sin(theta) - math.cos(theta) * force) / (4 / 3 * p['M'] * p['l'] - p['m'] * p['l'] * math.cos(theta) ** 2)
                assert sign * accel > 0
    return True


def analyze():
    registration = read(DEST / 'registration.json')
    hashes, cases = check_sources()
    assert hashes == registration['hashes'] and cases == registration['cases']
    m = model()
    assert m == registration['source_model'] and boundary_checks(m)
    if (DEST / 'report.json').exists():
        existing = read(DEST / 'report.json')
        assert existing['registration_sha256'] == digest(DEST / 'registration.json')
        assert digest(DEST / 'all_cases.csv') == existing['all_cases_sha256']
        print(json.dumps(dict(reused=True, groups=existing['groups'], aggregate=existing['aggregate']), indent=2))
        return
    rows = []
    for item in cases:
        trace = read(ROOT / item['trace'])
        assert trace and len(trace) <= 100
        states = [r['previous_state'] for r in trace] + [trace[-1]['state']]
        assert all(a['state'] == b['previous_state'] for a, b in zip(trace, trace[1:]))
        assert all(r['horizon'] == 30 for r in trace)
        controls = [abs(scalar(r['input']['u1'])) for r in trace]
        assert max(controls) <= m['force_bound'] + INPUT_TOLERANCE
        robust = [certificate(s, m['robust_threshold'], m['angular_bound']) for s in states]
        nominal = [certificate(s, m['nominal_threshold'], m['angular_bound']) for s in states]
        initial_failures = [i for i, r in enumerate(trace) if not r['recovery']['attempts'][0]['success']]
        first = next((i for i, flag in enumerate(robust) if flag), None)
        final = states[-1]
        termination = trace[-1]['termination']
        assert termination in ('steps', 'constraint')
        rows.append(dict(seed=item['seed'], case=item['case'], steps=len(trace),
            termination=termination, observed_failed=termination != 'steps',
            start_theta=states[0]['theta'], start_omega=states[0]['omega'],
            certified_at_scored_start=robust[0], nominal_certified_at_scored_start=nominal[0],
            first_certified_state_index=first,
            first_certified_time_seconds=None if first is None else first * m['dt'],
            entered_after_start=first is not None and first > 0,
            ever_certified=any(robust), nominal_ever_certified=any(nominal),
            nominal_robust_disagreements=sum(a != b for a, b in zip(nominal, robust)),
            initial_solver_failed_steps=len(initial_failures),
            first_initial_solver_failure=next(iter(initial_failures), None),
            final_solver_failed_steps=sum(not r['solver_success'] for r in trace),
            final_theta=final['theta'], final_pos=final['pos'],
            final_angular_violation=abs(final['theta']) > m['angular_bound'],
            final_cart_violation=final['pos'] < m['position_lower'] or final['pos'] > m['position_upper'],
            max_abs_input=max(controls), trace=item['trace']))
    assert len(rows) == 72 and len({(r['seed'], r['case']) for r in rows}) == 72
    def summarize(group):
        failed = [r for r in group if r['observed_failed']]
        return dict(cases=len(group), failed=len(failed),
            start_certified=sum(r['certified_at_scored_start'] for r in group),
            failed_start_certified=sum(r['certified_at_scored_start'] for r in failed),
            failed_later_certified=sum(r['entered_after_start'] for r in failed),
            failed_never_certified=sum(not r['ever_certified'] for r in failed),
            successful_ever_certified=sum(r['ever_certified'] for r in group if not r['observed_failed']),
            failed_initial_solver_failure_at_start=sum(r['first_initial_solver_failure'] == 0 for r in failed),
            nominal_robust_disagreements=sum(r['nominal_robust_disagreements'] for r in group),
            failed_angular_violation=sum(r['final_angular_violation'] for r in failed),
            failed_cart_violation=sum(r['final_cart_violation'] for r in failed))
    groups = [dict(seed=seed, **summarize([r for r in rows if r['seed'] == seed])) for seed in range(3)]
    csv_path = DEST / 'all_cases.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value = dict(passed=True, training_only=True, created_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=digest(DEST / 'registration.json'), all_cases_sha256=digest(csv_path),
        source_model=m, boundary_checks_passed=True, groups=groups, aggregate=summarize(rows),
        failures=[dict(seed=r['seed'], case=r['case'], start=r['certified_at_scored_start'],
                       first_certified=r['first_certified_state_index']) for r in rows if r['observed_failed']],
        limits=registration['limits'], extra_control_simulation_steps=0, extra_training_steps=0)
    write_new(DEST / 'report.json', value)
    print(json.dumps(dict(groups=groups, aggregate=value['aggregate'], robust_threshold_degrees=math.degrees(m['robust_threshold'])), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('register', 'analyze'), required=True)
    args = parser.parse_args()
    (register if args.mode == 'register' else analyze)()
