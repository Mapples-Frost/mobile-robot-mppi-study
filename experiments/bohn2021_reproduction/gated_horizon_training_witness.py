"""Bounded training-only search for numerically feasible saturated-LQR trajectories.

No reference-tracking or adaptive-H efficacy claim. Every registered combination
is retained, including failures. A failure to find a witness is inconclusive.
"""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
RUN = ART / 'results/gated_horizon_search_2026-09-25'
SOURCE = RUN / 'training_angular_diagnosis'
OUT = RUN / 'training_feasible_witness'
STATE_NAMES = ('pos', 'v', 'theta', 'omega')
DT, SUBSTEPS, STEPS = .04, 20, 100
FORCE, POSITION, ANGLE = 5., 1.5, math.pi / 2


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rhs(y, u):
    pos, velocity, theta, omega = y
    sn, cs = math.sin(theta), math.cos(theta)
    acceleration = (.2 * 9.81 * sn * cs - (4 / 3) * (u + .2 * .25 * omega**2 * sn)) / (.2 * cs**2 - (4 / 3) * .8)
    angular = (.8 * 9.81 * sn - cs * (u + .2 * .25 * omega**2 * sn)) / ((4 / 3) * .8 * .25 - .2 * .25 * cs**2)
    return np.array([velocity, acceleration, omega, angular])


def step(y, u):
    z = y.copy()
    points = [z.copy()]
    h = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        a = rhs(z, u)
        b = rhs(z + h * a / 2, u)
        c = rhs(z + h * b / 2, u)
        d = rhs(z + h * c, u)
        z = z + h * (a + 2 * b + 2 * c + d) / 6
        points.append(z.copy())
    return z, np.asarray(points)


def linear_model():
    A = np.array([[0, 1, 0, 0], [0, 0, -.2 * 9.81 / (4 * .8 / 3 - .2), 0],
                  [0, 0, 0, 1], [0, 0, .8 * 9.81 / ((4 * .8 / 3 - .2) * .25), 0]], dtype=float)
    B = np.array([[0], [(4 / 3) / (4 * .8 / 3 - .2)], [0], [-1 / ((4 * .8 / 3 - .2) * .25)]])
    for i in range(4):
        unit = np.eye(4)[i] * 1e-6
        np.testing.assert_allclose((rhs(unit, 0) - rhs(-unit, 0)) / 2e-6, A[:, i], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose((rhs(np.zeros(4), 1e-6) - rhs(np.zeros(4), -1e-6)) / 2e-6, B[:, 0], rtol=1e-9, atol=1e-9)
    return A, B


def controllers():
    A, B = linear_model()
    result = []
    for index, (qpos, qtheta, r) in enumerate(itertools.product((.1, 1., 10.), (10., 100., 1000.), (.01, .1, 1.))):
        Q = np.diag([qpos, 1., qtheta, 1.])
        P = solve_continuous_are(A, B, Q, np.array([[r]]))
        K = (B.T @ P / r).ravel()
        residual = A.T @ P + P @ A - P @ B @ B.T @ P / r + Q
        assert np.max(np.abs(residual)) < 1e-6
        assert max(np.linalg.eigvals(A - B @ K[None, :]).real) < 0
        for target in ('zero', 'initial_position'):
            result.append(dict(id='lqr%02d_%s' % (index, target), Q=np.diag(Q).tolist(), R=r, K=K.tolist(), target=target))
    assert len(result) == 54
    return result


def verify():
    reg = read(OUT / 'registration.json')
    for name, value in reg['hashes'].items():
        assert digest(ROOT / name) == value, name
    return reg


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'registration.json').exists():
        reg = verify()
        print(json.dumps(dict(reused=True, cases=len(reg['cases']), controllers=len(reg['controllers']))))
        return
    config = read(ART / 'configs/pendulum.json')
    assert config['plant']['model']['parameters'] == dict(M=.8, m=.2, l=.25, g=9.81)
    assert config['plant']['params']['t_step'] == DT and config['environment']['max_steps'] == STEPS
    bounds = {(r['name'], r['constraint_type']): r['value'] for r in config['mpc']['constraints']}
    assert bounds['u1', 'upper'] == FORCE == -bounds['u1', 'lower']
    assert bounds['pos', 'upper'] == POSITION == -bounds['pos', 'lower']
    assert bounds['theta', 'upper'] == ANGLE == -bounds['theta', 'lower']
    a = read(SOURCE / 'independent_check.json')
    assert a['passed']
    for name, value in a['inputs_sha256'].items():
        assert digest(SOURCE / name) == value
    cases = []
    paths = [Path(__file__).resolve(), ART / 'configs/pendulum.json', SOURCE / 'registration.json', SOURCE / 'report.json', SOURCE / 'all_cases.csv', SOURCE / 'independent_check.json']
    with (SOURCE / 'all_cases.csv').open(newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    assert len(all_rows) == 72
    for row in all_rows:
        if row['observed_failed'] != 'True' or row['certified_at_scored_start'] != 'False':
            continue
        source = ROOT / row['trace']
        trace = read(source)
        state = trace[0]['previous_state']
        paths.append(source)
        cases.append(dict(seed=int(row['seed']), case=int(row['case']), initial=[state[k] for k in STATE_NAMES],
                          baseline_steps=len(trace), baseline_termination=trace[-1]['termination'], trace=row['trace']))
    assert len(cases) == 15
    value = dict(training_only=True, registered_at=time.time(),
        question='Can a predeclared finite set of input-bounded state-feedback controllers produce a 100-step state-feasible trajectory from any of the 15 training failures lacking the initial angular certificate?',
        design='Post-hoc mechanism probe, full Cartesian product of all 15 previously identified cases and 54 declared controllers; no favorable case or controller omission.',
        cases=cases, controllers=controllers(), dt=DT, steps=STEPS, substeps=SUBSTEPS,
        state_order=list(STATE_NAMES), input_bound=FORCE, position_bound=POSITION, angle_bound=ANGLE,
        candidate_control='u=clip(-K @ (state-[target,0,0,0]), -5, 5), recomputed each 0.04 seconds; target is zero or the original scored initial cart position. No reference tracking claim.',
        simulation='RK4 with 20 uniform substeps per held control. Stop after first scored state constraint violation. Save every completed step, sample envelope and control.',
        independent_audit='Reintegrate every recorded control from the same initial state with DOP853 rtol=atol=1e-12; require state agreement rtol=atol=1e-7. Check all 21 evaluation samples per step; no claim these prove continuous-time constraint invariance.',
        witness_acceptance='Exactly 100 steps with |u|<=5, no sampled state violation, and independent DOP853 replay minimum margin >1e-6. Report margins and final state. Survival is not upright regulation or tracking performance.',
        selection='All candidate outcomes reported. Earliest registered independently verified candidate per case is only a reproducible representative witness; no learned strategy or efficacy selection.',
        budget='Count every diagnostic environment-step attempt, RK4 RHS evaluation and independent integration; separate from the frozen adaptive-H experiment. No optimizer/NLP calls or training updates.',
        limits=['No found witness is not an infeasibility certificate.', 'This feedback probe is not the paper algorithm or an adaptive-H improvement.',
                'No new validation/test or change to current gate policies, conditions or thresholds.', 'The scored start is after existing reset warmup; no new warmup is executed.'],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    write(OUT / 'registration.json', value)
    print(json.dumps(dict(registered=True, cases=15, controllers=54, conditions=810)))


def smoke():
    reg = verify()
    target = OUT / 'smoke.json'
    if target.exists():
        old = read(target)
        assert old['passed'] and old['source_hash'] == digest(Path(__file__))
        print(json.dumps(dict(reused=True, smoke_passed=True, maximum_error=old['maximum_state_error'])))
        return
    cases = [[0., 0., .05, 0.], [.2, -.1, -.1, .1]]
    K = np.array(reg['controllers'][0]['K'])
    max_error = 0.
    for initial in cases:
        y = np.array(initial)
        for _ in range(10):
            u = float(np.clip(-K @ y, -FORCE, FORCE))
            z, points = step(y, u)
            sol = solve_ivp(lambda t, s: rhs(s, u), (0, DT), y, method='DOP853', rtol=1e-12, atol=1e-12, t_eval=np.linspace(0, DT, SUBSTEPS + 1))
            assert sol.success
            error = float(np.max(abs(points - sol.y.T)))
            max_error = max(max_error, error)
            np.testing.assert_allclose(points, sol.y.T, rtol=1e-8, atol=1e-8)
            assert abs(z[0]) < POSITION and abs(z[2]) < ANGLE and abs(u) <= FORCE
            y = z
    write(target, dict(passed=True, synthetic_cases=cases, environment_steps=20, rk4_rhs_evaluations=1600,
                      independent_integration_calls=20, maximum_state_error=max_error, source_hash=digest(Path(__file__))))
    print(json.dumps(dict(smoke_passed=True, maximum_error=max_error)))


def run():
    reg = verify()
    assert read(OUT / 'smoke.json')['passed']
    lock = (OUT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = OUT / 'status.json'
    if status_path.exists():
        prior = read(status_path)
        if prior.get('complete'):
            for name, value in prior['output_hashes'].items():
                assert digest(OUT / name) == value
            print('Completed outputs verified; no repeated simulation')
            return
        raise RuntimeError('Prior attempt exists; audit interruption before restart')
    state = dict(pid=os.getpid(), active=True, complete=False, started=time.time(), step_attempts=0, step_completed=0,
                 rk4_rhs_evaluations=0, completed_conditions=0, condition=None, conditions=810, stage='simulation')
    write(status_path, state)
    folder = OUT / 'traces'
    folder.mkdir(exist_ok=True)
    results = []
    try:
        for case in reg['cases']:
            for controller in reg['controllers']:
                name = 's%d_c%02d_%s' % (case['seed'], case['case'], controller['id'])
                state['condition'] = name
                y = np.array(case['initial'], dtype=float)
                K = np.array(controller['K'])
                target = 0. if controller['target'] == 'zero' else float(y[0])
                trajectory = []
                for t in range(STEPS):
                    u = float(np.clip(-K @ (y - np.array([target, 0, 0, 0])), -FORCE, FORCE))
                    state['step_attempts'] += 1
                    write(status_path, state)
                    z, points = step(y, u)
                    state['step_completed'] += 1
                    state['rk4_rhs_evaluations'] += 4 * SUBSTEPS
                    row = dict(step=t, state=y.tolist(), input=u, next_state=z.tolist(),
                               substep_max_abs_pos=float(np.max(abs(points[:, 0]))), substep_max_abs_theta=float(np.max(abs(points[:, 2]))))
                    trajectory.append(row)
                    with (folder / (name + '.jsonl')).open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                    y = z
                    if abs(y[0]) > POSITION or abs(y[2]) > ANGLE:
                        break
                path = folder / (name + '.json')
                write(path, dict(case=case, controller=controller['id'], target=target, rows=trajectory))
                max_pos = max(r['substep_max_abs_pos'] for r in trajectory)
                max_angle = max(r['substep_max_abs_theta'] for r in trajectory)
                results.append(dict(seed=case['seed'], case=case['case'], controller=controller['id'], steps=len(trajectory),
                    full_horizon=len(trajectory) == STEPS, sampled_feasible=len(trajectory) == STEPS and max_pos <= POSITION and max_angle <= ANGLE,
                    max_abs_pos=max_pos, max_abs_theta=max_angle, final_state=y.tolist(), path=path.relative_to(OUT).as_posix(), sha256=digest(path)))
                state['completed_conditions'] += 1
                write(status_path, state)
            print(json.dumps(dict(case=[case['seed'], case['case']], completed_conditions=state['completed_conditions'])), flush=True)
        assert len(results) == 810
        write(OUT / 'results.json', dict(registration_hash=digest(OUT / 'registration.json'), results=results))
        state.update(active=False, complete=True, ended=time.time(), condition=None,
                     output_hashes={'results.json':digest(OUT / 'results.json')})
        write(status_path, state)
        print(json.dumps(dict(complete=True, steps=state['step_completed'], candidates_surviving=sum(r['sampled_feasible'] for r in results))))
    except BaseException as exc:
        state.update(active=False, complete=False, ended=time.time(), exception=repr(exc))
        write(status_path, state)
        raise


def audit():
    reg = verify()
    state = read(OUT / 'status.json')
    assert state['complete'] and not state['active']
    results_path = OUT / 'results.json'
    assert digest(results_path) == state['output_hashes']['results.json']
    results = read(results_path)['results']
    controllers_by_id = {c['id']:c for c in reg['controllers']}
    expected = {(c['seed'], c['case'], p['id']) for c in reg['cases'] for p in reg['controllers']}
    assert len(results) == len(expected) == 810
    assert {(r['seed'], r['case'], r['controller']) for r in results} == expected
    if (OUT / 'audit.json').exists():
        prior = read(OUT / 'audit.json')
        assert prior['results_hash'] == digest(results_path) and prior['source_hash'] == digest(Path(__file__))
        for result in results:
            assert digest(OUT / result['path']) == result['sha256']
        print(json.dumps(dict(reused=True, totals=prior['totals'], cases=prior['cases']), indent=2))
        return
    meter = dict(pid=os.getpid(), active=True, started=time.time(), conditions=0, integrations_attempted=0, integrations_completed=0, rhs_evaluations=0)
    write(OUT / 'audit_status.json', meter)
    audited = []
    try:
        for result in results:
            path = OUT / result['path']
            assert digest(path) == result['sha256']
            trace = read(path)
            rows_journal = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
            assert rows_journal == trace['rows']
            case = next(c for c in reg['cases'] if (c['seed'], c['case']) == (result['seed'], result['case']))
            assert trace['case'] == case
            K = np.array(controllers_by_id[result['controller']]['K'])
            y = np.array(case['initial'], dtype=float)
            target = np.array([trace['target'], 0, 0, 0])
            max_error, max_pos, max_angle = 0., abs(y[0]), abs(y[2])
            for index, row in enumerate(trace['rows']):
                assert row['step'] == index and abs(row['input']) <= FORCE
                np.testing.assert_allclose(y, row['state'], rtol=1e-7, atol=1e-7)
                u = float(np.clip(-K @ (np.array(row['state']) - target), -FORCE, FORCE))
                assert row['input'] == u
                meter['integrations_attempted'] += 1
                write(OUT / 'audit_status.json', meter)
                sol = solve_ivp(lambda t, s: rhs(s, u), (0, DT), y, method='DOP853', rtol=1e-12, atol=1e-12, t_eval=np.linspace(0, DT, SUBSTEPS + 1))
                assert sol.success
                meter['integrations_completed'] += 1
                meter['rhs_evaluations'] += sol.nfev
                y = sol.y[:, -1]
                max_error = max(max_error, float(np.max(abs(y - row['next_state']))))
                np.testing.assert_allclose(y, row['next_state'], rtol=1e-7, atol=1e-7)
                max_pos = max(max_pos, float(np.max(abs(sol.y[0]))))
                max_angle = max(max_angle, float(np.max(abs(sol.y[2]))))
            assert len(trace['rows']) == result['steps']
            margin = min(POSITION - max_pos, ANGLE - max_angle)
            witness = result['steps'] == STEPS and margin > 1e-6
            assert not witness or result['sampled_feasible']
            audited.append(dict(seed=result['seed'], case=result['case'], controller=result['controller'], steps=result['steps'],
                numerical_witness=witness, min_sampled_margin=margin, max_abs_pos=max_pos, max_abs_theta=max_angle,
                final_state=y.tolist(), max_state_discrepancy=max_error, trace=result['path']))
            meter['conditions'] += 1
            write(OUT / 'audit_status.json', meter)
        cases = []
        for c in reg['cases']:
            group = [r for r in audited if (r['seed'], r['case']) == (c['seed'], c['case'])]
            feasible = [r for r in group if r['numerical_witness']]
            assert len(group) == 54
            cases.append(dict(seed=c['seed'], case=c['case'], candidates=54, witness_count=len(feasible),
                representative=feasible[0] if feasible else None, best_recorded_survival_steps=max(r['steps'] for r in group), baseline_steps=c['baseline_steps']))
        assert sum(r['steps'] for r in audited) == state['step_completed'] == state['step_attempts'] == meter['integrations_completed']
        value = dict(passed=True, training_only=True, created=time.time(), results_hash=digest(results_path), registration_hash=digest(OUT / 'registration.json'), source_hash=digest(Path(__file__)),
            totals=dict(conditions=810, cases=15, cases_with_witness=sum(c['witness_count'] > 0 for c in cases),
                candidate_witnesses=sum(r['numerical_witness'] for r in audited), simulation_steps=state['step_completed'],
                rk4_rhs_evaluations=state['rk4_rhs_evaluations'], independent_integrations=meter['integrations_completed'],
                independent_rhs_evaluations=meter['rhs_evaluations'], smoke_steps=read(OUT / 'smoke.json')['environment_steps'],
                max_state_discrepancy=max(r['max_state_discrepancy'] for r in audited)), cases=cases, all_candidates=audited,
            limits=reg['limits'] + ['Sampled numerical feasibility, not a rigorous continuous-time certificate.'])
        write(OUT / 'audit.json', value)
        meter.update(active=False, complete=True, ended=time.time())
        write(OUT / 'audit_status.json', meter)
        print(json.dumps(dict(totals=value['totals'], cases=cases), indent=2))
    except BaseException as exc:
        meter.update(active=False, complete=False, ended=time.time(), exception=repr(exc))
        write(OUT / 'audit_status.json', meter)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('register', 'smoke', 'run', 'audit'), required=True)
    globals()[parser.parse_args().mode]()
