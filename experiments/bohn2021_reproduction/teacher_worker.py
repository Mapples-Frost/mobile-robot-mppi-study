"""Legacy MPC rollouts and exact-process-state branches for teacher labels."""
import argparse
import copy
import os
import sys
import time
import traceback
import numpy as np
from teacher_common import (OUT, HS, STEPS, GAMMA, OFFSET, SCALE, BRANCH,
    read, write, features, infer, reward_cost, verify)
from optimized_runtime import install_terminal
from forecast_runtime import make_env
from riccati_terminal_probe import prior
from mechanism_probe import checked_step
from plateau_screen import action as rule_action


def context(env):
    c = env.control_system
    refs = c.tvps['pos_r'].get_values(c._step_count, c._step_count+51)
    state = copy.deepcopy(c.current_state)
    return {'state': state, 'refs': refs, 'elapsed': env.steps_count,
            'clock': c._step_count, 'features': features(state, refs, STEPS-env.steps_count).tolist()}


def choose(env, arm, model):
    if model is not None:
        return HS[int(np.argmin(infer(model, context(env)['features'])))]
    return rule_action(env, arm)[0]


def step(env, h, w, b):
    before = context(env)
    _, done, row = checked_step(env, 'pendulum', h)
    after = context(env)
    row.update(env.optimized_solver_info)
    row.update(before=before, after=after, shaped_cost=reward_cost(row, env.steps_count))
    vf = env.control_system.controller.mpc.vf
    np.testing.assert_array_equal(np.asarray(vf.weights_num).ravel(), w.ravel())
    np.testing.assert_array_equal(np.asarray(vf.biases_num).ravel(), b.ravel())
    return row, done


def tail_cost(env, P):
    remaining = STEPS-env.steps_count
    if remaining <= 0:
        return 0.
    c = context(env)
    s = c['state']
    z = np.array([s['omega'], s['pos']-c['refs'][0], s['theta'], s['v']])
    # Stationary-reference local approximation, no entropy and no negative offset.
    return float((z @ P @ z + .003*5*(1-GAMMA**remaining)/(1-GAMMA))/SCALE)


def branch(env, h, w, b, P, path):
    initial = context(env)
    clock = env.control_system._step_count
    tvp = env.control_system.tvps['pos_r']
    final_visible = copy.deepcopy(tvp.values[clock+50])
    for k in range(clock+51, len(tvp.values)):
        tvp.values[k] = copy.deepcopy(final_visible)
    trace = []
    done = False
    for t in range(min(BRANCH, STEPS-env.steps_count)):
        chosen = h if t == 0 else choose(env, 'switch_5_30', None)
        row, done = step(env, chosen, w, b)
        trace.append(row)
        if done:
            break
    tail = 0. if done else tail_cost(env, P)
    cost = sum(GAMMA**t*r['shaped_cost'] for t, r in enumerate(trace)) + GAMMA**len(trace)*tail
    write(path, {'initial': initial, 'h': h, 'trace': trace, 'tail': tail,
                 'discounted_cost': cost, 'hindsight': False})


def fork_branch(env, h, w, b, P, path):
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        try:
            branch(env, h, w, b, P, path)
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (h, status)


def episode(split, index, arm, label=False, smoke=False):
    verify()
    scene = read(OUT/(split+'_bank.json'))['scenes'][index]
    dest = OUT/('labels' if label else 'evaluation')/split/arm/('scene_%02d'%index)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'completed.json').exists():
        return
    model = read(OUT/'models'/(arm+'.json')) if arm.startswith('round') else None
    install_terminal('pendulum')
    env = make_env('pendulum', 26091844)
    env.max_steps = STEPS
    env.config['environment']['max_steps'] = STEPS
    w, b, details = prior()
    w, b = w.astype(np.float32), b.astype(np.float32)
    P = np.asarray(details['P'])
    env.set_value_function_weights_and_biases([w], [b])
    env.reset(**copy.deepcopy(scene['case']))
    initial = context(env)
    trace, anchors = [], []
    started = time.monotonic()
    while True:
        elapsed = env.steps_count
        if label and elapsed in scene['anchors']:
            original = context(env)
            costs = []
            for h in HS:
                path = dest/('branch_t%03d_h%02d.json'%(elapsed, h))
                if not path.exists():
                    fork_branch(env, h, w, b, P, path)
                result = read(path)
                assert result['initial'] == original
                costs.append(result['discounted_cost'])
                assert context(env) == original, 'Child mutated parent state'
            anchors.append({'elapsed': elapsed, 'context': original, 'costs': costs,
                'relative_costs': (np.asarray(costs)-costs[HS.index(30)]).tolist()})
            write(dest/'anchors.json', anchors)
            print('%s scene%d anchor%d/%d elapsed=%d' % (split, index, len(anchors), len(scene['anchors']), elapsed), flush=True)
            if smoke:
                duplicate = dest/'fork_repeat.json'
                fork_branch(env, 30, w, b, P, duplicate)
                a, c = read(dest/('branch_t%03d_h30.json'%elapsed)), read(duplicate)
                for ra, rc in zip(a['trace'], c['trace']):
                    for key in ['state', 'input', 'cost', 'before', 'after', 'shaped_cost']:
                        assert ra[key] == rc[key], key
                assert a['discounted_cost'] == c['discounted_cost']
                row, _ = step(env, 30, w, b)
                for key in ['state', 'input', 'cost']:
                    assert row[key] == a['trace'][0][key], 'Fork differs from original state'
                write(dest/'smoke.json', {'exact_fork_replay': True, 'transitions': 8*BRANCH+1})
                return
        row, done = step(env, choose(env, arm, model), w, b)
        trace.append(row)
        if done:
            break
    write(dest/'trajectory.json', {'initial': initial, 'trace': trace})
    shaped = sum(r['shaped_cost'] for r in trace)
    raw = sum(r['cost'] for r in trace)
    assert np.isclose(shaped*SCALE, raw+OFFSET*STEPS, atol=1e-7)
    summary = {'scene': index, 'arm': arm, 'steps': len(trace), 'termination': trace[-1]['termination'],
        'raw_cost': raw, 'adjusted_cost': shaped*SCALE,
        'discounted_shaped_cost': sum(GAMMA**t*r['shaped_cost'] for t, r in enumerate(trace)),
        'physical_cost': sum(r['performance']+OFFSET for r in trace),
        'H_cost': sum(r['compute'] for r in trace),
        'constraint_cost': sum(r['constraint'] for r in trace),
        'mean_H': float(np.mean([r['horizon'] for r in trace])),
        'tracking_rmse': float(np.sqrt(np.mean([(r['state']['pos']-r['after']['refs'][0])**2 for r in trace]))),
        'solver_failures': sum(not r['solver_success'] for r in trace),
        'anchors': len(anchors), 'elapsed_s': time.monotonic()-started}
    write(dest/'completed.json', summary)
    print(str(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--scene', type=int, required=True)
    ap.add_argument('--arm', default='switch_5_30')
    ap.add_argument('--label', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    episode(a.split, a.scene, a.arm, a.label, a.smoke)
