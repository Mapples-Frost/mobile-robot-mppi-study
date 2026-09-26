"""Causal depth-two horizon trees and categorical cross-entropy policy search.

Parameters are learned from whole closed-loop returns, without an oracle policy,
critic, certificate threshold, transformed return or off-policy continuation.
"""
import copy
import hashlib
import json
import math
import numpy as np

HS = tuple(range(5, 51, 5))
BASE = {'vehicle': 25, 'pendulum': 30}
FEATURES = {
    'pendulum': ('abs_angle', 'outward_angular_velocity', 'abs_velocity', 'position_error',
                 'abs_input', 'reference_range_10', 'reference_range_30', 'remaining',
                 'previous_initial_failure', 'previous_final_failure'),
    'vehicle': ('tracking_error', 'heading_error_5', 'heading_error_15', 'heading_error_30',
                'reference_turn_30', 'current_clearance', 'preview_clearance_15', 'preview_clearance_30',
                'speed', 'abs_yaw_input', 'remaining', 'previous_initial_failure', 'previous_final_failure')}


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def features(task, ctx):
    s, p, u = ctx['state'], ctx['previews'], ctx['previous_input']
    failure = [float(ctx.get('previous_initial_failure', False)), float(ctx.get('previous_final_failure', False))]
    remaining = ((100 if task == 'pendulum' else 150) - ctx['elapsed']) / (100. if task == 'pendulum' else 150.)
    if task == 'pendulum':
        ref = np.asarray(p['pos_r'], dtype=float)
        x = [abs(s['theta']), (1. if s['theta'] >= 0 else -1.) * s['omega'], abs(s['v']),
             abs(s['pos'] - ref[0]), abs(scalar(u['u1'])),
             float(np.ptp(ref[:11])), float(np.ptp(ref[:31])), remaining] + failure
    else:
        assert task == 'vehicle'
        xy = np.column_stack([p['trajectory_x'], p['trajectory_y']]).astype(float)
        pos = np.array([s['x'], s['y']], dtype=float)
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        headings = [math.atan2(*(xy[k] - xy[0])[::-1]) for k in (5, 15, 30)]
        current, ahead15, ahead30 = [], [], []
        for j in range(len(ctx['noise'])):
            obj = np.column_stack([p['obj_%d_x' % j], p['obj_%d_y' % j]]).astype(float)
            rad = np.asarray(p['obj_%d_r' % j], dtype=float)
            current.append(float(np.linalg.norm(pos - obj[0]) - 1.5 * rad[0]))
            distances = np.linalg.norm(xy - obj, axis=1) - 1.5 * rad
            ahead15.append(float(distances[:16].min())); ahead30.append(float(distances[:31].min()))
        assert current, 'Vehicle object forecasts must be present'
        x = [float(np.linalg.norm(pos - xy[0]))] + [abs(wrap(a - s['theta'])) for a in headings]
        x += [abs(wrap(headings[-1] - headings[0])), min(current), min(ahead15), min(ahead30),
              scalar(u['u_s']), abs(scalar(u['u_omega'])), remaining] + failure
    assert len(x) == len(FEATURES[task]) and all(math.isfinite(v) for v in x)
    return [float(v) for v in x]


def constant(task, h=None):
    return dict(kind='constant', task=task, h=BASE[task] if h is None else int(h))


def policy_key(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def choose(policy, ctx):
    task = policy['task']
    if policy['kind'] == 'constant':
        assert policy['h'] in HS
        return policy['h'], dict(kind='constant')
    if policy['kind'] == 'certificate':
        assert task == 'pendulum'
        from failure_state_policy import decide
        return decide(task, 'certificate5', ctx)
    if policy['kind'] == 'combined':
        from failure_state_policy import decide
        h, gate = decide(task, 'certificate5', ctx)
        if gate['triggered']:
            return h, dict(kind='combined', override=True)
        h, details = choose(policy['learned'], ctx)
        return h, dict(kind='combined', override=False, learned=details)
    assert policy['kind'] == 'tree' and len(policy['nodes']) == 3 and len(policy['leaves']) == 4
    values = features(task, ctx)
    root = policy['nodes'][0]; side = int(values[root['feature']] > root['threshold'])
    node = policy['nodes'][1 + side]
    leaf = 2 * side + int(values[node['feature']] > node['threshold'])
    h = policy['leaves'][leaf]
    assert h in HS
    return h, dict(kind='tree', features=values, leaf=leaf)


def thresholds(task, episode_features):
    """Thresholds use new fit-bank fixed-policy states only, including failures."""
    data = np.asarray([x for scene in episode_features for x in scene], dtype=float)
    assert data.ndim == 2 and data.shape[1] == len(FEATURES[task]) and np.isfinite(data).all()
    values = []
    for j, name in enumerate(FEATURES[task]):
        if name.startswith('previous_'):
            values.append([.5])
        else:
            values.append(sorted(set(float(v) for v in np.quantile(data[:, j], [.1, .25, .5, .75, .9]))))
    return values


def initialize(task, cuts):
    nf = len(FEATURES[task])
    a = np.full(len(HS), .3 / (len(HS) - 1)); a[HS.index(BASE[task])] = .7
    return dict(feature=[[1. / nf] * nf for _ in range(3)],
                cut=[[[1. / len(c)] * len(c) for c in cuts] for _ in range(3)],
                action=[a.tolist() for _ in range(4)])


def sample(task, cuts, distribution, rng):
    nodes, genome = [], []
    for i in range(3):
        f = int(rng.choice(len(FEATURES[task]), p=distribution['feature'][i]))
        q = int(rng.choice(len(cuts[f]), p=distribution['cut'][i][f]))
        genome.append([f, q]); nodes.append(dict(feature=f, threshold=cuts[f][q]))
    acts = [int(rng.choice(len(HS), p=p)) for p in distribution['action']]
    policy = dict(kind='tree', task=task, nodes=nodes, leaves=[HS[a] for a in acts])
    return policy, dict(nodes=genome, actions=acts)


def update(distribution, elites):
    """Smoothed categorical CE update with 10% uniform exploration floor."""
    assert elites
    result = copy.deepcopy(distribution)
    def blended(old, indices):
        if not indices:
            return list(old)
        counts = np.bincount(indices, minlength=len(old)).astype(float) / len(indices)
        result = .4 * np.asarray(old) + .6 * counts
        result = .9 * result + .1 / len(old)
        return (result / result.sum()).tolist()
    for i in range(3):
        result['feature'][i] = blended(distribution['feature'][i], [g['nodes'][i][0] for g in elites])
        for f in range(len(distribution['feature'][i])):
            result['cut'][i][f] = blended(distribution['cut'][i][f], [g['nodes'][i][1] for g in elites if g['nodes'][i][0] == f])
    for i in range(4):
        result['action'][i] = blended(distribution['action'][i], [g['actions'][i] for g in elites])
    return result


def aggregate(episodes):
    assert episodes
    totals = {name: sum(int(e[name]) for e in episodes) for name in
              ('steps', 'success', 'constraint', 'initial_failed_steps', 'solver_failure_steps')}
    totals.update({name: math.fsum(e[name] for e in episodes) / len(episodes) for name in
                   ('total_cost', 'physical_constraint_cost')})
    totals['decision_mean_s'] = math.fsum(e['decision_total_s'] for e in episodes) / totals['steps']
    return totals


def rank(episodes, baseline):
    """Training-only lexicographic safety/NI ranking, then raw cost + measured time."""
    assert len(episodes) == len(baseline) and [e['case'] for e in episodes] == [e['case'] for e in baseline]
    a, b = aggregate(episodes), aggregate(baseline)
    assert abs(b['total_cost']) > 1e-12 and abs(b['physical_constraint_cost']) > 1e-12 and b['decision_mean_s'] > 0
    violations = sum(int(e['success'] < r['success']) + int(e['constraint'] > r['constraint']) for e, r in zip(episodes, baseline))
    for name in ('initial_failed_steps', 'solver_failure_steps'):
        violations += int(a[name] * b['steps'] > b[name] * a['steps'])
    physical = (a['physical_constraint_cost'] - b['physical_constraint_cost']) / abs(b['physical_constraint_cost'])
    cost = (a['total_cost'] - b['total_cost']) / abs(b['total_cost'])
    violations += int(physical > .02) + int(cost > .02)
    # A per-case cap guards against sacrificing one scene for average gains.
    violations += sum(int(e['physical_constraint_cost'] > r['physical_constraint_cost'] + .05 * abs(r['physical_constraint_cost']) + 1e-8)
                      for e, r in zip(episodes, baseline))
    objective = cost + .5 * (a['decision_mean_s'] / b['decision_mean_s'] - 1.)
    return dict(eligible=violations == 0, violations=violations, objective=objective,
                cost_change=cost, physical_change=physical, time_ratio=a['decision_mean_s'] / b['decision_mean_s'])
