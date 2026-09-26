"""Registered diagnostic policies; no fitted parameters or validation access."""
import math

PENDULUM = ('fixed30', 'certificate5', 'certificate10', 'certificate15')
VEHICLE = ('fixed25', 'fixed30', 'fixed35', 'selected')
THRESHOLD = math.atan((5.0 + 1e-5) / (.8 * 9.81))


def policies(task):
    return PENDULUM if task == 'pendulum' else VEHICLE


def decide(task, policy, ctx, selected=None):
    assert policy in policies(task)
    if policy.startswith('fixed'):
        return int(policy[5:]), dict(rule='fixed', triggered=False)
    if task == 'vehicle':
        from gated_horizon_policy import decide as original
        h, detail = original(selected, ctx)
        return h, dict(rule='inherited_selected_gate', triggered=detail['use_short'], original=detail)
    theta = ctx['state']['theta']
    omega = ctx['state']['omega']
    assert math.isfinite(theta) and math.isfinite(omega)
    flag = THRESHOLD + 1e-10 < abs(theta) < math.pi / 2 - 1e-10 and theta * omega >= 0
    return (int(policy[11:]) if flag else 30), dict(rule='angular_sufficient_condition', triggered=bool(flag))


def check():
    tested = 0
    for short in (5, 10, 15):
        for sign in (-1, 1):
            for angle, velocity, wanted in ((0., 1., False), (THRESHOLD, 0., False),
                                           (THRESHOLD + .01, 0., True), (THRESHOLD + .01, .1, True),
                                           (THRESHOLD + .01, -.1, False), (math.pi / 2, 0., False)):
                h, gate = decide('pendulum', 'certificate%d' % short,
                                 {'state': dict(theta=sign*angle, omega=sign*velocity)})
                assert h == (short if wanted else 30) and gate['triggered'] == wanted
                tested += 1
    for task in ('pendulum', 'vehicle'):
        for policy in policies(task):
            if policy.startswith('fixed'):
                assert decide(task, policy, {})[0] == int(policy[5:])
                tested += 1
    return dict(passed=True, checks=tested, no_simulation=True)
