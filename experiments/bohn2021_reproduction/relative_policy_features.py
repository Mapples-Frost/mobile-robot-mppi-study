"""Read-only causal features for the next policy-improvement experiment.

Queries exactly the forecast API used by ControlSystem.step, at most 50 steps
ahead. Never generates TVPs or queries an individual future true-value entry.
This module does not change the plant, reward, terminal value, or controller.
"""
import copy
import numpy as np


def encode(task, state, previews, remaining, previous_h, previous_input, noise):
    x = [remaining, previous_h / 50.]
    if task == 'pendulum':
        refs = np.asarray(previews['pos_r'])
        assert refs.shape == (51,)
        relative = (refs-state['pos']) / 1.5
        changes = np.flatnonzero(abs(refs[1:]-refs[0]) > 1e-8)
        first = int(changes[0])+1 if len(changes) else 50
        x += [state['pos']/1.5, state['v']/5., state['theta']/(np.pi/2),
              state['omega']/10., float(np.asarray(previous_input['u1']).ravel()[0])/5., relative[0],
              first/50., (refs[first]-refs[0])/1.5]
        # Full preview preserves the timing of redraws, which coarse averaging loses.
        x += relative[1:].tolist()
    else:
        assert task == 'vehicle'
        theta = state['theta']
        c, s = np.cos(theta), np.sin(theta)
        def body(dx, dy, scale):
            return [(c*dx+s*dy)/scale, (-s*dx+c*dy)/scale]
        x += [np.sin(theta), np.cos(theta)]
        x += [float(np.asarray(previous_input[k]).ravel()[0])/(5. if k=='u_s' else 4.) for k in sorted(previous_input)]
        for k in (0, 5, 10, 20, 25, 30, 40, 50):
            x += body(previews['trajectory_x'][k]-state['x'],
                      previews['trajectory_y'][k]-state['y'], 10.)
        obstacles = []
        for j in range(len(noise)):
            dx = previews['obj_%d_x' % j][0]-state['x']
            dy = previews['obj_%d_y' % j][0]-state['y']
            r = previews['obj_%d_r' % j][0]
            rel = body(dx, dy, 10.)
            # Noise seeds are known controller forecast parameters, not outcomes.
            rel += [r/2., (np.hypot(dx,dy)-r)/10.] + list(np.asarray(noise[j])/[5.,5.,1.])
            obstacles.append((np.hypot(dx,dy), j, rel))
        for _, _, obstacle in sorted(obstacles): x += obstacle
    arr = np.asarray(x, dtype=np.float32)
    assert np.isfinite(arr).all()
    return arr


def context(env, task):
    c = env.control_system
    clock = c._step_count
    assert clock == env.steps_count+1
    names = ['pos_r'] if task == 'pendulum' else ['trajectory_x','trajectory_y'] + [
        'obj_%d_%s' % (j, a) for j in range(c.controller.n_objects) for a in ('x','y','r')]
    previews = {}
    for name in names:
        tvp = c.tvps[name]
        assert len(tvp.values) >= clock+51, 'Build a complete scenario bank before rollout; feature reads never consume RNG'
        previews[name] = tvp.get_values(clock, clock+51, with_noise=True)
    noise = copy.deepcopy(c.controller.object_noise_seed) if task == 'vehicle' else []
    prev_h = c.controller.history['mpc_horizon'][-1]
    previous_input = copy.deepcopy(c.controller.current_input)
    state = copy.deepcopy(c.current_state)
    remaining = (env.max_steps-env.steps_count)/env.max_steps
    return dict(clock=clock, elapsed=env.steps_count, state=state, previews=previews,
        noise=noise, previous_h=prev_h, previous_input=previous_input,
        features=encode(task,state,previews,remaining,prev_h,previous_input,noise).tolist())
