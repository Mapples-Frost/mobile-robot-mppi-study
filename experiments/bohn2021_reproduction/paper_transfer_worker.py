"""Legacy MPC evaluations of frozen, portable horizon policies."""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'icra_scene_design'))
from sweep_guard import require_no_live_sweep

from paper_transfer_common import OUT, PRESERVE, TEACHER, LEARNED, read, write, verify, metrics


def policy(name):
    import numpy as np
    from sac_preserve_inference import HorizonPolicy, encode
    from teacher_common import HS, infer
    from plateau_screen import action
    if name in LEARNED:
        actor = HorizonPolicy(PRESERVE/'export'/(name+'.json'))
        return lambda env, obs: actor.predict(obs)
    if name.startswith('fixed_'):
        h = int(name.split('_')[1])
        return lambda env, obs: h
    if name == 'rule_baseline':
        return lambda env, obs: action(env, 'switch_5_30')[0]
    if name == 'value_teacher':
        teachers = [read(TEACHER/'models'/('round1_s%d.json' % s)) for s in range(3)]
        return lambda env, obs: HS[int(np.argmin(np.mean([infer(m, encode(obs)) for m in teachers], axis=0)))]
    raise ValueError(name)


def environment():
    from optimized_runtime import install_terminal
    from forecast_runtime import make_env
    install_terminal('pendulum')
    return make_env('pendulum', 260923411)


def run(name, domain, split):
    require_no_live_sweep('paper transfer MPC evaluation')
    import numpy as np
    from riccati_terminal_probe import prior
    from mechanism_probe import checked_step
    verify()
    folder = OUT/'evaluation'/split/domain/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        return
    choose = policy(name)
    summaries = []
    for scene in read(OUT/(split+'_bank.json'))['domains'][domain]:
        target = folder/('scene_%02d.json' % scene['id'])
        if target.exists():
            summaries.append(read(target)['summary'])
            continue
        require_no_live_sweep('paper transfer MPC scene')
        attempt = folder/('attempt_scene%02d_%d_%d.json' % (scene['id'], os.getpid(), time.time_ns()))
        write(attempt, {'scene': scene['id'], 'status': 'started', 'observed_steps': 0,
                        'accounting': 'Progress is an observed lower bound if interrupted.'})
        env = environment()
        try:
            T = scene['steps']
            env.max_steps = T
            env.config['environment']['max_steps'] = T
            w, b, _ = prior()
            w, b = w.astype(np.float32), b.astype(np.float32)
            env.set_value_function_weights_and_biases([w], [b])
            obs = env.reset(**copy.deepcopy(scene['case']))
            initial = obs.tolist()
            trace = []
            for t in range(1, T+1):
                h = int(choose(env, obs))
                before = obs.tolist()
                obs, done, row = checked_step(env, 'pendulum', h)
                row.update(step=t, obs=before, next_obs=obs.tolist(), done=bool(done))
                trace.append(row)
                assert abs(float(obs[-1])-(T-t)/T) < 1e-6
                np.testing.assert_array_equal(np.asarray(env.control_system.controller.mpc.vf.weights_num).ravel(), w.ravel())
                np.testing.assert_array_equal(np.asarray(env.control_system.controller.mpc.vf.biases_num).ravel(), b.ravel())
                if t % 25 == 0 or done:
                    write(attempt, {'scene': scene['id'], 'status': 'running', 'observed_steps': t,
                                    'reset_warmups': 1})
                if done:
                    break
            assert trace[-1]['done']
            summary = dict(metrics(trace, T), scene=scene['id'], name=name, domain=domain)
            write(target, {'initial': initial, 'summary': summary, 'trace': trace})
            write(attempt, {'scene': scene['id'], 'status': 'complete', 'observed_steps': len(trace),
                            'reset_warmups': 1, 'result': target.name})
            summaries.append(summary)
            print(split, domain, name, scene['id'], 'complete', flush=True)
        finally:
            env.close()
    write(folder/'completed.json', {'episodes': summaries})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args()
    run(args.name, args.domain, args.split)
