"""Load pinned author sources without changing the downloaded repositories."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:
    os.environ[key] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLBACKEND'] = 'Agg'
for folder in ['gym-horizon','do-mpc-horizon','stable-baselines-horizon']:
    sys.path.insert(0, str(ART/'sources'/folder))


def imports():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from gym_let_mpc.let_mpc import LetMPCEnv
    from stable_baselines import SAC
    from stable_baselines.sac.policies import AHMPCPolicy
    return LetMPCEnv, SAC, AHMPCPolicy


def make_env(task, seed, fixed_horizon=None, corrected=True, aligned=False, scaled_obs=False):
    import numpy as np
    Env, _, _ = imports()

    class ReconstructedEnv(Env):
        def _get_variable_value(self, var):
            if aligned and var['type'] == 'tvp':
                val = self.control_system.tvps[var['name']].get_values(self.control_system._step_count)
                return float(np.asarray(val).reshape(-1)[0])
            return super()._get_variable_value(var)

        def get_observation(self):
            obs = super().get_observation()
            if scaled_obs:
                obs = obs.astype(float).copy()
                if task == 'vehicle':
                    # Invertible, fixed transform; preserves all 14 inputs.
                    xy = obs[:2].copy()
                    obs[3:5] = (obs[3:5]-xy)/5.
                    for j in range(3):
                        obs[5+3*j:7+3*j] = (obs[5+3*j:7+3*j]-xy)/10.
                    obs[:2] /= 30.
                    obs[2] /= np.pi
                else:
                    obs /= np.array([1.5,5.,np.pi/2,10.,1.])
            return obs

        def get_reward(self, rew_expr=None, done=False, info=None):
            # Upstream textual substitution makes (-v)^2 become -v^2. Evaluate
            # the locally generated expression with numeric variables instead.
            scope = {'np':np, 'done':int(done)}
            for v in self.config['environment']['reward']['variables']:
                value = self._get_variable_value(v)
                scope[v['name']] = float(np.asarray(value).reshape(-1)[0])
            expression = rew_expr if rew_expr is not None else self.config['environment']['reward']['expression']
            return float(eval(expression, {'__builtins__':{}}, scope))

        def reset(self, **kwargs):
            # Upstream reset reads observations before initializing this counter.
            self.steps_count = 0
            return super().reset(**kwargs)

        def step(self, action):
            horizon = int(np.clip(np.rint(fixed_horizon if fixed_horizon is not None else action[0]), 1, 50))
            obs, reward, done, info = super().step(np.array([float(horizon)]))
            if aligned:
                mpc = self.control_system.controller.mpc
                # Bellman stage cost is l(x_t,u_t,p_t), see paper Eq. (4b).
                info['data']['mpc_rewards'] = float(mpc.lterm_fun(
                    mpc.opt_p_num['_x0'],mpc.opt_x_num_unscaled['_u',0,0],
                    mpc.opt_x_num_unscaled['_z',1,0,-1],
                    mpc.opt_p_num['_tvp',0],mpc.opt_p_num['_p',0]))
                info['mpc_avg_stage_cost'] = info['data']['mpc_rewards']
                if task == 'pendulum':
                    violated = any(abs(self.control_system.current_state[k])>lim
                                   for k,lim in [('pos',1.5),('theta',np.pi/2)])
                    if violated:
                        old = info.get('reward/constraint',0.)
                        penalty = 10*(self.max_steps-self.steps_count)
                        reward += old-penalty
                        info['reward/constraint']=penalty
                        info['termination']='constraint'
                        done=True
            if corrected and task == 'vehicle':
                ctrl = self.control_system.controller
                collided = any(ctrl.get_obj_distance(self.control_system.current_state, j) <= ctrl.obj_data[j]['r'][0]
                               for j in range(ctrl.n_objects))
                if collided:
                    old = info.get('reward/constraint', 0.)
                    penalty = 2 * (self.max_steps-self.steps_count)
                    reward += old-penalty
                    info['reward/constraint'] = penalty
                    info['termination'] = 'constraint'
                    done = True
            info['executed_horizon'] = horizon
            info['solver_success'] = bool(self.control_system.controller.mpc.solver_stats.get('success', False))
            return obs, reward, done, info

    env = ReconstructedEnv(str(ART/'configs'/(task+'.json')))
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def install_correct_nstep():
    """Fix terminal inclusion and off-by-one in the author's 32-step replay sampler."""
    import numpy as np
    from stable_baselines.common.buffers import ReplayBuffer
    original = ReplayBuffer._encode_sample

    def encode(self, idxes, env=None, n_step=1, gamma=1):
        if n_step == 1:
            return original(self, idxes, env=env, n_step=1, gamma=gamma)
        starts, acts, rews, ends, masks, counts = [], [], [], [], [], []
        for i in idxes:
            start = self._storage[i]
            total, count, end, mask = 0., 0, start[3], 0.
            for j in range(i, min(i+n_step, len(self._storage))):
                # Never cross the newest-to-oldest boundary of a full ring buffer.
                if j > i and j == self._next_idx:
                    break
                tr = self._storage[j]
                total += gamma**count * float(tr[2])
                count += 1
                end = tr[3]
                extra = dict(zip(self._extra_data_names, tr[5:]))
                mask = float(not extra['bootstrap']) if 'bootstrap' in extra else float(tr[4])
                if tr[4]:
                    break
            starts.append(start[0]); acts.append(start[1]); rews.append(total)
            ends.append(end); masks.append(mask); counts.append(count)
        return np.asarray(starts), np.asarray(acts), np.asarray(rews), np.asarray(ends), np.asarray(masks), {'n_step':np.asarray(counts)}
    ReplayBuffer._encode_sample = encode


if __name__ == '__main__':
    Env, SAC, Policy = imports()
    import numpy as np
    import time
    import json
    task = sys.argv[1] if len(sys.argv)>1 else 'pendulum'
    env = make_env(task, 91701)
    obs = env.reset()
    print('reset', task, obs, flush=True)
    t = time.perf_counter()
    for i in range(10):
        obs, rew, done, info = env.step(np.array([25.]))
        print(i, rew, info.get('termination'),info['data']['mpc_rewards'], flush=True)
        if done: break
    print('elapsed',time.perf_counter()-t)
