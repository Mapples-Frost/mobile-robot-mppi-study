"""Single-factor full MPC-reference preview for the pendulum horizon policy."""
import numpy as np
from optimized_runtime import make_env as original_env


def make_env(task,seed,fixed_horizon=None):
    assert task=='pendulum'
    from gym import spaces
    env=original_env(task,seed,fixed_horizon)
    old=env.get_observation
    def observation():
        base=old();c=env.control_system;clock=c._step_count;tvp=c.tvps['pos_r']
        required=clock+51
        if len(tvp.values)<required:tvp.generate_values(required-len(tvp.values))
        position=c.current_state['pos']
        preview=[(float(np.asarray(tvp.get_values(clock+k)).ravel()[0])-position)/1.5 for k in range(1,51)]
        return np.concatenate([base[:5],preview,base[-1:]]).astype(np.float32)
    env.get_observation=observation
    env.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(56,),dtype=np.float32)
    # Optimized reset has a nine-feature assertion; retain its zero-terminal
    # warmup while returning the expanded observation only after that assertion.
    reset9=env.reset
    def reset(**kw):
        env.get_observation=old
        try:reset9(**kw)
        finally:env.get_observation=observation
        return observation()
    env.reset=reset
    return env
