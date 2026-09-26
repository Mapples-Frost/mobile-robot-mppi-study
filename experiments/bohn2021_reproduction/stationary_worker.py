"""JSON pipe bridge to the pinned Python 3.7 MPC runtime."""
import copy
import json
import sys
import time
import numpy as np
from runtime import ART
import recoverable_runtime as distribution
from optimized_runtime import install_terminal
from riccati_terminal_probe import prior
from mechanism_probe import checked_step
from run import serial


def main():
    distribution.OUT = ART / 'results/stationary_terminal'
    install_terminal('pendulum')
    distribution.install_distribution()
    env = None
    w, b, _ = prior()
    w, b = w.astype(np.float32), b.astype(np.float32)
    first_training_step = False
    for line in sys.stdin:
        request = json.loads(line)
        command = request['command']
        if command == 'close':
            break
        if command == 'init':
            seed = request['seed']
            np.random.seed(seed)
            env = distribution.make_env('pendulum', seed)
            first_training_step = request.get('training', False)
            if not first_training_step:
                env.set_value_function_weights_and_biases([w], [b])
            result = {'observation_dim': 56, 'terminal_w': w, 'terminal_b': b}
        elif command == 'reset':
            obs = env.reset(**request.get('case', {}))
            result = {'obs': obs, 'initial_state': copy.deepcopy(env.control_system.current_state)}
        elif command == 'step':
            h = request['horizon']
            assert isinstance(h, int) and 1 <= h <= 50
            start = time.perf_counter()
            obs, done, row = checked_step(env, 'pendulum', h)
            row.update(env.optimized_solver_info)
            row['elapsed_s'] = time.perf_counter() - start
            if first_training_step:
                env.set_value_function_weights_and_biases([w], [b])
                first_training_step = False
            vf = env.control_system.controller.mpc.vf
            np.testing.assert_array_equal(np.asarray(vf.weights_num).ravel(), w.ravel())
            np.testing.assert_array_equal(np.asarray(vf.biases_num).ravel(), b.ravel())
            result = {'obs': obs, 'reward': -row['cost'], 'done': done, 'row': row}
        else:
            raise ValueError(command)
        print('@RPC ' + json.dumps(result, default=serial, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
