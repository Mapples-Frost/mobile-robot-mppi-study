"""Freeze only the terminal optimizer; preserve author SAC and replay RNG use."""
import hashlib
import numpy as np
import run
import recoverable_runtime as distribution
from runtime import ART, imports
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
from riccati_terminal_probe import prior


def main():
    distribution.OUT = ART / 'results/stationary_terminal'
    install_terminal('pendulum')
    distribution.install_distribution()
    import tensorflow as tf
    w, b, details = prior()
    original_variable = tf.get_variable

    def initialized(name, *args, **kwargs):
        if tf.get_variable_scope().name.endswith('mpc_value_fns/mpc_value_fn') and 'initializer' in kwargs:
            if name == 'kernel':
                kwargs['initializer'] = w.astype(np.float32)
            elif name == 'bias':
                kwargs['initializer'] = b.astype(np.float32)
        return original_variable(name, *args, **kwargs)

    tf.get_variable = initialized
    Env, SAC, Policy = imports()

    class FrozenTerminalSAC(SAC):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            terminal_op = self.mpc_value_fn_train_op
            assert sum(op is terminal_op for op in self.step_ops) == 1
            self.step_ops = [op for op in self.step_ops if op is not terminal_op]
            self.frozen_terminal = [a.copy() for group in self.policy_tf.get_mpc_vfn_weights_and_biases() for a in group]
            for actual, expected in zip(self.frozen_terminal, [w.astype(np.float32), b.astype(np.float32)]):
                np.testing.assert_array_equal(actual, expected)
            self.terminal_checks = 0

        def _train_step(self, *args, **kwargs):
            result = super()._train_step(*args, **kwargs)
            actual = [a for group in self.policy_tf.get_mpc_vfn_weights_and_biases() for a in group]
            for before, after in zip(self.frozen_terminal, actual):
                np.testing.assert_array_equal(before, after)
            self.terminal_checks += 1
            return result

    run.imports = lambda: (Env, FrozenTerminalSAC, Policy)
    run.make_env = lambda task, seed, fixed_horizon=None, **kwargs: distribution.make_env(task, seed, fixed_horizon)
    run.evaluate = evaluate
    original_write = run.write

    def write(path, data):
        if path.name == 'manifest.json':
            data['terminal_prior'] = dict(details, initialization='Discounted discrete Riccati linearization', joint_learning=False)
            data['freeze_method'] = 'Omit only terminal optimizer from step_ops; preserve graph, replay sampling and RNG consumption; assert unchanged after every update.'
            data['forecast_refinement'] = {'observation': 'Full50step reference preview,56D'}
            data['config_path'] = str(distribution.prepare_config())
            data['config_sha256'] = hashlib.sha256(distribution.prepare_config().read_bytes()).hexdigest()
            data['fidelity'] = 'Frozen analytic terminal extension; not an exact paper reproduction.'
        if path.name == 'completed.json':
            data['terminal_unchanged_every_update'] = True
        original_write(path, data)

    run.write = write
    run.main()


if __name__ == '__main__':
    main()
