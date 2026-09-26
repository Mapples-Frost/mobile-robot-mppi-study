"""Verify final frozen TF terminal and save/load rollout against training output."""
import json
import numpy as np
from runtime import ART, imports
import recoverable_runtime as distribution
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
from riccati_terminal_probe import prior
from run import write, weights_hash


def main():
    root = ART / 'results/stationary_terminal'
    distribution.OUT = root
    install_terminal('pendulum')
    distribution.install_distribution()
    _, SAC, _ = imports()
    w, b, _ = prior()
    case = json.loads((root / 'validation_bank.json').read_text())['cases'][0]
    rows = []
    for seed in range(3):
        folder = root / ('pendulum_rl_s%d' % seed)
        env = distribution.make_env('pendulum', 926)
        model = SAC.load(str(folder / 'model.zip'), env=env, reward_scale=.6)
        assert model.reward_scale == .6
        actual = [v for group in model.policy_tf.get_mpc_vfn_weights_and_biases() for v in group]
        for left, right in zip(actual, [w.astype(np.float32), b.astype(np.float32)]):
            np.testing.assert_array_equal(left, right)
        target = folder / 'reload_validation0'
        target.mkdir(exist_ok=True)
        before = weights_hash(model)
        evaluate(model, env, [case], target, None, True)
        old = json.loads((folder / 'eval_value/trace_00.json').read_text())
        new = json.loads((target / 'trace_00.json').read_text())
        assert len(old) == len(new)
        for left, right in zip(old, new):
            for field in ['state', 'input', 'horizon', 'cost', 'performance', 'compute', 'constraint']:
                assert left[field] == right[field], (seed, field)
        assert before == weights_hash(model)
        rows.append({'seed': seed, 'reload_rollout_exact': True, 'terminal_prior_exact': True, 'reward_scale': model.reward_scale})
        model.sess.close()
    write(root / 'reload_verification.json', {'passed': True, 'models': rows})


if __name__ == '__main__':
    main()
