"""Query frozen actor objectives at saved observations without environment replay."""
import json
import sys
from pathlib import Path

import numpy as np

from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal

OUT = ART / 'results/horizon_credit_probe'


def main():
    install_terminal('pendulum')
    _, SAC, _ = imports()
    import tensorflow as tf
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    noise, weights = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
    rows = []
    for group in ['recoverable_distribution', 'prior_refinement']:
        for seed in range(3):
            folder = ART / 'results' / group / ('pendulum_rl_s%d' % seed)
            model = SAC.load(str(folder / 'model.zip'))
            before = weights_hash(model)
            with model.graph.as_default():
                gradient = tf.gradients(model.step_ops[4], model.actions_ph)[0]
            source = OUT / ('%s_s%d' % (group, seed))
            for result in json.loads((source / 'summary.json').read_text())['results']:
                case, t = result['case'], result['t']
                branch = json.loads((source / ('case%d_t%d' % (case, t)) / ('det_h%d.json' % result['picks']['actor'])).read_text())
                obs = np.array(branch['initial_observation'], dtype=np.float32)
                mu, std, raw = result['actor_mu'], result['actor_std'], result['raw_actor_action']
                grad = model.sess.run(gradient, {model.observations_ph: obs[None, :], model.actions_ph: [[raw]]})[0, 0]
                candidate_mu = np.r_[mu, np.arctanh(np.clip(np.linspace(-1, 1, 50), -.999999, .999999))]
                latent = candidate_mu[:, None] + std * noise[None, :]
                actions = np.tanh(latent)
                batch = np.repeat(obs[None, :], actions.size, axis=0)
                q1 = model.sess.run(model.step_ops[4], {model.observations_ph: batch, model.actions_ph: actions.astype(np.float32).reshape(-1, 1)}).reshape(actions.shape)
                logp = -.5 * ((std * noise / (std + 1e-6))**2 + 2 * np.log(std) + np.log(2 * np.pi))
                logp = logp[None, :] - np.log(1 - actions**2 + 1e-6)
                objective = (q1 - model.ent_coef * logp) @ weights
                best = int(np.argmax(objective))
                rows.append({'group': group, 'seed': seed, 'case': case, 't': t,
                             'actor_h': result['picks']['actor'], 'q1_best_h': result['picks']['q1'],
                             'mu': mu, 'std': std, 'tanh_derivative': float(1 - raw**2),
                             'q1_action_gradient': float(grad), 'q1_mean_path_gradient': float(grad * (1 - raw**2)),
                             'same_std_best_mean_h': int(np.rint(1 + (np.tanh(candidate_mu[best]) + 1) * 24.5)),
                             'current_expected_q1_minus_alpha_logp': float(objective[0]),
                             'same_std_objective_gain': float(objective[best] - objective[0])})
            assert weights_hash(model) == before
            model.sess.close()
    write(OUT / 'actor_objective.json', {'complete': True, 'frozen': True, 'quadrature_points': 64,
                                       'scope': 'Frozen learned objective, not physical return ground truth. Same standard deviation; mean varied only.',
                                       'rows': rows})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
