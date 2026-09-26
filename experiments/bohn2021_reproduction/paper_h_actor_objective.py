"""Same-standard-deviation mean scans of frozen learned SAC objectives."""
from pathlib import Path
import json
import numpy as np

from min_q_eval_suite import model_dir
from paper_h_soft_probe import OUT, TASKS, METHODS, CASES, ANCHORS, read, digest
from runtime import imports
from run import weights_hash, write


def main():
    assert read(OUT / 'audit.json')['passed']
    _, SAC, _ = imports()
    rows, inputs = [], {}
    for task in TASKS:
        for method in METHODS:
            for seed in range(3):
                source = model_dir(task, method, seed)
                model = SAC.load(str(source / 'model.zip'))
                before = weights_hash(model)
                assert before == read(source / 'completed.json')['final_hash']
                alpha = float(model.ent_coef)
                folder = OUT / ('%s_%s_s%d' % (task, method, seed))
                for case in CASES:
                    for anchor in ANCHORS:
                        path = folder / ('case%02d_t%03d.json' % (case, anchor))
                        data = read(path)
                        inputs[str(path)] = digest(path)
                        summary = data['summary']
                        obs = np.asarray(data['branches'][str(summary['picks']['actor'])][0]['initial_observation'], dtype=np.float32)
                        mu, std = model.sess.run([model.policy_tf.act_mu, model.policy_tf.std],
                                                {model.observations_ph: obs[None, :]})
                        mu, std = float(mu.ravel()[0]), float(std.ravel()[0])
                        assert mu == summary['actor_mu'] and std == summary['actor_std']
                        candidates = np.r_[mu, np.arctanh(np.clip(np.linspace(-1, 1, 50), -.999999, .999999))]
                        objectives = {}
                        for count in (64, 128):
                            nodes, weights = np.polynomial.hermite.hermgauss(count)
                            noise = np.sqrt(2) * nodes
                            weights /= np.sqrt(np.pi)
                            latent = candidates[:, None] + std * noise[None, :]
                            actions = np.tanh(latent)
                            values = model.sess.run([model.step_ops[4], model.step_ops[5]], {
                                model.observations_ph: np.repeat(obs[None, :], actions.size, axis=0),
                                model.actions_ph: actions.astype(np.float32).reshape(-1, 1)})
                            q1, q2 = [v.reshape(actions.shape) for v in values]
                            q = q1 if method == 'author_rl' else np.minimum(q1, q2)
                            logp = -.5 * ((std * noise / (std + 1e-6))**2 + 2 * np.log(std) + np.log(2 * np.pi))
                            logp = logp[None, :] - np.log(1 - actions**2 + 1e-6)
                            objectives[str(count)] = (q - alpha * logp) @ weights
                        high, low = objectives['128'], objectives['64']
                        best = int(np.argmax(high))
                        high_gain = float(high[best] - high[0])
                        low_gain = float(low[best] - low[0])
                        discrepancy = abs(high_gain - low_gain)
                        rows.append({'task': task, 'method': method, 'seed': seed, 'case': case, 'anchor': anchor,
                                     'target': 'Q1' if method == 'author_rl' else 'min(Q1,Q2)',
                                     'mu': mu, 'std': std, 'actor_h': summary['picks']['actor'],
                                     'mean_tanh_derivative': float(1 - np.tanh(mu)**2),
                                     'best_mu': float(candidates[best]),
                                     'current_objective_128': float(high[0]),
                                     'gain_128': high_gain, 'gain_64_same_candidate': low_gain,
                                     'quadrature_gain_discrepancy': discrepancy,
                                     'stable_positive_gain': bool(min(high_gain, low_gain) > max(.01, 2 * discrepancy)),
                                     'candidate_means': candidates.tolist(),
                                     'objectives': {k: v.tolist() for k, v in objectives.items()}})
                assert weights_hash(model) == before
                model.sess.close()
    result = {'complete': True, 'frozen': True, 'quadrature_points': [64, 128], 'rows': rows,
              'input_hashes': inputs, 'script_sha256': digest(Path(__file__)),
              'scope': 'Post-hoc objective diagnostic using the trained Q1 or min-Q actor target at fixed std. No optimization, training, physical simulation or policy deployment.',
              'limit': 'Quadrature agreement is a numerical check, not a statistical confidence bound. Per-state mean freedom exceeds a shared neural actor. Better learned-objective value does not guarantee better physical control.'}
    write(OUT / 'actor_objective.json', result)
    print(json.dumps({'complete': True, 'rows': len(rows),
                      'stable_positive_gain': sum(r['stable_positive_gain'] for r in rows)}))


if __name__ == '__main__':
    main()
