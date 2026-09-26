"""Validation-only full-episode extraction from frozen continuous critics."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
import recoverable_runtime as distribution

OUT = ART / 'results/critic_execution_probe'
GROUPS = ['recoverable_distribution', 'prior_refinement']


def read(path):
    return json.loads(path.read_text())


def worker(group, seed):
    source = ART / 'results' / group
    distribution.OUT = source
    distribution.install_distribution()
    install_terminal('pendulum')
    _, SAC, _ = imports()
    folder = source / ('pendulum_rl_s%d' % seed)
    model = SAC.load(str(folder / 'model.zip'))
    before = weights_hash(model)
    assert before == read(folder / 'completed.json')['final_hash']
    cases = read(source / 'validation_bank.json')['cases']
    starts = [e['initial_state'] for e in read(folder / 'eval_value/summary.json')['episodes']]
    dest = OUT / ('%s_s%d' % (group, seed))
    dest.mkdir(exist_ok=True)
    rows = []
    for mode in ['q1', 'min_q']:
        def predict(obs, deterministic=True):
            values = model.sess.run([model.step_ops[4], model.step_ops[5]], {
                model.observations_ph: np.repeat(obs[None, :], 50, axis=0),
                model.actions_ph: np.linspace(-1, 1, 50, dtype=np.float32)[:, None]})
            score = values[0] if mode == 'q1' else np.minimum(values[0], values[1])
            return np.array([float(np.argmax(score) + 1)]), None
        model.predict = predict
        target = dest / mode
        target.mkdir(exist_ok=True)
        env = distribution.make_env('pendulum', 26091880 + seed)
        evaluate(model, env, cases, target, None, True)
        summary = read(target / 'summary.json')
        assert [e['initial_state'] for e in summary['episodes']] == starts
        assert weights_hash(model) == before
        row = {'mode': mode, 'cost': summary['mean_total_cost'], 'constraints': summary['constraint_episodes'],
               'mean_h': float(np.mean([e['mean_horizon'] for e in summary['episodes']]))}
        rows.append(row)
        write(target / 'completed.json', {'frozen': True, 'weights_sha256': before, 'matched_starts': True})
        print(json.dumps({'group': group, 'seed': seed, **row}), flush=True)
    write(dest / 'summary.json', {'complete': True, 'seed': seed, 'group': group, 'results': rows})
    model.sess.close()


def main():
    if len(sys.argv) > 1:
        worker(sys.argv[1], int(sys.argv[2]))
        return
    OUT.mkdir(exist_ok=True)
    write(OUT / 'protocol.json', {'groups': GROUPS, 'seeds': [0, 1, 2], 'cases': 'all10original validation scenes',
                                  'modes': ['argmax Q1', 'argmax min(Q1,Q2)'], 'horizons': list(range(1, 51)),
                                  'scope': 'Post-hoc diagnosis of actor extraction; no training, holdout tuning, deployment or new paper-reproduction claim.',
                                  'controls': 'Preserve each RL terminal, exact same initial states and frozen model weights.'})
    def launch(job):
        group, seed = job
        with open(OUT / ('%s_s%d.log' % job), 'a') as log:
            result = subprocess.run([sys.executable, '-u', __file__, group, str(seed)], stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        assert result.returncode == 0, job
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(launch, [(g, s) for g in GROUPS for s in range(3)]))
    write(OUT / 'summary.json', {'complete': True, 'episodes': 120,
                                'results': [read(OUT / ('%s_s%d' % (g, s)) / 'summary.json') for g in GROUPS for s in range(3)]})


if __name__ == '__main__':
    main()
