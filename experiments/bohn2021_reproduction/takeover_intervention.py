"""Validation-only horizon intervention on the two unfinished pendulum studies."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
import recoverable_runtime as distribution


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['recoverable_distribution', 'prior_refinement'], required=True)
    args = parser.parse_args()
    source = ART / 'results' / args.group
    out = ART / 'results/takeover_intervention' / args.group
    out.mkdir(parents=True, exist_ok=True)
    distribution.OUT = source
    distribution.install_distribution()
    install_terminal('pendulum')
    _, SAC, _ = imports()
    bank_path = source / 'validation_bank.json'
    cases = read(bank_path)['cases']
    h = read(source / 'selected_fixed.json')['h']
    protocol = {
        'scope': 'Post-hoc validation diagnosis only; no training, test selection or new performance claim.',
        'group': args.group, 'intervention': 'Keep each RL terminal and force the validation-selected fixed horizon.',
        'horizon': h, 'seeds': [0, 1, 2], 'cases': len(cases),
        'bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        'replay': 'Before intervention, exactly replay case 0 with the original policy for every seed.',
    }
    write(out / 'protocol.json', protocol)
    rows = []
    starts = None
    for seed in range(3):
        folder = source / ('pendulum_rl_s%d' % seed)
        model = SAC.load(str(folder / 'model.zip'))
        before = weights_hash(model)
        assert before == read(folder / 'completed.json')['final_hash']
        env = distribution.make_env('pendulum', 26091800 + seed)
        original = read(folder / 'eval_value/summary.json')
        with tempfile.TemporaryDirectory(prefix='takeover-replay-') as tmp:
            evaluate(model, env, cases[:1], Path(tmp), None, True)
            actual = read(Path(tmp) / 'trace_00.json')
        expected = read(folder / 'eval_value/trace_00.json')
        keys = ['state', 'input', 'horizon', 'cost', 'performance', 'compute', 'constraint', 'selected_objective', 'candidate_objectives']
        assert [{k: t[k] for k in keys} for t in actual] == [{k: t[k] for k in keys} for t in expected]
        dest = out / ('s%d_force%d' % (seed, h))
        dest.mkdir(exist_ok=True)
        evaluate(model, env, cases, dest, h, True)
        assert before == weights_hash(model)
        changed = read(dest / 'summary.json')
        initial = [e['initial_state'] for e in changed['episodes']]
        assert initial == [e['initial_state'] for e in original['episodes']]
        if starts is None:
            starts = initial
        assert starts == initial
        row = {'seed': seed, 'horizon': h, 'original_cost': original['mean_total_cost'],
               'forced_cost': changed['mean_total_cost'], 'original_constraints': original['constraint_episodes'],
               'forced_constraints': changed['constraint_episodes'], 'replay_exact': True, 'weights_sha256': before}
        write(dest / 'completed.json', {'frozen': True, 'weights_sha256': before, 'starts_match': True})
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.sess.close()
    write(out / 'summary.json', {'complete': True, 'protocol': protocol, 'rows': rows})


if __name__ == '__main__':
    main()
