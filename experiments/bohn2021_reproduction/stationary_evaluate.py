"""Common independent evaluation for joint/frozen policies and analytic fixed H."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import ART, imports
import recoverable_runtime as distribution
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
from riccati_terminal_probe import prior
from run import write, weights_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--fixed-horizon', type=int)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if (args.out / 'completed.json').exists():
        return
    args.out.mkdir(parents=True, exist_ok=True)
    distribution.OUT = ART / 'results/stationary_terminal'
    install_terminal('pendulum')
    distribution.install_distribution()
    env = distribution.make_env('pendulum', 926)
    cases = json.loads(args.bank.read_text())['cases']
    if args.model_dir:
        _, SAC, _ = imports()
        model = SAC.load(str(args.model_dir / 'model.zip'), env=env, reward_scale=.6)
        before = weights_hash(model)
        evaluate(model, env, cases, args.out, args.fixed_horizon, True)
        assert before == weights_hash(model)
    else:
        assert args.fixed_horizon is not None
        w, b, _ = prior()
        evaluate(None, env, cases, args.out, args.fixed_horizon, True,
                 terminal_weights=([w.astype(np.float32)], [b.astype(np.float32)]))
        before = None
    write(args.out / 'completed.json', {'frozen': True, 'weights_sha256': before,
          'bank_sha256': hashlib.sha256(args.bank.read_bytes()).hexdigest(),
          'episodes': len(cases), 'physical_costs_verified': True})


if __name__ == '__main__':
    main()
