"""Matched library continuous/categorical SAC, with unchanged physical reward."""
import argparse
import fcntl
import json
import random
import time
from pathlib import Path
from categorical_runtime import (np, torch, Batch, ReplayBuffer, Bridge, make_policy,
                                 predict, horizon, policy_hash, evaluate)
from run import write


def save(path, policy, spec):
    torch.save({'policy': policy.state_dict(), 'spec': spec,
                'actor_optim': policy.actor_optim.state_dict(),
                'critic1_optim': policy.critic1_optim.state_dict(),
                'critic2_optim': policy.critic2_optim.state_dict()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['continuous', 'discrete'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = open(args.out / 'run.lock', 'a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if (args.out / 'completed.json').exists() and not args.eval_only:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy = make_policy(args.mode, args.seed)
    if args.eval_only:
        saved = torch.load(args.out / 'model.pt', weights_only=False)
        policy.load_state_dict(saved['policy'])
        evaluate(policy, args.mode, args.bank, args.out / ('eval_' + args.bank.stem))
        return
    initial_hash = policy_hash(policy)
    spec = {'mode': args.mode, 'seed': args.seed, 'steps': args.steps, 'library': 'tianshou==0.5.1',
        'reward_scale': .6, 'gamma': .97, 'alpha': .01, 'tau': .005, 'batch_size': 256,
        'actor_layers': [32, 32], 'critic_layers': [256, 256], 'initial_hash': initial_hash,
        'terminal': 'Frozen float32 analytic Riccati; identical legacy MPC worker for both arms',
        'action': 'categorical 0..49 maps to H1..50' if args.mode == 'discrete' else 'tanh Gaussian [-1,1] maps to rounded H1..50',
        'warmup': 'First100 actions share uniform integer H across matched arms',
        'timeout': 'Bootstrap at steps truncation; stop bootstrap for physical constraint termination',
        'replay_capacity': 1000000, 'checkpoint_selection': 'final only'}
    write(args.out / 'manifest.json', spec)
    replay = ReplayBuffer(1000000)
    warmup = np.random.RandomState(args.seed)
    env = Bridge(args.seed, training=True)
    write(args.out / 'terminal.json', env.metadata)
    started = time.perf_counter()
    updates, rows, losses = 0, [], []
    current = {'steps': 0, 'total_cost': 0., 'solver_failure_steps': 0}
    policy.train()
    try:
        obs = np.asarray(env.call('reset')['obs'], dtype=np.float32)
        for step in range(1, args.steps + 1):
            if step <= 100:
                h = int(warmup.randint(1, 51))
                action = h - 1 if args.mode == 'discrete' else np.array([(h - 1) / 24.5 - 1], dtype=np.float32)
            else:
                action = predict(policy, obs)
                h = horizon(args.mode, action)
            result = env.call('step', horizon=h)
            next_obs = np.asarray(result['obs'], dtype=np.float32)
            termination = result['row']['termination']
            terminated = bool(result['done'] and termination != 'steps')
            truncated = bool(result['done'] and termination == 'steps')
            replay.add(Batch(obs=obs, act=action, rew=result['reward'] / .6, obs_next=next_obs,
                             terminated=terminated, truncated=truncated, info={}))
            current['steps'] += 1
            current['total_cost'] -= result['reward']
            current['solver_failure_steps'] += int(not result['row']['solver_success'])
            if result['done']:
                rows.append(dict(current, termination=termination))
                current = {'steps': 0, 'total_cost': 0., 'solver_failure_steps': 0}
                obs = np.asarray(env.call('reset')['obs'], dtype=np.float32)
            else:
                obs = next_obs
            if len(replay) >= 256:
                metrics = policy.update(256, replay)
                assert all(np.isfinite(v) for v in metrics.values())
                updates += 1
                if step % 100 == 0:
                    losses.append(dict(metrics, step=step))
            if step % 100 == 0:
                progress = {'steps': step, 'target': args.steps, 'updates': updates,
                            'elapsed_s': time.perf_counter() - started, 'episodes': len(rows)}
                write(args.out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            if step % 2500 == 0:
                save(args.out / ('checkpoint_%05d.pt' % step), policy, spec)
                write(args.out / 'training_episodes.json', rows)
    finally:
        env.close()
    assert updates == max(0, args.steps - 255)
    save(args.out / 'model.pt', policy, spec)
    before = policy_hash(policy)
    reloaded = make_policy(args.mode, args.seed)
    reloaded.load_state_dict(torch.load(args.out / 'model.pt', weights_only=False)['policy'])
    assert policy_hash(reloaded) == before
    policy.eval()
    reloaded.eval()
    np.testing.assert_array_equal(predict(policy, obs), predict(reloaded, obs))
    write(args.out / 'training_episodes.json', rows)
    write(args.out / 'training_losses.json', losses)
    evaluate(reloaded, args.mode, args.bank, args.out / 'eval_validation_bank')
    write(args.out / 'completed.json', {'steps': args.steps, 'updates': updates, 'final_hash': before,
          'weights_changed': initial_hash != before, 'save_load_exact': True,
          'terminal_unchanged_every_step': True, 'elapsed_s': time.perf_counter() - started})


if __name__ == '__main__':
    main()
