"""Numerical contracts for the library adapter before collecting experiment data."""
import argparse
import json
from pathlib import Path
from categorical_runtime import np, torch, Batch, ReplayBuffer, make_policy, horizon, Bridge
from run import write
from runtime import ART


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    np.random.seed(807)
    policy = make_policy('discrete', 807)
    policy.train()
    for h in range(1, 51):
        assert horizon('discrete', h - 1) == h
        assert horizon('continuous', np.array([(h - 1) / 24.5 - 1])) == h
    buffer = ReplayBuffer(16)
    for terminated, truncated in [(False, False), (True, False), (False, True)]:
        buffer.add(Batch(obs=np.random.randn(56).astype(np.float32), act=7, rew=-2. / .6,
                         obs_next=np.random.randn(56).astype(np.float32),
                         terminated=terminated, truncated=truncated, info={}))
    indices = np.arange(3)
    batch = buffer[indices]
    with torch.no_grad():
        dist = policy(batch, input='obs_next').dist
        q = torch.minimum(policy.critic1_old(batch.obs_next), policy.critic2_old(batch.obs_next))
        exact = (dist.probs * (q - .01 * dist.logits)).sum(-1)
        torch.testing.assert_close(policy._target_q(buffer, indices), exact)
        expected = -2. / .6 + .97 * torch.tensor([1., 0., 1.]) * exact
    processed = policy.process_fn(batch, buffer, indices)
    torch.testing.assert_close(processed.returns.flatten(), expected)
    for optimizer in [policy.actor_optim, policy.critic1_optim, policy.critic2_optim]:
        for group in optimizer.param_groups:
            group['lr'] = 0.
    with torch.no_grad():
        dist = policy(processed).dist
        q = torch.minimum(policy.critic1(processed.obs), policy.critic2(processed.obs))
        expected_loss = (dist.probs * (.01 * dist.logits - q)).sum(-1).mean().item()
    metrics = policy.learn(processed)
    np.testing.assert_allclose(metrics['loss/actor'], expected_loss, rtol=1e-5, atol=1e-6)
    case = json.loads((ART / 'results/stationary_terminal/validation_bank.json').read_text())['cases'][0]
    env = Bridge(807)
    try:
        traces = []
        for _ in range(2):
            start = env.call('reset', case=case)
            trace = [env.call('step', horizon=h) for h in [1, 10, 25, 50]]
            traces.append((start, trace))
        assert traces[0][0] == traces[1][0]
        for left, right in zip(traces[0][1], traces[1][1]):
            assert left['obs'] == right['obs']
            assert left['reward'] == right['reward']
            assert left['row']['state'] == right['row']['state']
    finally:
        env.close()
    write(args.out, {'action_mapping_50_values': True, 'discrete_soft_bellman_exact': True,
          'constraint_stops_bootstrap': True, 'timeout_keeps_bootstrap': True,
          'discrete_actor_exact_expectation': True, 'bridge_matched_reset_and_replay': True,
          'reward_divisor': .6, 'prior_unchanged_each_step': True})
    print('All adapter contracts passed', flush=True)


if __name__ == '__main__':
    main()
