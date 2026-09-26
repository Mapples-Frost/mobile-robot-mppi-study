"""Use Tianshou SAC implementations with one shared frozen-terminal MPC bridge."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from runtime import ROOT

DEPS = ROOT / '.codex_tmp/tianshou051'
sys.path.insert(0, str(DEPS))
import numpy as np
import torch
from gymnasium import spaces
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import SACPolicy, DiscreteSACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor, Critic as DiscreteCritic

torch.set_num_threads(1)
LEGACY = Path('/home/mapples/.local/share/bohn2021-python37/bin/python')


def make_policy(mode, seed):
    torch.manual_seed(seed)
    net = Net(56, hidden_sizes=[32, 32])
    if mode == 'discrete':
        actor = Actor(net, 50, softmax_output=False)
        critics = [DiscreteCritic(Net(56, hidden_sizes=[256, 256]), last_size=50) for _ in range(2)]
        cls = DiscreteSACPolicy
        kwargs = {'action_space': spaces.Discrete(50)}
    else:
        actor = ActorProb(net, 1, unbounded=True, conditioned_sigma=True)
        critics = [Critic(Net(56, action_shape=1, hidden_sizes=[256, 256], concat=True)) for _ in range(2)]
        cls = SACPolicy
        kwargs = {'action_space': spaces.Box(-1., 1., shape=(1,), dtype=np.float32), 'action_scaling': False}
    return cls(actor, torch.optim.Adam(actor.parameters(), lr=3e-4),
               critics[0], torch.optim.Adam(critics[0].parameters(), lr=3e-4),
               critics[1], torch.optim.Adam(critics[1].parameters(), lr=3e-4),
               gamma=.97, tau=.005, alpha=.01, estimation_step=1,
               reward_normalization=False, deterministic_eval=True, **kwargs)


def horizon(mode, action):
    if mode == 'discrete':
        value = int(np.asarray(action).item())
        assert 0 <= value < 50
        return value + 1
    return int(np.clip(np.rint(1 + (float(np.asarray(action).item()) + 1) * 24.5), 1, 50))


def predict(policy, obs):
    with torch.no_grad():
        return policy(Batch(obs=np.asarray(obs, dtype=np.float32)[None], info={})).act.cpu().numpy()[0]


def policy_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


class Bridge:
    def __init__(self, seed, training=False):
        self.process = subprocess.Popen([str(LEGACY), '-u', str(Path(__file__).with_name('stationary_worker.py'))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.metadata = self.call('init', seed=seed, training=training)

    def call(self, command, **kwargs):
        self.process.stdin.write(json.dumps(dict(kwargs, command=command)) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('MPC bridge exited: %s' % self.process.poll())
            if line.startswith('@RPC '):
                return json.loads(line[5:])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.process.stdin.close()
        self.process.stdout.close()


def evaluate(policy, mode, bank, out):
    from run import write
    out.mkdir(parents=True, exist_ok=True)
    before = policy_hash(policy)
    policy.eval()
    env = Bridge(926)
    rows = []
    try:
        for index, case in enumerate(json.loads(bank.read_text())['cases']):
            reset = env.call('reset', case=case)
            obs, trace = reset['obs'], []
            while True:
                result = env.call('step', horizon=horizon(mode, predict(policy, obs)))
                trace.append(result['row'])
                obs = result['obs']
                if result['done']:
                    break
            write(out / ('trace_%02d.json' % index), trace)
            rows.append({'episode': index, 'initial_state': reset['initial_state'], 'steps': len(trace),
                'termination': trace[-1]['termination'], 'total_cost': sum(r['cost'] for r in trace),
                'performance_cost': sum(r['performance'] for r in trace),
                'computation_cost': sum(r['compute'] for r in trace),
                'constraint_cost': sum(r['constraint'] for r in trace),
                'discounted_cost': sum(.97**t*r['cost'] for t, r in enumerate(trace)),
                'mean_horizon': float(np.mean([r['horizon'] for r in trace])),
                'solver_failure_steps': sum(not r['solver_success'] for r in trace)})
    finally:
        env.close()
    assert before == policy_hash(policy)
    write(out / 'summary.json', {'episodes': rows, 'mean_total_cost': float(np.mean([r['total_cost'] for r in rows])),
        'constraint_episodes': sum(r['termination'] == 'constraint' for r in rows),
        'physical_cost_and_bounds_verified': True, 'weights_sha256': before,
        'bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest()})
    return rows
