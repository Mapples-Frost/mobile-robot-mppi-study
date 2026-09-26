"""Preserve resource-interrupted attempts and rerun with bounded replay memory."""
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import categorical_suite as suite
from categorical_runtime import np, Batch, ReplayBuffer
from run import write


def verify_capacity():
    large, bounded = ReplayBuffer(100), ReplayBuffer(15)
    for step in range(15):
        item = Batch(obs=np.array([step]), act=step % 3, rew=-step,
                     obs_next=np.array([step + 1]), terminated=False, truncated=step % 5 == 4, info={})
        large.add(item)
        bounded.add(item)
        np.random.seed(770 + step)
        a, indices_a = large.sample(20)
        np.random.seed(770 + step)
        b, indices_b = bounded.sample(20)
        np.testing.assert_array_equal(indices_a, indices_b)
        for key in ['obs', 'act', 'rew', 'obs_next', 'terminated', 'truncated', 'done']:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(large.unfinished_index(), bounded.unfinished_index())
    return True


def job(spec):
    mode, seed, evaluation = spec
    name = '%s_s%d' % (mode, seed)
    command = [sys.executable, '-u', str(suite.SCRIPTS / 'categorical_resource_run.py'), '--mode', mode,
               '--seed', str(seed), '--steps', '15000', '--out', str(suite.OUT / name), '--bank',
               str(suite.BASE / ('holdout_bank.json' if evaluation else 'validation_bank.json'))]
    if evaluation:
        command.append('--eval-only')
    with open(suite.OUT / (name + ('_holdout' if evaluation else '') + '.log'), 'a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=18000)
    assert result.returncode == 0, (name, evaluation)
    print(json.dumps({'complete': name, 'holdout': evaluation}), flush=True)


class LimitedPool(ThreadPoolExecutor):
    def __init__(self, max_workers=None):
        super().__init__(max_workers=3)


def main():
    assert (suite.BASE / 'completed.json').exists(), 'Finish legacy experiments before resource recovery'
    assert verify_capacity()
    amendment = suite.OUT / 'resource_amendment.json'
    if not amendment.exists():
        archive = suite.OUT / 'interrupted_memory'
        archive.mkdir(exist_ok=True)
        attempts = []
        for mode in ['continuous', 'discrete']:
            for seed in range(3):
                folder = suite.OUT / ('%s_s%d' % (mode, seed))
                if folder.exists() and not (folder / 'completed.json').exists():
                    progress = json.loads((folder / 'progress.json').read_text())
                    attempts.append(dict(progress, name=folder.name,
                                         transitions_upper_bound=progress['steps'] + 100))
                    shutil.move(str(folder), str(archive / folder.name))
                    logfile = suite.OUT / (folder.name + '.log')
                    if logfile.exists():
                        shutil.move(str(logfile), str(archive / logfile.name))
        write(amendment, {'reason': 'Host memory pressure and WSL command timeouts under combined concurrency',
            'interrupted_attempts': attempts, 'selection_used': False,
            'restart': 'All incomplete runs restarted from their original seeds; interrupted checkpoints never selected',
            'allocation': 'Allocate min(replay_capacity,total_training_steps); no eviction before end, same sampling domain and RNG',
            'capacity_equivalence_test': True, 'max_workers': 3,
            'protocol_unchanged': 'Rewards, terminal, algorithms, seeds, final15000 checkpoints and evaluation gates retained',
            'source_hashes': {n: hashlib.sha256((suite.SCRIPTS / n).read_bytes()).hexdigest()
                              for n in ['categorical_resource_run.py', 'categorical_recover.py']}})
    suite.job = job
    suite.ThreadPoolExecutor = LimitedPool
    suite.main()


if __name__ == '__main__':
    main()
