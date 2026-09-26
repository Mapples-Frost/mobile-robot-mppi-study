"""Serial controller timing, separately registered from the efficacy experiment.

Uses all validation scenes, all seeds and all arms. Timing never selects a model
or unlocks the independent test. No mutation of the registered training code.
"""
import argparse
import copy
import fcntl
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

from branch_calibration_protocol import OUT, TASKS, ARMS, BASE_H, bank_path, verify, model_dir
from branch_calibration_run import Greedy, actor_h, observed_step
from branch_calibration_audit import audit_trace, verify_artifact_hashes
from paper_h_soft_probe import read, digest
from runtime import ROOT, imports, make_env
from run import write, weights_hash

DEST = OUT / 'serial_timing'
REPEATS = 2


def protocol():
    return {
        'scope': 'Validation-only measured timing, descriptive; no model selection or test unlocking.',
        'tasks': list(TASKS), 'arms': list(ARMS), 'seeds': [0, 1, 2],
        'scenes': 'All ten registered validation scenes per task, including failures.',
        'repeats': REPEATS, 'workers': 1, 'order_seed': 2609241400,
        'order': 'Two independent seeded permutations of the 24 task/seed/arm conditions.',
        'primary': 'Wall time of horizon selection plus controller.get_action, including MPC solve, prediction extraction and controller bookkeeping. Excludes simulator, reward/audit, disk writes and reset.',
        'secondary': 'Policy, controller and full environment-step wall time separately; episode totals, median, mean, p95 and control-period exceedances.',
        'warmup': 'One untimed inference per loaded model. Author reset H50 action excluded from decision latency but timed and counted separately for every episode.',
        'audit': 'Every saved state, action, observation, cost, terminal label and solver outcome must exactly match the previously audited efficacy validation trace; timing fields excluded.',
        'claim_rule': 'No confirmed acceleration claim from validation. Report every seed, paired episode time ratios, failures, and unchanged efficacy gate. A shorter failing episode is not a performance gain.',
        'environment': 'Pinned Python3.7 and author sources; single-thread library settings inherited from runtime. No concurrent project simulation/training process. WSL scheduling and host activity remain limitations.',
    }


def prepare():
    verify()
    DEST.mkdir(exist_ok=True)
    p = DEST / 'protocol.json'
    if p.exists(): assert read(p) == protocol()
    else: write(p, protocol())
    paths = [p, Path(__file__), OUT / 'inputs_sha256.json']
    hashes = {str(p): digest(p) for p in paths}
    path = DEST / 'registration.json'
    if path.exists(): assert read(path)['hashes'] == hashes
    else: write(path, {'registered_unix': time.time(), 'hashes': hashes})


def check_idle():
    conflicts = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid(): continue
        try: args = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError): continue
        if args and 'python' in Path(args[0]).name and any(
                'experiments/' in a and a.endswith('.py') for a in args[1:]):
            conflicts.append({'pid': int(path.parent.name), 'args': args})
    assert not conflicts, conflicts


def run():
    prepare()
    lock = (DEST / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = read(OUT / 'validation_gate.json')
    assert gate['data_audit_passed']
    # A negative efficacy gate must not prevent reporting actual cost.
    for p, h in gate['hashes'].items(): assert digest(Path(p)) == h
    verify_artifact_hashes(OUT / 'training_audit.json')
    _, SAC, _ = imports()
    jobs = [(t, s, a) for t in TASKS for s in range(3) for a in ARMS]
    if not (DEST / 'environment.json').exists():
        write(DEST / 'environment.json', {
            'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
            'cpuinfo': Path('/proc/cpuinfo').read_text(),
            'thread_environment': {k: v for k, v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
            'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode(),
            'started_unix': time.time(), 'validation_gate_hash': digest(OUT / 'validation_gate.json')})
    for repeat in range(REPEATS):
        order = np.random.RandomState(protocol()['order_seed'] + repeat).permutation(len(jobs))
        for i in order:
            task, seed, arm = jobs[i]
            check_idle()
            dest = DEST / ('r%d_%s_%s_s%d' % (repeat, task, arm, seed))
            dest.mkdir(exist_ok=True)
            if (dest / 'completed.json').exists():
                verify_artifact_hashes(dest / 'completed.json')
                continue
            assert not list(dest.glob('trace_*.json')), 'Retain and inspect partial attempt before retry'
            write(dest / 'attempt.json', {'started_unix': time.time(), 'pid': os.getpid(), 'reset_attempts': 0, 'step_attempts': 0})
            counters = read(dest / 'attempt.json')
            source = OUT / ('%s_s%d' % (task, seed)) if arm == 'calibrated_greedy' else model_dir(task, 'fixed' if arm == 'fixed' else 'min_q', seed)
            env = make_env(task, 923, fixed_horizon=BASE_H[task] if arm == 'fixed' else None, aligned=True, scaled_obs=True)
            model = SAC.load(str(source / 'model.zip'), env=env)
            before = weights_hash(model)
            policy = Greedy(model) if arm.endswith('greedy') else model
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
            original = env.control_system.controller.get_action
            measured = []
            def timed_control(*args, **kwargs):
                start = time.perf_counter()
                result = original(*args, **kwargs)
                measured.append(time.perf_counter() - start)
                return result
            env.control_system.controller.get_action = timed_control
            cases = read(bank_path(task, 'validation'))['cases']
            episodes = []
            reference = OUT / 'evaluations/validation' / task / ('%s_s%d' % (arm, seed))
            for j, case in enumerate(cases):
                counters['reset_attempts'] += 1
                write(dest / 'attempt.json', counters)
                start = time.perf_counter()
                obs = env.reset(**copy.deepcopy(case))
                reset_s = time.perf_counter() - start
                assert len(measured) == 1
                reset_controller_s = measured.pop()
                if j == 0: policy.predict(obs, deterministic=True)
                trace = []
                for t in range(env.max_steps):
                    # Durable accounting is outside every timed segment.
                    counters['step_attempts'] += 1
                    write(dest / 'attempt.json', counters)
                    start = time.perf_counter()
                    h = BASE_H[task] if arm == 'fixed' else actor_h(policy, obs)
                    selection_s = time.perf_counter() - start
                    start = time.perf_counter()
                    obs, done, row = observed_step(env, task, h, case, t)
                    environment_s = time.perf_counter() - start
                    assert len(measured) == 1
                    controller_s = measured.pop()
                    row['timing'] = {'selection_s': selection_s, 'controller_s': controller_s,
                                     'decision_s': selection_s + controller_s, 'environment_s': environment_s}
                    trace.append(row)
                    if done: break
                assert done
                write(dest / ('trace_%02d.json' % j), trace)
                audit_trace(task, case, trace)
                expected = read(reference / ('trace_%02d.json' % j))
                assert len(trace) == len(expected)
                for actual, ref in zip(trace, expected):
                    assert {k: v for k, v in actual.items() if k != 'timing'} == {k: v for k, v in ref.items() if k != 'elapsed_s'}
                times = np.array([r['timing']['decision_s'] for r in trace])
                episodes.append({'case': j, 'steps': len(trace), 'termination': trace[-1]['termination'],
                    'total_cost': sum(-r['reward'] for r in trace), 'solver_failures': sum(not r['solver_success'] for r in trace),
                    'decision_total_s': float(times.sum()), 'decision_mean_s': float(times.mean()),
                    'decision_median_s': float(np.median(times)), 'decision_p95_s': float(np.percentile(times, 95)),
                    'deadline_exceed_steps': int(sum(times > (.1 if task == 'vehicle' else .04))),
                    'reset_s': reset_s, 'reset_controller_s': reset_controller_s})
            assert weights_hash(model) == before
            model.sess.close()
            write(dest / 'summary.json', {'task': task, 'seed': seed, 'arm': arm, 'repeat': repeat, 'episodes': episodes})
            paths = list(dest.glob('trace_*.json')) + [dest / 'summary.json', source / 'model.zip', bank_path(task, 'validation'), reference / 'completed.json']
            write(dest / 'completed.json', {'passed': True, 'model_hash': before, 'exact_replay': True,
                'hashes': {str(p): digest(p) for p in paths}})
            print(dest.name + ' timing and exact replay passed', flush=True)
    paths = list(DEST.glob('r*/completed.json'))
    assert len(paths) == len(jobs) * REPEATS
    write(DEST / 'completed.json', {'passed': True, 'conditions': len(paths),
        'hashes': {str(p): digest(p) for p in paths}})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    args = ap.parse_args()
    if args.prepare: prepare()
    else: run()
