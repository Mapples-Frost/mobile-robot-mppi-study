"""Training-only policy-improvement mechanism pilot from strong fixed terminals.

No learned tail, entropy bonus, test access, or policy efficacy claim.
"""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from runtime import ART, ROOT, imports, make_env
import numpy as np
from run import snapshot, weights_hash, write
from min_q_eval_suite import model_dir
from paper_h_soft_probe import digest, read
from branch_calibration_run import meter, observed_step
from branch_calibration_audit import audit_trace

OUT = ART / 'results/fixed_policy_branches_2026-09-24'
TASKS = ('vehicle', 'pendulum')
BASE = {'vehicle': 25, 'pendulum': 30}
HS = (1, 5, 10, 20, 25, 30, 40, 50)
ANCHORS = (0, 25, 60)


def protocol():
    return dict(stage='Training-only mechanism pilot, method extension', tasks=TASKS,
        seeds=[0, 1, 2], cases_per_seed=2, anchors=ANCHORS, candidates=HS,
        base_h=BASE, train_seed_base=2609243000, smoke_seed_base=2609242998,
        scene_seed_rule='base + task_index*100 + training_seed; smoke base + task_index',
        terminal='Reuse each independently trained 15000-step selected fixed-H terminal unchanged',
        intervention='Force H for one step, then restore fixed H until actual termination',
        target='Finite undiscounted sum of physical cost + paper H penalty + constraint penalty; no entropy or learned tail',
        metrics=['total_cost', 'performance_cost', 'h_penalty', 'constraint_cost',
                 'success', 'constraint', 'solver_failure_steps', 'mean_horizon'],
        cost_gate='Mechanism promising only if safety-admissible oracle reductions sum to >=1% of source suffix cost in every seed of a task. Descriptive training gate, not efficacy.',
        admissible='Do not reduce goal/survival success, introduce constraints, or increase solver-failure steps relative to the paired baseline suffix.',
        controls='Exact deterministic prefix replay; baseline branch equals saved source suffix; smoke repeats every branch exactly.',
        selection='Inspect all seeds/anchors/failures. Do not discard early termination; skip only anchors after termination. No validation/test or policy selection in this pilot.',
        future='Any subsequent method must register fresh validation and sealed test before collecting evaluation outcomes.',
        timing='Collection wall time is budget only. No acceleration inference from H or parallel collection timing.',
        budget='All explicit reset/step attempts durably counted, including bank generation and replay. Env construction separately disclosed. Maximum 37500 formal step calls before failures/skips (12 sources and 288 branches with prefixes).',
        workers=2, timeout_seconds_per_job=14400)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT / 'experiments/bohn2021_reproduction').glob('*.py'))
    paths += list((ART / 'sources').glob('*/**/*.py'))
    for task in TASKS:
        paths.append(ART / 'configs' / (task + '.json'))
        for seed in range(3):
            source = model_dir(task, 'fixed', seed)
            manifest, done = read(source / 'manifest.json'), read(source / 'completed.json')
            assert done['status'] == 'complete' and done['steps'] == 15000
            assert manifest['fixed_horizon'] == BASE[task] and manifest['seed'] == seed
            paths.extend(source / name for name in ('manifest.json', 'completed.json', 'model.zip'))
    p = json.loads(json.dumps(protocol()))
    hashes = {str(path): digest(path) for path in sorted(set(paths))}
    for name, data in [('protocol.json', p), ('inputs_sha256.json', hashes)]:
        path = OUT / name
        if path.exists(): assert read(path) == data, 'Frozen input changed: ' + name
        else: write(path, data)


def verify():
    assert read(OUT / 'protocol.json') == json.loads(json.dumps(protocol()))
    for path, expected in read(OUT / 'inputs_sha256.json').items():
        assert digest(Path(path)) == expected, path


def metrics(task, trace):
    term = trace[-1]['termination']
    return dict(total_cost=sum(-r['reward'] for r in trace),
        performance_cost=sum(r['performance'] for r in trace),
        h_penalty=sum(r['compute'] for r in trace),
        constraint_cost=sum(r['constraint'] for r in trace),
        success=term == ('goal' if task == 'vehicle' else 'steps'),
        constraint=term == 'constraint', termination=term,
        solver_failure_steps=sum(not r['solver_success'] for r in trace),
        mean_horizon=float(np.mean([r['horizon'] for r in trace])), steps=len(trace))


def job(task, seed, smoke=False):
    verify()
    dest = OUT / ('smoke_%s' % task if smoke else '%s_s%d' % (task, seed))
    dest.mkdir(exist_ok=True)
    lock = (dest / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest / 'completed.json').exists():
        for path, expected in read(dest / 'completed.json')['hashes'].items():
            assert digest(Path(path)) == expected
        print('Already audited complete: ' + str(dest), flush=True)
        return
    start = time.time()
    write(dest / 'running.json', dict(pid=os.getpid(), started=start))
    env = make_env(task, seed, aligned=True, scaled_obs=True)
    meter(env, dest)
    _, SAC, _ = imports()
    source = model_dir(task, 'fixed', seed)
    model = SAC.load(str(source / 'model.zip'))
    assert weights_hash(model) == read(source / 'completed.json')['final_hash']
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    model.sess.close()
    bank_path = dest / 'train_bank.json'
    if not bank_path.exists():
        scene_seed = 2609242998 + TASKS.index(task) if smoke else 2609243000 + TASKS.index(task)*100 + seed
        env.seed(scene_seed)
        np.random.seed(scene_seed)
        cases = []
        for _ in range(1 if smoke else 2):
            env.reset()
            cases.append(snapshot(env))
        write(bank_path, dict(seed=scene_seed, split='smoke' if smoke else 'training', cases=cases))
    cases = read(bank_path)['cases']
    groups, skipped = [], []
    hashes = {str(bank_path): digest(bank_path)}
    for cid, case in enumerate(cases):
        source_path = dest / ('source_%02d.json' % cid)
        if not source_path.exists():
            env.reset(**copy.deepcopy(case))
            trace = []
            for t in range(env.max_steps):
                _, done, row = observed_step(env, task, BASE[task], case, t)
                trace.append(row)
                if done: break
            assert done
            audit_trace(task, case, trace)
            write(source_path, trace)
        original = read(source_path)
        audit_trace(task, case, original)
        assert all(r['horizon'] == BASE[task] for r in original)
        hashes[str(source_path)] = digest(source_path)
        for anchor in ((0, 25) if smoke else ANCHORS):
            if anchor >= len(original):
                skipped.append(dict(case=cid, anchor=anchor, source=metrics(task, original)))
                continue
            candidates = sorted(set((10, BASE[task]) if smoke else HS))
            branches = {}
            for h in candidates:
                branch_path = dest / ('case%02d_t%03d_h%02d.json' % (cid, anchor, h))
                def rollout():
                    obs = env.reset(**copy.deepcopy(case))
                    for t in range(anchor):
                        obs, done, row = observed_step(env, task, BASE[task], case, t)
                        assert not done and row == original[t], ('prefix mismatch', cid, anchor, h, t)
                    assert obs.tolist() == original[anchor]['observation']
                    trace = []
                    for t in range(anchor, env.max_steps):
                        _, done, row = observed_step(env, task, h if t == anchor else BASE[task], case, t)
                        trace.append(row)
                        if done: break
                    assert done
                    audit_trace(task, case, trace, offset=anchor)
                    return trace
                if not branch_path.exists():
                    trace = rollout()
                    if smoke: assert rollout() == trace, 'Nondeterministic branch replay'
                    write(branch_path, dict(case=cid, anchor=anchor, first_h=h,
                        trace=trace, metrics=metrics(task, trace), exact_repeat=smoke))
                b = read(branch_path)
                assert (b['case'], b['anchor'], b['first_h']) == (cid, anchor, h)
                audit_trace(task, case, b['trace'], offset=anchor)
                assert b['metrics'] == metrics(task, b['trace'])
                assert b['trace'][0]['horizon'] == h
                assert all(r['horizon'] == BASE[task] for r in b['trace'][1:])
                if h == BASE[task]: assert b['trace'] == original[anchor:], 'Baseline branch differs from source suffix'
                hashes[str(branch_path)] = digest(branch_path)
                branches[str(h)] = b['metrics']
                print(json.dumps(dict(task=task, seed=seed, case=cid, anchor=anchor, h=h,
                    cost=b['metrics']['total_cost'])), flush=True)
            groups.append(dict(case=cid, anchor=anchor, observation=original[anchor]['observation'], branches=branches))
    attempts = [read(p) for p in dest.glob('attempt_*.json')]
    write(dest / 'completed.json', dict(task=task, seed=seed, smoke=smoke,
        audit_passed=True, groups=groups, skipped=skipped, hashes=hashes,
        inputs_hash=digest(OUT / 'inputs_sha256.json'),
        explicit_step_calls=sum(x['step_calls'] for x in attempts),
        explicit_reset_calls=sum(x['reset_calls'] for x in attempts),
        environment_constructions=len(attempts), training_updates=0,
        elapsed_latest_attempt_s=time.time()-start))


def suite(smoke):
    verify()
    if not smoke:
        for task in TASKS:
            assert read(OUT / ('smoke_' + task) / 'completed.json')['audit_passed']
    jobs = [(task, seed) for task in TASKS for seed in ([0] if smoke else range(3))]
    def launch(pair):
        task, seed = pair
        cmd = [sys.executable, '-u', __file__, '--mode', 'job', '--task', task, '--seed', str(seed)]
        if smoke: cmd.append('--smoke')
        log = OUT / ('%s_%s_s%d_%d.log' % ('smoke' if smoke else 'pilot', task, seed, time.time_ns()))
        with log.open('w') as stream:
            p = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=14400)
        return dict(task=task, seed=seed, exit_code=p.returncode, log=str(log))
    status = dict(smoke=smoke, pid=os.getpid(), finished=[], active=True)
    path = OUT / ('smoke_status.json' if smoke else 'pilot_status.json')
    write(path, status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(launch, j) for j in jobs]):
            status['finished'].append(future.result())
            write(path, status)
    status['active'] = False
    status['complete'] = all(r['exit_code'] == 0 for r in status['finished'])
    write(path, status)
    assert status['complete'], 'Inspect failure logs before resuming'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['freeze', 'suite', 'job'], required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--task', choices=TASKS)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'suite': suite(args.smoke)
    else: job(args.task, args.seed, args.smoke)
