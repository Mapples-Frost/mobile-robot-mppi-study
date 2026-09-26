"""Serial formal rollouts and independently repeated timing for frozen trees."""
import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

from runtime import imports
from latency_tree_evaluation_spec import (
    OUT, REG, EVAL_REG, TASKS, BASE, read, write, sha, bank_name, arm_name,
    initial_jobs, smoke_jobs, selected_jobs, normalize, source_for, check_model,
    deployed_policy, verify_hashes, verify_evaluation, freeze_policies,
    confirmation, result_folder, frozen_write,
)
from latency_tree_policy import choose, features
from latency_tree_run import acquire, clean
from conservative_canonical_reset import make_env
from conservative_iteration_timing import idle
from branch_calibration_run import meter, observed_step
from relative_policy_features import context
from gated_horizon_timing import LoggingTimer
from gated_horizon_shared_reset import install as install_shared
from gated_horizon_search import case_metrics
from run import weights_hash, serial
import conservative_solver_recovery as recovery_module
import numpy as np


def verify_complete(folder):
    value = read(folder / 'completed.json')
    assert value['passed']
    verify_hashes(value['hashes'])
    return value


def condition(job, split, folder, repeats=1, reference=None):
    """Same measured selection/control boundaries as the registered training runner."""
    task, family, seed, h = normalize(*job)
    assert split in ('smoke', 'validation', 'test')
    assert repeats in (1, 2)
    idle()
    lock = acquire(folder)
    if (folder / 'completed.json').exists():
        done = verify_complete(folder)
        summary = read(folder / 'summary.json')
        assert (summary['task'], summary['family'], summary['seed'], summary['h'],
                summary['split'], summary['repeats']) == (task, family, seed, h, split, repeats)
        assert done['reference'] == (str(reference) if reference else None)
        return summary
    assert not [p for p in folder.iterdir() if p.name != 'run.lock'], (
        'Preserve and inspect partial rollout before recovery', str(folder))
    bank = bank_name(task, split)
    assert sha(bank) == read(OUT / 'banks/completed.json')['hashes'][str(bank)]
    cases = read(bank)['cases']
    assert len(cases) == {'smoke': 2, 'validation': 64, 'test': 128}[split]
    if split == 'test':
        assert list(job) in confirmation()['jobs']
    policy = deployed_policy(task, family, seed, h)
    source = source_for(task, family, seed, h)
    trained_h = h if family == 'grid' else BASE[task]
    done = check_model(source, task, trained_h, seed)
    if reference is not None:
        verify_complete(reference)
        saved = read(reference / 'summary.json')
        assert (saved['task'], saved['family'], saved['seed'], saved['h'], saved['split']) == (
            task, family, seed, h, split)
        assert saved['policy'] == policy
    write(folder / 'policy.json', policy)
    env = make_env(task, seed, aligned=True, scaled_obs=True)
    counts = meter(env, folder)
    _, SAC, _ = imports()
    model = SAC.load(str(source / 'model.zip'))
    try:
        assert weights_hash(model) == done['final_hash']
        terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:
        model.sess.close()
    env.set_value_function_weights_and_biases(*terminal)
    reset_adapter = install_shared(
        env, task, seed, terminal, independent_terminal=family == 'grid' and h != BASE[task])
    controller = env.control_system.controller
    original = controller.get_action
    measured, episodes = [], []
    write(folder / 'environment.json', dict(
        pid=os.getpid(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), started=time.time(),
        thread_environment={k: v for k, v in os.environ.items()
                            if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
        constructor_calls_unmeasured=True, model_source=str(source),
        terminal_hash=done['final_hash'], reset_adapter=reset_adapter,
        evaluation_times_are_not_formal_timing=reference is None,
    ))
    with LoggingTimer(folder) as logging:
        recovery = recovery_module.install(controller.mpc, logging)

        def timed(*args, **kwargs):
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            assert 0 <= logged < gross
            measured.append(dict(controller_gross_s=gross, logging_s=logged,
                                 controller_s=gross - logged))
            return value

        controller.get_action = timed
        for repeat in range(repeats):
            for cid, case in enumerate(cases):
                recovery.update(enabled=False, events=[], case=cid, step=-1)
                start = time.perf_counter()
                env.reset(**copy.deepcopy(case))
                reset_s = time.perf_counter() - start
                assert len(measured) == 1
                reset = dict(reset_gross_s=reset_s, **measured.pop())
                recovery['enabled'] = True
                trace = []
                previous_initial = previous_final = False
                rawpath = folder / ('r%d_trace_%02d.jsonl' % (repeat, cid))
                with rawpath.open('x') as stream:
                    for t in range(env.max_steps):
                        recovery['step'] = t
                        start = time.perf_counter()
                        if policy['kind'] in ('tree', 'combined'):
                            ctx = context(env, task)
                            ctx.update(previous_initial_failure=previous_initial,
                                       previous_final_failure=previous_final)
                            selected, decision = choose(policy, ctx)
                        else:
                            selected, decision = choose(
                                policy, {'state': env.control_system.current_state})
                            ctx = None
                        selection_s = time.perf_counter() - start
                        if ctx is None:
                            ctx = context(env, task)
                            ctx.update(previous_initial_failure=previous_initial,
                                       previous_final_failure=previous_final)
                        _, terminated, row = observed_step(env, task, selected, case, t)
                        assert len(measured) == 1
                        timing = measured.pop()
                        timing.update(selection_s=selection_s,
                                      decision_s=selection_s + timing['controller_s'],
                                      decision_gross_s=selection_s + timing['controller_gross_s'])
                        row.update(policy_context=ctx, tree_features=features(task, ctx),
                                   decision=decision, recovery=recovery['events'][-1], timing=timing)
                        stream.write(json.dumps(row, default=serial, allow_nan=False) + '\n')
                        stream.flush()
                        trace.append(row)
                        previous_initial = not row['recovery']['attempts'][0]['success']
                        previous_final = not row['solver_success']
                        if terminated:
                            break
                assert terminated
                # Preserve the whole episode before exact replay checks can fail.
                write(folder / ('r%d_trace_%02d.json' % (repeat, cid)), trace)
                write(folder / ('r%d_reset_%02d.json' % (repeat, cid)), reset)
                if repeat:
                    assert clean(trace) == clean(read(folder / ('r0_trace_%02d.json' % cid)))
                if reference is not None:
                    assert clean(trace) == clean(read(reference / ('r0_trace_%02d.json' % cid))), (
                        'Non-time replay mismatch', str(folder), cid)
                values = np.asarray([r['timing']['decision_s'] for r in trace])
                ep = case_metrics(task, trace)
                ep.update(case=cid, repeat=repeat, decision_total_s=float(values.sum()),
                          decision_mean_s=float(values.mean()),
                          gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),
                          logging_total_s=sum(r['timing']['logging_s'] for r in trace),
                          decision_p95_s=float(np.percentile(values, 95)),
                          deadline_exceed_steps=int(sum(values > (.1 if task == 'vehicle' else .04))))
                episodes.append(ep)
                write(folder / 'progress.json', dict(
                    pid=os.getpid(), episodes=len(episodes), expected=len(cases) * repeats,
                    steps=counts['step_calls']))
        assert logging.operations == 1 + 3 * recovery['counts']['solve_completed']
        summary = dict(task=task, family=family, seed=seed, h=h, split=split,
                       policy=policy, episodes=episodes, repeats=repeats,
                       steps=counts['step_calls'], resets=counts['reset_calls'],
                       solver_counts=recovery['counts'], logging_operations=logging.operations,
                       bank=str(bank), test_accessed=split == 'test')
        write(folder / 'summary.json', summary)
    files = [p for p in folder.iterdir() if p.is_file()
             and p.name not in ('run.lock', 'completed.json', 'progress.json')]
    files += [REG, EVAL_REG, OUT / 'fitted_policy_registration.json', bank]
    files += [source / n for n in ('model.zip', 'manifest.json', 'completed.json')]
    if reference is not None:
        files.append(reference / 'completed.json')
    if split == 'test':
        files.append(OUT / 'confirmation_registration.json')
    write(folder / 'completed.json', dict(
        passed=True, hashes={str(p): sha(p) for p in files},
        reference=str(reference) if reference else None,
        exact_replay=reference is not None or repeats == 2,
        record_audit_pending=True, formal_effect_evidence=False))
    print(json.dumps(dict(condition=str(folder.relative_to(OUT)), episodes=len(episodes),
                          steps=summary['steps'])), flush=True)
    return summary


def suite(split, stage):
    freeze_policies()
    delivery_review = read(OUT / 'training_delivery/independent_review.json')
    assert delivery_review['passed'] and delivery_review['training_only']
    assert not delivery_review['validation_accessed'] and not delivery_review['test_accessed']
    verify_hashes(delivery_review['hashes'])
    delivery = read(OUT / 'training_delivery/manifest.json')
    assert delivery['passed'] and delivery['training_only']
    verify_hashes(delivery['hashes'])
    verify_hashes(delivery['source_hashes'])
    idle()
    assert stage in ('smoke', 'initial', 'supplement', 'timing')
    if stage == 'smoke':
        assert split == 'smoke'
        jobs, root, repeats = smoke_jobs(), OUT / 'evaluation_smoke', 2
    else:
        assert split in ('validation', 'test')
        assert split != 'test' or stage in ('initial', 'timing')
        smoke = read(OUT / 'audit_evaluation_smoke.json')
        assert smoke['passed']
        verify_hashes(smoke['hashes'])
        if split == 'test':
            frozen = confirmation()
            jobs = [tuple(j) for j in frozen['jobs']]
        elif stage == 'initial':
            jobs = initial_jobs()
        else:
            selection = read(OUT / 'baseline_selection.json')
            verify_hashes(selection['hashes'])
            if stage == 'supplement':
                completed = read(OUT / 'baseline_completion.json')
                assert completed['passed']
                verify_hashes(completed['hashes'])
                jobs = [tuple(j) for j in selection['supplement_jobs']]
            else:
                assert stage == 'timing'
                audited = read(OUT / 'audit_evaluation_validation_full.json')
                assert audited['passed']
                verify_hashes(audited['hashes'])
                jobs = selected_jobs()
        root = OUT / ('timing_' + split if stage == 'timing' else 'evaluations/' + split)
        repeats = 1
        if split == 'test' and stage == 'timing':
            audited = read(OUT / 'audit_evaluation_test_initial.json')
            assert audited['passed']
            verify_hashes(audited['hashes'])
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire(OUT / 'execution_locks' / (split + '_' + stage))
    status_path = OUT / ('evaluation_%s_%s_status.json' % (split, stage))
    if status_path.exists():
        old = read(status_path)
        assert old.get('complete') and not old['active'], 'Partial suite requires preserved recovery audit'
        verify_hashes(old['hashes'])
        return
    if stage == 'timing':
        scheduled = [(r, jobs[int(i)]) for r in range(2)
                     for i in np.random.RandomState(2609268000 + r).permutation(len(jobs))]
    else:
        scheduled = [(0, j) for j in jobs]
    frozen_write(OUT / ('execution_%s_%s.json' % (split, stage)),
                 dict(stage=stage, split=split, schedule=scheduled, repeats_per_condition=repeats,
                      evaluation_registration_sha256=sha(EVAL_REG)))
    state = dict(pid=os.getpid(), active=True, complete=False, stage=stage,
                 split=split, expected_conditions=len(scheduled), completed=[],
                 started=time.time(), test_accessed=split == 'test')
    write(status_path, state)
    try:
        for repeat, job in scheduled:
            task, family, seed, h = job
            if stage in ('smoke', 'timing'):
                name = ('r%d_' % repeat if stage == 'timing' else '') + task + '_' + arm_name(family, seed, h)
                folder = root / name
            else:
                folder = root / task / arm_name(family, seed, h)
            reference = result_folder(*job, split=split) if stage == 'timing' else None
            condition(job, split, folder, repeats=repeats, reference=reference)
            state['completed'].append(str(folder / 'completed.json'))
            write(status_path, state)
        state.update(active=False, complete=True, ended=time.time(),
                     hashes={p: sha(Path(p)) for p in state['completed']})
        write(status_path, state)
    except BaseException as exc:
        state.update(active=False, complete=False, ended=time.time(), exception=repr(exc))
        write(status_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('smoke', 'validation', 'test'), default='validation')
    parser.add_argument('--stage', choices=('smoke', 'initial', 'supplement', 'timing'), required=True)
    args = parser.parse_args()
    suite(args.split, args.stage)
