"""Saved-record audit for the new evaluator, including cross-terminal starts."""
import argparse
import json
from pathlib import Path
from latency_tree_evaluation_spec import (
    OUT, EVAL_REG, TASKS, BASE, read, sha, verify_hashes, freeze_policies,
    frozen_write, initial_jobs, smoke_jobs, selected_jobs, result_folder,
    arm_name, source_for, check_model, deployed_policy, confirmation,
)
from latency_tree_audit import condition as audit_condition, clean


def audit(split, stage):
    freeze_policies()
    assert stage in ('smoke', 'initial', 'full', 'timing')
    if stage == 'smoke':
        assert split == 'smoke'
        jobs = smoke_jobs()
        files = [(j, OUT / 'evaluation_smoke' / (j[0] + '_' + arm_name(j[1], j[2], j[3])), 0)
                 for j in jobs]
        statuses = [OUT / 'evaluation_smoke_smoke_status.json']
    else:
        assert split in ('validation', 'test')
        assert split != 'test' or stage in ('initial', 'timing')
        if split == 'test':
            jobs = [tuple(j) for j in confirmation()['jobs']]
        elif stage == 'initial':
            jobs = initial_jobs()
        else:
            selection = read(OUT / 'baseline_selection.json')
            verify_hashes(selection['hashes'])
            jobs = sorted(set(initial_jobs() + [tuple(j) for j in selection['supplement_jobs']]))
        if stage == 'timing':
            if split == 'validation':
                jobs = selected_jobs()
            files = [(j, OUT / ('timing_' + split) / ('r%d_%s_%s' % (
                r, j[0], arm_name(j[1], j[2], j[3]))), r) for r in range(2) for j in jobs]
            statuses = [OUT / ('evaluation_%s_timing_status.json' % split)]
        else:
            files = [(j, result_folder(*j, split=split), 0) for j in jobs]
            statuses = [OUT / ('evaluation_%s_initial_status.json' % split)]
            if stage == 'full':
                statuses.append(OUT / ('evaluation_%s_supplement_status.json' % split))
    hashes = {str(EVAL_REG): sha(EVAL_REG), str(Path(__file__).resolve()): sha(Path(__file__))}
    for p in statuses:
        state = read(p)
        assert state['complete'] and not state['active']
        verify_hashes(state['hashes'])
        cmdline = Path('/proc') / str(state['pid']) / 'cmdline'
        if cmdline.exists():
            assert b'latency_tree_evaluate.py' not in cmdline.read_bytes()
        hashes[str(p)] = sha(p)
    rows, starts = [], {}
    for job, folder, repeat in files:
        task, family, seed, h = job
        summary = read(folder / 'summary.json')
        assert (summary['task'], summary['family'], summary['seed'], summary['h'], summary['split']) == (
            task, family, seed, h, split)
        assert summary['repeats'] == (2 if stage == 'smoke' else 1)
        assert summary['policy'] == deployed_policy(task, family, seed, h)
        source = source_for(*job)
        check_model(source, task, h if family == 'grid' else BASE[task], seed)
        environment = read(folder / 'environment.json')
        assert environment['model_source'] == str(source)
        assert environment['terminal_hash'] == read(source / 'completed.json')['final_hash']
        assert environment['reset_adapter']['independent_terminal'] == (
            family == 'grid' and h != BASE[task])
        result = audit_condition(folder)
        observed = [dict(case=r['case'], state=r['state'], input=r['input'])
                    for r in result.pop('starts') if r['repeat'] == 0]
        key = task, seed
        if key in starts:
            assert observed == starts[key], ('Unpaired scored initial state', job)
        else:
            starts[key] = observed
        if stage == 'timing':
            reference = result_folder(*job, split=split)
            completed = read(folder / 'completed.json')
            assert completed['exact_replay'] and completed['reference'] == str(reference)
            count = 64 if split == 'validation' else 128
            for cid in range(count):
                name = 'r0_trace_%02d.json' % cid
                assert clean(read(folder / name)) == clean(read(reference / name))
            hashes[str(reference / 'completed.json')] = sha(reference / 'completed.json')
        elif stage != 'smoke':
            assert not read(folder / 'completed.json')['exact_replay']
        result.update(family=family, h=h, timing_repeat=repeat)
        rows.append(result)
        hashes[str(folder / 'completed.json')] = sha(folder / 'completed.json')
        print(json.dumps(result), flush=True)
    expected_episodes = len(files) * (4 if stage == 'smoke' else 64 if split == 'validation' else 128)
    assert sum(r['episodes'] for r in rows) == expected_episodes
    result = dict(passed=True, split=split, stage=stage, conditions=len(rows),
                  episodes=expected_episodes, steps=sum(r['steps'] for r in rows),
                  solves=sum(r['solves'] for r in rows), retries=sum(r['retries'] for r in rows),
                  independent_integrations=sum(r['independent_integrations'] for r in rows),
                  max_dynamics_error=max(r['max_dynamics_error'] for r in rows),
                  shared_initial_state_groups=len(starts), groups=rows, hashes=hashes,
                  test_accessed=split == 'test',
                  scope='Raw costs, causal context, independently integrated dynamics, routing, '
                        'logging subtraction, counters and exact replay; feature transform reuses '
                        'production implementation with separately registered known-answer checks. '
                        'This audit alone does not establish an effect.')
    target = OUT / ('audit_evaluation_smoke.json' if stage == 'smoke'
                    else 'audit_evaluation_%s_%s.json' % (split, stage))
    frozen_write(target, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'hashes')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('smoke', 'validation', 'test'), default='validation')
    parser.add_argument('--stage', choices=('smoke', 'initial', 'full', 'timing'), required=True)
    args = parser.parse_args()
    audit(args.split, args.stage)
