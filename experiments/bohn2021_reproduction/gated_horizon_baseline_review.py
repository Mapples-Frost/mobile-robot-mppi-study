"""Independent fixed-H nomination/model audit after validation and serial timing.

Does not import evaluation/selection functions, simulate, retrain, or open tests.
The frozen production selection and all experiment outputs remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
BASE = {'vehicle': 25, 'pendulum': 30}
HS = tuple(range(5, 51, 5))
DEST = RUN / 'baseline_selection_review'
REG = RUN / 'baseline_selection_review_registration.json'
SETTINGS = dict(aligned=True, scaled_obs=True, ent_coef='1.0', no_online_value=False,
                batch_size=256, buffer_size=1000000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def resolve(name):
    path = ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def registration():
    paths = [Path(__file__), RUN / 'protocol.json', RUN / 'inputs_sha256.json', RUN / 'evaluation_registration.json',
             RUN / 'baseline_completion_registration.json']
    return dict(source_hashes={relative(p): sha(p) for p in paths},
                design='Recompute every10-H independent seed0 and matched3-seed fixed summary and eligibility from audited32-episode records; exact raw-cost ordering with smaller-H tie break. Check all selected independent seeds have15k terminal training, matched arms share the correct reference terminal, and model provenance is recorded.',
                prerequisites='Complete validation/timing continuation, all controller PIDs absent, passing validation audit, no pending independent seeds. Guard runs before reading outcome summaries.',
                scope='Independent selection/model bookkeeping only. Existing dynamics audit is a prerequisite, not repeated. No efficacy claim, timing experiment, simulation, or test access.')


def guard():
    for name in ('validation_finish_status.json', 'validation_status.json', 'posttrain_status.json',
                 'baseline_completion_status.json'):
        state = read(RUN / name)
        if state.get('active') or not state.get('complete'):
            raise RuntimeError('PENDING: ' + name + ' must be complete before reading validation scores')
        proc = Path('/proc') if Path('/proc').exists() else Path('//wsl.localhost/Ubuntu-20.04/proc')
        if (proc / str(state['pid']) / 'cmdline').exists():
            raise RuntimeError('PENDING: prior controller PID still present: ' + str(state['pid']))


def aggregate(episodes, n=32):
    assert len(episodes) == n and [e['case'] for e in episodes] == list(range(n)), 'Incomplete/duplicate cases'
    counts = {'steps': 'steps', 'success': 'success', 'constraints': 'constraint',
              'initial_failures': 'initial_failed_steps', 'final_failures': 'solver_failure_steps',
              'switches': 'switches', 'retries': 'retries'}
    for row in episodes:
        for field in counts.values():
            v = row[field]
            assert isinstance(v, (int, bool)) and v >= 0, (field, v)
        assert row['steps'] > 0 and row['success'] in (0, 1) and row['constraint'] in (0, 1)
        assert row['initial_failed_steps'] <= row['steps'] and row['solver_failure_steps'] <= row['steps']
        assert all(math.isfinite(row[k]) for k in ('total_cost', 'physical_constraint_cost'))
    result = {key: sum(e[field] for e in episodes) for key, field in counts.items()}
    result.update(total_cost=float(np.mean([e['total_cost'] for e in episodes])),
                  physical_constraint_cost=float(np.mean([e['physical_constraint_cost'] for e in episodes])))
    result['initial_failure_rate'] = result['initial_failures'] / result['steps']
    result['final_failure_rate'] = result['final_failures'] / result['steps']
    return result


def exclusions(candidate, reference):
    reasons = []
    if candidate['success'] < reference['success']:
        reasons.append('fewer_successes')
    if candidate['constraints'] > reference['constraints']:
        reasons.append('more_constraints')
    for field in ('initial_failure_rate', 'final_failure_rate'):
        if candidate[field] > reference[field] + 1e-12:
            reasons.append('higher_' + field)
    return reasons


def nomination(candidates, reference):
    eligible = [h for h, arms in candidates.items()
                if all(not exclusions(a, b) for a, b in zip(arms, reference)) and len(arms) == len(reference)]
    assert eligible, 'No eligible fixed baseline'
    return min(eligible, key=lambda h: (float(np.mean([a['total_cost'] for a in candidates[h]])), h))


def audit():
    guard()
    assert read(REG) == registration(), 'Review source/registration changed'
    hashes = {}

    def record(path, expected=None):
        key = relative(path)
        actual = sha(path)
        if expected is not None:
            assert actual == expected, 'Hash mismatch: ' + key
        if key in hashes:
            assert hashes[key] == actual, 'Input changed while auditing'
        hashes[key] = actual

    def graph(mapping):
        for name, expected in mapping.items():
            record(resolve(name), expected)

    # Compare inherited model manifests/completions with their pre-validation hashes.
    # Source amendments do not authorize changes to these model artifacts.
    frozen_models = {}
    for mapping in (read(RUN / 'inputs_sha256.json'), read(RUN / 'evaluation_registration.json')['hashes']):
        for name, expected_hash in mapping.items():
            if Path(name).name in ('model.zip', 'manifest.json', 'completed.json'):
                path = resolve(name)
                if path in frozen_models:
                    assert frozen_models[path] == expected_hash
                frozen_models[path] = expected_hash
                record(path, expected_hash)

    finish = read(RUN / 'validation_finish_status.json')
    graph(finish['output_hashes'])
    audit_data = read(RUN / 'audit_validation.json')
    assert audit_data['passed']
    graph(audit_data['hashes'])
    selection = read(RUN / 'baseline_selection.json')
    assert not selection['pending_independent_baselines']
    graph(selection['hashes'])
    for p in (REG, RUN / 'baseline_selection.json', RUN / 'audit_validation.json', RUN / 'validation_finish_status.json'):
        record(p)
    groups = {(g['task'], g['family'], g['seed'], g['h']): g for g in audit_data['groups']}
    assert len(groups) == audit_data['conditions'] == len(audit_data['groups'])
    expected = {(t, f, s, h) for t in BASE for s in range(3)
                for f, h in [('adaptive', 0), ('primary', BASE[t])] + [('matched', h) for h in HS if h != BASE[t]]}
    expected |= {(t, 'grid', 0, h) for t in BASE for h in HS if h != BASE[t]}
    expected |= {(t, 'grid', s, selection['nominations'][t]['independent_h']) for t in BASE for s in (1, 2)
                 if selection['nominations'][t]['independent_h'] != BASE[t]}
    assert set(groups) == expected and len(expected) >= 84
    cache, models, rows, result = {}, {}, [], {}

    def arm(task, family, seed, h):
        family = 'primary' if h == BASE[task] else family
        key = (task, family, seed, h)
        if key in cache:
            return cache[key]
        assert key in groups
        folder = RUN / 'evaluations/validation' / task / ('%s_h%d_s%d' % (family, h, seed))
        done = read(folder / 'completed.json')
        assert done['passed']
        graph(done['hashes'])
        summary = read(folder / 'summary.json')
        assert (summary['task'], summary['family'], summary['seed'], summary['h']) == key
        assert summary['split'] == 'validation' and summary['episodes'] == done['result']['episodes']
        values = aggregate(summary['episodes'])
        assert values['steps'] == groups[key]['steps'] and groups[key]['episodes'] == 32
        assert values['switches'] == 0, 'Fixed arm changed H'
        model_paths = [resolve(p) for p in done['hashes'] if Path(p).name == 'model.zip']
        assert len(model_paths) == 1
        model = model_paths[0]
        manifest, complete = (read(model.parent / name) for name in ('manifest.json', 'completed.json'))
        training_h = h if family in ('grid', 'primary') else BASE[task]
        assert manifest['seed'] == seed and manifest['fixed_horizon'] == training_h
        assert manifest['steps'] == complete['steps'] == 15000 and manifest['adaptations'] == SETTINGS
        assert complete['status'] == 'complete' and complete['final_hash'] == done['model_hash']
        for name in ('manifest.json', 'completed.json'):
            record(model.parent / name)
        if not (family == 'grid' and seed > 0):
            assert all((model.parent / name) in frozen_models for name in ('model.zip', 'manifest.json', 'completed.json'))
        if family == 'grid' and seed > 0:
            log = model.parent / 'independent_log_audit.json'
            trained = read(log)
            assert trained['passed'] and trained['training_steps'] == 15000
            assert (trained['task'], trained['seed'], trained['fixed_h']) == (task, seed, h)
            graph(trained['hashes'])
            record(log)
        models[relative(model)] = dict(task=task, seed=seed, training_h=training_h, training_steps=15000,
                                      final_parameter_hash=done['model_hash'], model_sha256=sha(model))
        values.update(task=task, family=family, seed=seed, h=h, path=relative(folder), model=relative(model))
        cache[key] = values
        return values

    def compare_saved(actual, saved):
        for key in ('task', 'family', 'seed', 'h', 'total_cost', 'physical_constraint_cost', 'success',
                    'constraints', 'steps', 'initial_failures', 'final_failures', 'initial_failure_rate',
                    'final_failure_rate', 'switches', 'retries'):
            assert actual[key] == saved[key], ('Saved nomination summary mismatch', key)
        assert resolve(saved['path']) == ROOT / actual['path']

    for task in BASE:
        references = [arm(task, 'primary', s, BASE[task]) for s in range(3)]
        grid = {h: [arm(task, 'grid', 0, h)] for h in HS}
        matched = {h: [arm(task, 'matched', s, h) for s in range(3)] for h in HS}
        chosen_grid = nomination(grid, references[:1])
        chosen_matched = nomination(matched, references)
        saved = selection['nominations'][task]
        assert (chosen_grid, chosen_matched) == (saved['independent_h'], saved['matched_h'])
        assert [a['h'] for a in saved['independent_grid']] == list(HS)
        assert set(saved['matched_grid']) == {str(h) for h in HS}
        for h, saved_arm in zip(HS, saved['independent_grid']):
            compare_saved(grid[h][0], saved_arm)
        for h in HS:
            assert len(saved['matched_grid'][str(h)]) == 3
            for seed, a in enumerate(matched[h]):
                compare_saved(a, saved['matched_grid'][str(h)][seed])
                assert a['model'] == references[seed]['model'], 'Matched terminal differs from reference'
        for label, candidates, refs, chosen in (('independent', grid, references[:1], chosen_grid),
                                               ('matched', matched, references, chosen_matched)):
            for h, arms in candidates.items():
                reasons = ['s%d:%s' % (a['seed'], reason) for a, b in zip(arms, refs) for reason in exclusions(a, b)]
                rows.append(dict(task=task, comparator=label, h=h, eligible=not reasons, selected=h == chosen,
                                 ranking_cost=float(np.mean([a['total_cost'] for a in arms])),
                                 exclusion_reasons=';'.join(reasons), seeds=';'.join(str(a['seed']) for a in arms)))
        selected_independent = [arm(task, 'grid', s, chosen_grid) for s in range(3)]
        selected_matched = matched[chosen_matched]
        result[task] = dict(independent_h=chosen_grid, matched_h=chosen_matched,
                            selected_independent_models=[a['model'] for a in selected_independent],
                            selected_matched_models=[a['model'] for a in selected_matched],
                            identical_comparator_arms=[a['path'] == b['path'] for a, b in zip(selected_independent, selected_matched)])
    assert len(rows) == 40
    report = dict(passed=True, source_hash=sha(Path(__file__)), nominations=result,
                  unique_fixed_conditions=len(cache), audited_total_conditions=len(groups),
                  inspected_fixed_episodes=32 * len(cache), unique_terminal_models=len(models),
                  terminal_training_steps_referenced=15000 * len(models), hashes=hashes,
                  new_simulation_steps=0, new_training_steps=0, new_numerical_integrations=0,
                  scope='Model-training total is referenced provenance, not new or whole-study budget. Independent grid selection uses seed0; matched selection uses all3 seeds. Matching labels can reference identical arms and do not increase independent evidence. Passing this review does not imply efficacy or successful reproduction.')
    if (DEST / 'report.json').exists():
        assert read(DEST / 'report.json') == report, 'Existing review differs; inspect before changing'
        print('Existing independent baseline review verified; no rewrite')
        return
    for p, value in hashes.items():
        assert sha(resolve(p)) == value, 'Input changed during review'
    DEST.mkdir(exist_ok=True)
    for name, values in [('all_h_candidates.csv', rows), ('terminal_models.csv', [dict(path=p, **v) for p, v in models.items()])]:
        with (DEST / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(DEST / 'report.json', report)
    lines = ['# 固定H选择与终端模型独立核验', '',
             '十值固定H网格的资格、原始成本排序、平局规则与最终提名独立重算一致。所有所选独立固定基线均有三个15k步训练种子的模型证据，共用终端比较器逐项核对来源。', '',
             '|任务|独立H|共用终端H|三种子中比较臂完全相同|', '|---|---:|---:|---|']
    for task, value in result.items():
        lines.append('|%s|%d|%d|%s|' % (task, value['independent_h'], value['matched_h'], value['identical_comparator_arms']))
    lines += ['', '全部40个任务/比较器/H候选（含不合格项）及排除原因见CSV；按唯一模型路径列出训练来源，不把标签重复算为独立证据。', '',
              '本项只核对已通过动力学审计的回合汇总及选择/模型账目，没有重跑控制仿真、训练或数值积分；不替代完整效果与独立测试。独立网格由种子0提名，随后补齐所选H的其他种子，这一选择范围保持公开。']
    (DEST / 'report_CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'hashes'}, indent=2))


def self_check():
    reference = dict(total_cost=10., success=32, constraints=0, initial_failure_rate=0., final_failure_rate=0.)
    tests = []
    candidates = {h: [dict(reference, total_cost=10. + h)] for h in HS}
    candidates[10][0]['total_cost'] = -50.
    for field, value in [('success', 31), ('constraints', 1), ('initial_failure_rate', .01), ('final_failure_rate', .01)]:
        altered = {h: [dict(v[0])] for h, v in candidates.items()}
        altered[10][0][field] = value
        assert nomination(altered, [reference]) == 5
        tests.append('reject_' + field)
    tied = {h: [dict(reference)] for h in HS}
    assert nomination(tied, [reference]) == 5
    tests.append('exact_cost_tie_chooses_smaller_h')
    three = {h: [dict(reference, total_cost=float(h)) for _ in range(3)] for h in HS}
    three[5][2]['constraints'] = 1
    assert nomination(three, [reference] * 3) == 10
    tests.append('matched_excludes_one_seed_safety_loss')
    episode = dict(steps=10, success=True, constraint=False, initial_failed_steps=0, solver_failure_steps=0,
                   switches=0, retries=0, total_cost=-1., physical_constraint_cost=-2.)
    valid = [dict(episode, case=i) for i in range(32)]
    assert aggregate(valid)['total_cost'] == -1.
    for invalid in (valid[:-1], [dict(e, case=0) for e in valid], [dict(e, total_cost=float('nan')) for e in valid]):
        try:
            aggregate(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid episode set accepted')
    tests += ['negative_cost_retained', 'missing_duplicate_nonfinite_episodes_rejected']
    print(json.dumps(dict(passed=True, checks=tests, real_outcomes_read=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['register', 'self-check', 'audit'], required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        value = registration()
        if REG.exists():
            assert read(REG) == value
        else:
            write(REG, value)
        print('Independent baseline review registered; no outcome summaries read')
    elif args.mode == 'self-check':
        self_check()
    else:
        audit()
