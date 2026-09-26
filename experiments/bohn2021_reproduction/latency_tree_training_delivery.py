"""Complete pre-validation training delivery; no outcomes are omitted or selected.

Reports already-audited evidence only. Every candidate, repeated candidate, fixed
reference, C5 selection arm and scored episode is retained. Run after training.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from collections import Counter
from latency_tree_protocol import ROOT, OUT, REG, SCRIPTS, TASKS, read, write, sha, verify
from latency_tree_evaluation_spec import (
    EVAL_REG, freeze_policies, verify_hashes, frozen_write, source_for,
)
from latency_tree_policy import policy_key

FILES = ('latency_tree_training_delivery.py', 'latency_tree_training_delivery_review.py')
DEST = OUT / 'training_delivery'
COUNTS = ('steps', 'success', 'constraint', 'solver_failure_steps',
          'initial_failed_steps', 'retries', 'recovered_steps', 'switches')
COSTS = ('total_cost', 'performance_cost', 'h_penalty', 'constraint_cost',
         'physical_constraint_cost')
TIMES = ('decision_total_s', 'gross_total_s', 'logging_total_s')


def aggregate(rows):
    assert rows
    for e in rows:
        assert all(type(e[k]) in (bool, int) and e[k] >= 0 for k in COUNTS)
        assert e['steps'] > 0 and e['success'] in (0, 1) and e['constraint'] in (0, 1)
        assert e['initial_failed_steps'] <= e['steps'] and e['solver_failure_steps'] <= e['steps']
        assert all(math.isfinite(e[k]) for k in COSTS + TIMES)
        assert e['decision_total_s'] > 0 and e['logging_total_s'] > 0
        assert math.isclose(e['gross_total_s'], e['decision_total_s'] + e['logging_total_s'],
                            rel_tol=1e-10, abs_tol=1e-9)
    result = dict(episodes=len(rows), **{k: sum(int(e[k]) for e in rows) for k in COUNTS})
    result.update({k: math.fsum(e[k] for e in rows) / len(rows) for k in COSTS})
    result.update({k: math.fsum(e[k] for e in rows) for k in TIMES})
    result.update(decision_mean_s=result['decision_total_s'] / result['steps'],
                  initial_failure_rate=result['initial_failed_steps'] / result['steps'],
                  final_failure_rate=result['solver_failure_steps'] / result['steps'],
                  mean_horizon=math.fsum(e['mean_horizon'] * e['steps'] for e in rows) / result['steps'])
    return result


def expected_folders():
    entries, candidate_rows, policies = [], [], []
    for task in TASKS:
        for seed in range(3):
            root = OUT / 'train' / ('%s_s%d' % (task, seed))
            fit = read(root / 'fit.json')
            assert len(fit['all_candidates']) == 48
            expected_ids = ['g%d_c%02d' % (g, i) for g in range(4) for i in range(12)]
            assert [r['id'] for r in fit['all_candidates']] == expected_ids
            policy = read(root / 'policy.json')
            assert policy['kind'] in ('constant', 'tree') and policy['task'] == task
            assert fit['learned_tree_selected'] == (policy['kind'] == 'tree')
            assert fit['new_gradient_steps'] == 0 and not fit['validation_access'] and not fit['test_access']
            signatures = [policy_key(r['policy']) for r in fit['all_candidates']]
            multiplicities = Counter(signatures)
            policies.append(dict(task=task, seed=seed, selected=fit['selected'],
                                 learned_tree_selected=fit['learned_tree_selected'], policy=policy,
                                 candidates=48, unique_candidates=len(multiplicities),
                                 repeated_candidate_evaluations=48-len(multiplicities)))
            entries.append((root / 'threshold_reference', task, seed, 'threshold', -1, 'fixed', 0))
            for g in range(4):
                entries.append((root / ('generation%d' % g) / 'fixed', task, seed,
                                'generation_reference', g, 'fixed', 0))
                for i in range(12):
                    item = fit['all_candidates'][12*g+i]
                    folder = root / ('generation%d' % g) / item['id']
                    assert Path(item['folder']) == folder
                    entries.append((folder, task, seed, 'candidate', g, item['id'], 0))
                    summary = read(folder / 'summary.json')
                    assert summary['policy'] == item['policy']
                    rank = item['rank']
                    candidate_rows.append(dict(task=task, seed=seed, generation=g, candidate=item['id'],
                        policy_sha256=policy_key(item['policy']),
                        duplicate_multiplicity=multiplicities[policy_key(item['policy'])],
                        eligible=rank['eligible'], violations=rank['violations'],
                        objective=rank['objective'], cost_change=rank['cost_change'],
                        physical_change=rank['physical_change'], time_ratio=rank['time_ratio'],
                        selected_final=item['id'] == fit['selected'],
                        **aggregate(summary['episodes'])))
            selection = read(root / 'selection_registration.json')
            ids = [r['id'] for r in selection['arms']]
            assert ids[0] == 'fixed' and len(ids) == len(set(ids))
            assert ('certificate' in ids) == (task == 'pendulum')
            assert len(selection['finalists']) <= 4
            assert set(ids) == {'fixed'} | {r['id'] for r in selection['finalists']} | (
                {'certificate'} if task == 'pendulum' else set())
            assert fit['selected'] in ids and fit['selected'] != 'certificate'
            for repeat in range(2):
                for arm in selection['arms']:
                    entries.append((root / 'selection' / ('r%d_%s' % (repeat, arm['id'])),
                                    task, seed, 'selection', -1, arm['id'], repeat))
    assert len(candidate_rows) == 288 and len(policies) == 6
    actual = {p.parent for p in (OUT / 'train').rglob('summary.json')}
    assert actual == {e[0] for e in entries}, ('Missing/unexpected training condition', actual ^ {e[0] for e in entries})
    return entries, candidate_rows, policies


def write_csv(path, rows):
    assert rows
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader()
    writer.writerows(rows)
    value = buffer.getvalue()
    if path.exists():
        with path.open(newline='') as stream:
            assert stream.read() == value, str(path)
    else:
        with path.open('x', newline='') as stream:
            stream.write(value)


def check():
    # Known-answer aggregation checks require no actual training/test data.
    e = dict(steps=2, success=True, constraint=False, solver_failure_steps=0,
             initial_failed_steps=1, retries=1, recovered_steps=1, switches=1,
             total_cost=3., performance_cost=2., h_penalty=1., constraint_cost=0.,
             physical_constraint_cost=2., decision_total_s=.2, gross_total_s=.22,
             logging_total_s=.02, mean_horizon=5.)
    b = dict(e, steps=8, success=False, constraint=True, total_cost=7.,
             performance_cost=6., physical_constraint_cost=6.,
             decision_total_s=.8, gross_total_s=.88, logging_total_s=.08, mean_horizon=25.)
    a = aggregate([e, b])
    assert a['episodes'] == 2 and a['steps'] == 10 and a['success'] == 1
    assert a['total_cost'] == 5. and a['mean_horizon'] == 21.
    assert a['initial_failure_rate'] == .2 and a['decision_mean_s'] == .1
    checks = ['paired_counts', 'arithmetic_mean_cost', 'step_weighted_horizon',
              'step_weighted_latency', 'raw_failure_denominator']
    from latency_tree_training_delivery_review import stats, exact_record
    independent = stats([e, b])
    assert set(independent) == set(a)
    for field in a:
        assert math.isclose(a[field], independent[field], rel_tol=1e-12, abs_tol=1e-12)
    checks.append('independent_aggregate_known_answer')
    record_values = dict(count=12, flag=True, value=.125, missing=None, label='candidate')
    rendered = dict(count='12', flag='True', value='0.125', missing='', label='candidate')
    exact_record(record_values, rendered, 'known_record')
    checks.append('independent_csv_typed_record')
    for field, value in [('count','13'),('flag','False'),('value','nan'),
                         ('missing','0'),('label','different')]:
        try:
            exact_record(record_values, dict(rendered, **{field:value}), 'adversarial_record')
        except AssertionError:
            checks.append('independent_csv_reject_' + field)
        else:
            raise AssertionError(field)
    for field, value in [('steps', 0), ('success', 2), ('initial_failed_steps', 3),
                         ('total_cost', float('nan')), ('decision_total_s', 0.),
                         ('gross_total_s', .3)]:
        try:
            aggregate([dict(e, **{field: value})])
        except AssertionError:
            checks.append('reject_' + field)
        else:
            raise AssertionError(field)
    record = dict(passed=True, checks=checks, count=len(checks), simulations=0,
                  test_accessed=False, hashes={str(SCRIPTS/n): sha(SCRIPTS/n) for n in FILES})
    frozen_write(OUT / 'training_delivery_checks.json', record)
    print(json.dumps(dict(passed=True, checks=len(checks), simulations=0)))


def build():
    freeze_policies()
    checks = read(OUT / 'training_delivery_checks.json')
    assert checks['passed']
    verify_hashes(checks['hashes'])
    status = read(OUT / 'status.json')
    assert status['complete'] and not status['active']
    assert not (OUT / 'evaluations').exists(), 'Freeze full training delivery before validation'
    proofs = [REG, EVAL_REG, OUT/'status.json', OUT/'fitted_policy_registration.json',
              OUT/'audit_train.json', OUT/'audit_smoke.json', OUT/'learning_audit.json',
              OUT/'banks/completed.json', OUT/'split_audit.json', OUT/'training_delivery_checks.json']
    hashes = {str(p): sha(p) for p in proofs}
    for name in ('audit_train.json', 'audit_smoke.json', 'learning_audit.json', 'banks/completed.json'):
        receipt = read(OUT/name)
        assert receipt['passed']
        verify_hashes(receipt['hashes'])
    entries, candidates, policies = expected_folders()
    conditions, episodes, budget_rows, raw_completions = [], [], [], {}
    for folder, task, seed, stage, generation, arm, repeat in entries:
        saved = read(folder/'summary.json')
        assert saved['task'] == task and saved['seed'] == seed
        assert saved['split'] == ('select' if stage == 'selection' else 'fit')
        assert saved['repeats'] == 1
        count = 16 if stage == 'selection' else 12
        assert len(saved['episodes']) == count and [e['case'] for e in saved['episodes']] == list(range(count))
        assert all(e['repeat'] == 0 for e in saved['episodes'])
        metadata = dict(task=task, seed=seed, stage=stage, generation=generation,
                        arm=arm, selection_repeat=repeat, folder=str(folder))
        conditions.append(dict(metadata, **aggregate(saved['episodes'])))
        episodes += [dict(metadata, **e) for e in saved['episodes']]
    roots = [('bank_generation', p.parent) for p in sorted((OUT/'bank_generation').glob('*/solver_attempts.json'))]
    roots += [('smoke', p.parent) for p in sorted((OUT/'smoke').glob('*/summary.json'))]
    roots += [('train', entry[0]) for entry in entries]
    assert len([p for group,p in roots if group == 'bank_generation']) == 2
    assert len([p for group,p in roots if group == 'smoke']) == 12
    assert len(roots) == len(set(p for _,p in roots))
    for group, folder in roots:
        done_path = folder/'completed.json'
        done = read(done_path)
        assert done['passed']
        verify_hashes(done['hashes'])
        hashes[str(done_path)] = sha(done_path)
        raw_completions[str(done_path)] = sha(done_path)
        meters = sorted(folder.glob('attempt_*.json'))
        assert len(meters) == 1, ('Unexpected interrupted meter', str(folder))
        meter = read(meters[0])
        counts = read(folder/'solver_attempts.json')
        assert counts['solve_attempts'] == counts['solve_completed']
        assert meter['reset_calls'] == counts['warmup_attempts']
        if group == 'bank_generation':
            assert meter['step_calls'] == 0 and counts['retry_attempts'] == 0
            scored_steps, resets = 0, done['resets']
        else:
            saved = read(folder/'summary.json')
            scored_steps = sum(e['steps'] for e in saved['episodes'])
            resets = len(saved['episodes'])
            assert scored_steps == saved['steps'] and resets == saved['resets']
            assert counts == saved['solver_counts']
        assert scored_steps == meter['step_calls'] and resets == meter['reset_calls']
        assert counts['solve_completed'] == scored_steps + resets + counts['retry_attempts']
        budget_rows.append(dict(group=group, folder=str(folder), scored_steps=scored_steps,
                                step_attempts=meter['step_calls'], step_completed_inferred=scored_steps,
                                reset_attempts=resets, reset_completed_inferred=resets, **counts))
        for p in meters + [folder/'solver_attempts.json']:
            hashes[str(p)] = sha(p)
    expected_meter_paths = {str(p) for _,folder in roots for p in folder.glob('attempt_*.json')}
    actual_meter_paths = {str(p) for name in ('bank_generation','smoke','train')
                          for p in (OUT/name).rglob('attempt_*.json')}
    assert actual_meter_paths == expected_meter_paths, 'Unaccounted archived/partial attempts'
    expected_solver_paths = {str(folder/'solver_attempts.json') for _,folder in roots}
    assert expected_solver_paths == {str(p) for name in ('bank_generation','smoke','train')
                                     for p in (OUT/name).rglob('solver_attempts.json')}
    fields = ('scored_steps','step_attempts','step_completed_inferred','reset_attempts',
              'reset_completed_inferred','solve_attempts','solve_completed','warmup_attempts','retry_attempts')
    groups = {g: {k: sum(r[k] for r in budget_rows if r['group'] == g) for k in fields}
              for g in ('bank_generation','smoke','train')}
    totals = {k: sum(g[k] for g in groups.values()) for k in fields}
    assert groups['bank_generation']['reset_attempts'] == 556
    assert groups['train']['scored_steps'] == sum(r['steps'] for r in conditions)
    assert groups['train']['reset_attempts'] == len(episodes)
    assert len(episodes) + groups['smoke']['reset_attempts'] <= 4920
    assert groups['train']['scored_steps'] + groups['smoke']['scored_steps'] <= 738000
    inherited = []
    inventory = read(EVAL_REG)['model_inventory']
    seen = set()
    for key, folder in sorted(inventory.items()):
        if folder is None or folder in seen:
            continue
        seen.add(folder)
        p = Path(folder)
        manifest, done = read(p/'manifest.json'), read(p/'completed.json')
        assert done['status'] == 'complete' and done['steps'] == manifest['steps'] == 15000
        inherited.append(dict(key=key, folder=folder, training_steps=done['steps'],
                              previously_completed=True, new_compute_in_this_experiment=False,
                              elapsed_s_recorded=done.get('elapsed_s'),
                              final_diagnostic_episodes_recorded=done.get('test_episodes')))
        for file in ('manifest.json','completed.json','model.zip'):
            hashes[str(p/file)] = sha(p/file)
    offline = {name: read(OUT/name)['independent_integrations'] for name in ('audit_train.json','audit_smoke.json')}
    budget = dict(groups=groups, totals=totals, conditions=budget_rows, inherited_models=inherited,
                  inherited_terminal_training_steps=sum(m['training_steps'] for m in inherited),
                  new_gradient_steps=0, whole_training_wall_s=status['ended']-status['started'],
                  per_condition_wall_s=None, process_cpu_s=None,
                  completed_offline_integrations=offline,
                  constructor_internal_calls=None, constructor_internal_solves=None,
                  scope='Only bank generation, registered training smoke and complete search/selection. '
                        'Full repeated and failed conditions included. Inherited training is separate; '
                        'historical evaluation budgets are not guessed from final episode metadata. '
                        'CPU time and individual full condition wall time were not instrumented.')
    DEST.mkdir(exist_ok=True)
    write_csv(DEST/'all_candidates.csv', candidates)
    write_csv(DEST/'all_conditions.csv', conditions)
    write_csv(DEST/'all_episodes.csv', episodes)
    write_csv(DEST/'all_counters.csv', budget_rows)
    frozen_write(DEST/'selected_policies.json', dict(policies=policies, training_only=True))
    frozen_write(DEST/'budget.json', budget)
    lines = ['# 浅树搜索：全训练交付', '',
             '仅训练内、经过选择的结果；未观察正式验证或测试。不能作为独立效果证据。', '',
             '全部288个候选（含重复）、每代固定参考、阈值参考和两轮复选均保留。C5只作单独诊断，不参与主策略选择。', '',
             '|任务|种子|最终策略|树被选中|独特候选|重复候选执行|',
             '|---|---:|---|---|---:|---:|']
    for p in policies:
        lines.append('|%s|%d|%s|%s|%d|%d|' % (p['task'],p['seed'],p['selected'],
            p['learned_tree_selected'],p['unique_candidates'],p['repeated_candidate_evaluations']))
    lines += ['', '实际新工作预算（包括失败和重复）：',
              '训练控制步：%d；训练回合：%d；冒烟控制步：%d；场景生成reset：556。' % (
                  groups['train']['scored_steps'],len(episodes),groups['smoke']['scored_steps']),
              '合计求解尝试/完成：%d/%d；重试：%d。' % (
                  totals['solve_attempts'],totals['solve_completed'],totals['retry_attempts']),
              '继承终端模型%d个、历史训练步数%d；不计作本轮新训练，不声称等预算优越。' % (
                  len(inherited),budget['inherited_terminal_training_steps']),
              '构造器内部调用、每条件完整墙时与CPU时间未仪表化，未知不记零。原始状态、动作、成本、H和时间在条件目录。', '',
              '离线积分单列；场景和训练种子是不同层级，复选两次计时不是新增独立场景。', '',
              '所有真实训练结果须经独立交付复核；完整验证、计时和封存确认尚未完成。', '']
    text = '\n'.join(lines)
    report = DEST/'report_CN.md'
    if report.exists():
        assert report.read_text() == text
    else:
        report.write_text(text)
    outputs = [p for p in DEST.iterdir() if p.name in (
        'all_candidates.csv','all_conditions.csv','all_episodes.csv','all_counters.csv',
        'selected_policies.json','budget.json','report_CN.md')]
    manifest = dict(passed=True, training_only=True, validation_accessed=False, test_accessed=False,
                    goal_complete=False, candidate_rows=288, condition_rows=len(conditions),
                    episode_rows=len(episodes), source_hashes={str(SCRIPTS/n):sha(SCRIPTS/n) for n in FILES},
                    hashes=dict(hashes, **{str(p):sha(p) for p in outputs}),
                    raw_completion_hashes=raw_completions,
                    scope='Immutable full training delivery; independent delivery review required before validation.')
    frozen_write(DEST/'manifest.json', manifest)
    print(json.dumps(dict(passed=True,candidates=288,conditions=len(conditions),episodes=len(episodes),
                          budget=totals,independent_review_pending=True),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    check() if args.check else build()
