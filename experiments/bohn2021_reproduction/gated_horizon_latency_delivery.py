"""Export complete validation latency distributions after the serial pipeline.

Descriptive delivery only: no simulation, model selection, new effect criterion,
or test access. Pooled percentiles use individual steps, never episode medians.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
DEST = RUN / 'validation_latency_delivery'
REG = RUN / 'latency_delivery_registration.json'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
np = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name):
    path = ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    path.relative_to(ROOT)
    assert '/evaluations/test/' not in path.as_posix() and '/timing_test/' not in path.as_posix()
    return path


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def guard():
    proc = Path('/proc') if Path('/proc').exists() else Path('//wsl.localhost/Ubuntu-20.04/proc')
    for name in ('validation_finish_status.json', 'validation_status.json',
                 'posttrain_status.json', 'baseline_completion_status.json'):
        state = read(RUN / name)
        if state.get('active') or not state.get('complete'):
            raise RuntimeError('PENDING: ' + name + ' is not complete')
        if (proc / str(state['pid']) / 'cmdline').exists():
            raise RuntimeError('PENDING: controller PID still exists: ' + str(state['pid']))


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def distribution(values, deadline):
    a = np.asarray(values, dtype=float)
    assert a.ndim == 1 and a.size and np.isfinite(a).all() and (a > 0).all()
    assert math.isfinite(deadline) and deadline > 0
    return dict(steps=int(a.size), total_s=float(a.sum()), mean_s=float(a.mean()),
                median_s=float(np.median(a)), p95_s=float(np.percentile(a, 95)),
                maximum_s=float(a.max()), deadline_exceed_steps=int((a > deadline).sum()),
                deadline_exceed_fraction=float((a > deadline).mean()))


def summarize(columns, deadline):
    assert set(columns) == {'decision', 'gross', 'logging', 'selection'}
    counts = {len(a) for a in columns.values()}
    assert len(counts) == 1 and next(iter(counts)) > 0
    for net, gross, logged in zip(columns['decision'], columns['gross'], columns['logging']):
        close(net + logged, gross)
    result = dict(deadline_s=deadline)
    for key in ('decision', 'gross'):
        result.update({key + '_' + k: v for k, v in distribution(columns[key], deadline).items()})
    for key in ('logging', 'selection'):
        values = np.asarray(columns[key], dtype=float)
        assert np.isfinite(values).all() and (values > 0).all()
        result[key + '_total_s'] = float(values.sum())
        result[key + '_mean_s'] = float(values.mean())
    result['logging_fraction_of_gross'] = result['logging_total_s'] / result['gross_total_s']
    return result


def self_check():
    # Unequal episode lengths expose the erroneous mean-of-medians shortcut.
    values = [1.] + [10.] * 9
    stats = distribution(values, 10.)
    assert stats['steps'] == 10 and stats['median_s'] == 10. and stats['mean_s'] == 9.1
    assert stats['p95_s'] == 10. and stats['deadline_exceed_steps'] == 0
    assert np.mean([np.median([1.]), np.median([10.] * 9)]) != stats['median_s']
    close(distribution([1., 2., 3., 4.], 3.)['p95_s'], 3.85)
    assert distribution([1., 2., 3., 4.], 3.)['deadline_exceed_steps'] == 1
    checked = ['step_weighted_mean_and_pooled_quantiles', 'linear_p95', 'strict_deadline_boundary']
    for label, invalid in [('empty', []), ('nan', [float('nan')]), ('zero', [0.]), ('negative', [-1.])]:
        try:
            distribution(invalid, .1)
        except AssertionError:
            checked.append('reject_' + label)
        else:
            raise AssertionError('Accepted ' + label)
    columns = dict(decision=[1., 2.], gross=[1.1, 2.1], logging=[.1, .1], selection=[.01, .01])
    close(summarize(columns, 1.)['logging_fraction_of_gross'], .2 / 3.2)
    for invalid in (dict(columns, gross=[1.2, 2.1]), dict(columns, selection=[.01])):
        try:
            summarize(invalid, 1.)
        except AssertionError:
            pass
        else:
            raise AssertionError('Accepted incompatible timing columns')
    checked += ['gross_net_decomposition', 'reject_inconsistent_lengths_or_decomposition']
    return dict(passed=True, checks=checked, real_outcomes_read=False)


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def export():
    guard()
    hashes = {}

    def record(path, expected=None):
        key, value = relative(path), sha(path)
        assert expected is None or value == expected, 'Hash mismatch: ' + key
        assert key not in hashes or hashes[key] == value, 'Input changed: ' + key
        hashes[key] = value

    def graph(mapping):
        for name, value in mapping.items():
            record(resolve(name), value)

    graph(read(REG)['source_hashes'])
    record(REG)
    finish = read(RUN / 'validation_finish_status.json')
    graph(finish['output_hashes'])
    record(RUN / 'validation_finish_status.json')
    audit = read(RUN / 'audit_timing_validation.json')
    assert audit['passed'] and audit['conditions'] == 36 and audit['episodes'] == 1152
    graph(audit['hashes'])
    effect = read(RUN / 'validation_delivery/effect_gate.json')
    assert effect['timing_audited'] and effect['extra_baselines_audited']
    assert effect['independent_test_unlocked'] is False
    roles = {}
    for comparison in effect['comparisons']:
        for label, arm in [('adaptive', comparison['adaptive']), (comparison['comparator'], comparison['fixed'])]:
            key = (arm['task'], arm['family'], arm['seed'], arm['h'])
            roles.setdefault(key, set()).add(label)
    assert len(roles) == 18 and len(effect['comparisons']) == 12
    expected = {(repeat,) + key for repeat in (0, 1) for key in roles}
    seen, episodes, conditions, pool, pooled_resets = set(), [], [], {}, {}
    for group in sorted(audit['groups'], key=lambda g: g['folder']):
        folder = resolve(group['folder'])
        assert folder.parent == RUN / 'timing_validation'
        done = read(folder / 'completed.json')
        assert done['passed'] and done['exact_replay']
        record(folder / 'completed.json', audit['hashes'][str(folder / 'completed.json')])
        record(folder / 'summary.json', done['hashes'][str(folder / 'summary.json')])
        summary = read(folder / 'summary.json')
        assert summary['split'] == 'validation' and summary['smoke'] is False
        key = tuple(summary[k] for k in ('task', 'family', 'seed', 'h'))
        identity = (summary['repeat'],) + key
        assert identity in expected and identity not in seen
        seen.add(identity)
        assert folder.name == 'r%d_%s_%s_h%d_s%d' % (summary['repeat'], key[0], key[1], key[3], key[2])
        metadata = dict(task=key[0], family=key[1], seed=key[2], h=key[3],
                        roles='+'.join(sorted(roles[key])), repeat=summary['repeat'])
        deadline = .1 if key[0] == 'vehicle' else .04
        all_columns = {k: [] for k in ('decision', 'gross', 'logging', 'selection')}
        reset_total = 0.
        assert len(summary['episodes']) == 32 and [e['case'] for e in summary['episodes']] == list(range(32))
        for episode in summary['episodes']:
            path = folder / ('r0_trace_%02d.json' % episode['case'])
            record(path, done['hashes'][str(path)])
            trace = read(path)
            assert len(trace) == episode['steps']
            columns = {k: [r['timing'][f] for r in trace] for k, f in
                       [('decision', 'decision_s'), ('gross', 'decision_gross_s'),
                        ('logging', 'logging_s'), ('selection', 'selection_s')]}
            stats = summarize(columns, deadline)
            for field in ('total_s', 'mean_s', 'median_s', 'p95_s'):
                close(stats['decision_' + field], episode['decision_' + field])
            assert stats['decision_deadline_exceed_steps'] == episode['deadline_exceed_steps']
            close(stats['gross_total_s'], episode['decision_gross_total_s'])
            close(stats['logging_total_s'], episode['logging_total_s'])
            reset = episode['reset']
            assert math.isfinite(reset['reset_gross_s']) and reset['reset_gross_s'] > 0
            reset_total += reset['reset_gross_s']
            episodes.append(dict(metadata, case=episode['case'], reset_gross_s=reset['reset_gross_s'], **stats))
            for field in columns:
                all_columns[field].extend(columns[field])
        stats = summarize(all_columns, deadline)
        assert stats['decision_steps'] == group['steps'] and group['episodes'] == 32
        assert summary['solver_counts'] == group['solver_counts']
        counts = summary['solver_counts']
        conditions.append(dict(metadata, episodes=32, reset_gross_total_s=reset_total,
                               solve_attempts=counts['solve_attempts'], retry_attempts=counts['retry_attempts'],
                               warmup_attempts=counts['warmup_attempts'], **stats))
        bucket = pool.setdefault(key, {k: [] for k in all_columns})
        for field in bucket:
            bucket[field].extend(all_columns[field])
        pooled_resets[key] = pooled_resets.get(key, 0.) + reset_total
    assert seen == expected and len(conditions) == 36 and len(episodes) == 1152
    assert sum(c['decision_steps'] for c in conditions) == audit['steps']
    pooled = []
    for key, columns in sorted(pool.items()):
        stats = summarize(columns, .1 if key[0] == 'vehicle' else .04)
        pooled.append(dict(task=key[0], family=key[1], seed=key[2], h=key[3],
                           roles='+'.join(sorted(roles[key])), timing_repeats=2, unique_scenes=32,
                           measured_episodes=64, reset_gross_total_s=pooled_resets[key], **stats))
    lines = ['# 完整验证的实际决策延迟', '',
             '描述性汇总：36个计时条件、18个唯一比较臂、两遍串行重复，共1,152回合。所有失败、提前终止和重试均保留。', '',
             '主指标为门控选择加控制器决策时间，扣除单独测量的恢复日志开销；毛耗时包含该日志开销。物理仿真与外层轨迹写入不在其中，reset另列。这不是完整控制周期，也不是纯IPOPT时间。', '',
             '以下每行合并同一臂两遍记录的所有控制步重算中位数与p95；不平均逐回合分位数。完整逐重复表见all_conditions.csv，逐回合表见all_episodes.csv。重复不是新增独立场景，短失败回合仍参与，不按结果筛选。', '',
             '|任务|比较臂|种子|H|控制步|均值ms|中位ms|p95 ms|毛均值ms|净/毛超时步|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in pooled:
        lines.append('|%s|%s|%d|%d|%d|%.4f|%.4f|%.4f|%.4f|%d / %d|' % (
            row['task'], row['roles'], row['seed'], row['h'], row['decision_steps'],
            1000 * row['decision_mean_s'], 1000 * row['decision_median_s'], 1000 * row['decision_p95_s'],
            1000 * row['gross_mean_s'], row['decision_deadline_exceed_steps'], row['gross_deadline_exceed_steps']))
    lines += ['', '车辆阈值100ms，倒立摆40ms，严格大于阈值计为超时。所有CSV均保留单位为秒的数值、总时间、最大值、日志及策略选择时间、reset时间。', '',
              'WSL宿主调度、串行条件顺序和扣除日志的计时扰动仍是限制。均值按控制步加权；不同方法提前结束会改变其状态与步数构成。该表不作新的显著性或速度验收，正式效果沿用已冻结的配对场景bootstrap与全部种子规则。', '',
              '登记验证门槛：%s；本导出不解封测试。' % effect['validation_effect_passed'], '']
    payloads = {'all_conditions.csv': csv_text(conditions), 'all_episodes.csv': csv_text(episodes),
                'pooled_arms.csv': csv_text(pooled), 'report_CN.md': '\n'.join(lines)}
    delivery_hashes = {name: hashlib.sha256(body.encode('utf-8')).hexdigest() for name, body in payloads.items()}
    graph(dict(hashes))
    manifest = dict(passed=True, conditions=36, episodes=1152, unique_arms=18, measured_steps=audit['steps'],
                    validation_effect_passed=effect['validation_effect_passed'], new_simulation_steps=0,
                    test_accessed=False, changed_acceptance_criteria=False, hashes=hashes,
                    self_checks=self_check(), source_hash=sha(Path(__file__)), delivery_hashes=delivery_hashes,
                    pooling='Descriptive pooled individual control steps across both repeats. 32 unique scenes per arm; repeated timing does not increase independent sample size.')
    if DEST.exists():
        assert (DEST / 'manifest.json').exists(), 'Inspect partial delivery before retrying'
        assert read(DEST / 'manifest.json') == manifest, 'Preserve differing previous delivery'
        for name, value in delivery_hashes.items():
            assert sha(DEST / name) == value
    else:
        DEST.mkdir()
        for name, body in payloads.items():
            (DEST / name).write_bytes(body.encode('utf-8'))
        (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('passed', 'conditions', 'episodes', 'unique_arms',
                                             'measured_steps', 'test_accessed', 'new_simulation_steps')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'export'), required=True)
    args = parser.parse_args()
    guard()  # Also guard synthetic checks: no extra Python work during timing.
    import numpy as np
    if args.mode == 'check':
        print(json.dumps(self_check(), indent=2))
    else:
        export()
