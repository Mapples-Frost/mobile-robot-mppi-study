"""Reconcile an existing primary snapshot and separately instrumented diagnostics.

Read-only toward experiments and frozen accounting scripts. Outputs immutable
ledger snapshots; a live primary snapshot is rejected if its source hashes moved.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
DIAGNOSTIC = RUN / 'training_feasible_witness'
LINUX_ROOT = '/home/mapples/projects/mobile-robot-mppi-study/'
FIELDS = ('environment_meters', 'explicit_step_attempts', 'explicit_reset_attempts',
          'raw_solve_attempts', 'raw_solve_completed', 'warmup_attempts', 'retry_attempts',
          'training_instrumentation_steps', 'training_instrumentation_resets')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def resolve(name):
    path = ROOT / name[len(LINUX_ROOT):] if name.startswith(LINUX_ROOT) else Path(name)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def canonical(path):
    return path.resolve().relative_to(ROOT).as_posix()


def checked_hashes(mapping):
    for name, expected in mapping.items():
        assert sha(resolve(name)) == expected, 'Snapshot source changed: ' + name


def reconcile_primary(base):
    assert sha(ROOT / 'experiments/bohn2021_reproduction/gated_horizon_budget.py') == base['source_hash']
    checked_hashes(base['hashes'])
    reconstructed = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    seen = set()
    unresolved = {}
    for name in base['hashes']:
        path = resolve(name)
        assert path not in seen, 'Duplicate counter path after canonicalization'
        seen.add(path)
        rel = path.relative_to(RUN)
        assert rel.parts[0] != 'training_feasible_witness', 'Diagnostic already in primary; refusing double count'
        data = read(path)
        group = reconstructed[rel.parts[0]]
        if path.name.startswith('attempt_'):
            assert 'step_calls' in data
            group['environment_meters'] += 1
            group['explicit_step_attempts'] += data['step_calls']
            group['explicit_reset_attempts'] += data['reset_calls']
        elif path.name == 'solver_attempts.json':
            for source, target in [('solve_attempts', 'raw_solve_attempts'), ('solve_completed', 'raw_solve_completed'),
                                   ('warmup_attempts', 'warmup_attempts'), ('retry_attempts', 'retry_attempts')]:
                group[target] += data[source]
            if data['solve_attempts'] != data['solve_completed']:
                unresolved[canonical(path)] = data
        elif path.name.startswith('instrumentation_attempt_'):
            group['training_instrumentation_steps'] += data['step_attempts']
            group['training_instrumentation_resets'] += data['reset_attempts']
        else:
            raise AssertionError('Unexpected primary counter type: ' + name)
    assert dict(reconstructed) == base['groups']
    totals = {field: sum(group[field] for group in reconstructed.values()) for field in FIELDS}
    assert totals == base['totals']
    assert unresolved == {canonical(resolve(row['path'])): row['counts'] for row in base['unresolved_or_current_solve_attempts']}
    checked_hashes(base['hashes'])
    return totals, len(seen)


def reconcile_diagnostic():
    check = read(DIAGNOSTIC / 'coverage_check.json')
    assert check['passed'] and check['conditions'] == 810 and check['steps_checked'] == 12331
    checked_hashes(check['hashes'])
    status, smoke, audit, audit_status, budget = (read(DIAGNOSTIC / name) for name in
        ('status.json', 'smoke.json', 'audit.json', 'audit_status.json', 'budget.json'))
    assert status['complete'] and not status['active'] and audit_status['complete'] and not audit_status['active']
    assert smoke['passed'] and audit['passed']
    results = read(DIAGNOSTIC / 'results.json')['results']
    assert len(results) == 810
    steps = 0
    trace_keys = set()
    journal_hashes = {}
    for result in results:
        key = (result['seed'], result['case'], result['controller'])
        assert key not in trace_keys
        trace_keys.add(key)
        trace_path = DIAGNOSTIC / result['path']
        assert sha(trace_path) == result['sha256']
        trace = read(trace_path)['rows']
        journal_path = trace_path.with_suffix('.jsonl')
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        assert trace == journal and len(trace) == result['steps']
        steps += len(trace)
        journal_hashes[canonical(journal_path)] = sha(journal_path)
    assert steps == status['step_completed'] == status['step_attempts'] == budget['diagnostic_step_attempts']
    assert steps == budget['diagnostic_completed_steps'] == audit_status['integrations_attempted'] == audit_status['integrations_completed']
    assert budget['total_new_simulated_steps'] == steps + smoke['environment_steps']
    assert budget['rk4_rhs_evaluations'] == status['rk4_rhs_evaluations'] + smoke['rk4_rhs_evaluations']
    assert budget['independent_formal_integrations'] == steps
    assert budget['independent_smoke_integrations'] == smoke['independent_integration_calls']
    assert budget['independent_formal_rhs_evaluations'] == audit_status['rhs_evaluations']
    assert budget['independent_smoke_rhs_evaluations'] is None
    assert all(budget[k] == 0 for k in ('training_steps', 'reset_warmups', 'nlp_calls', 'policy_or_terminal_updates', 'validation_test_episodes'))
    return budget, journal_hashes


def main(base_path, label, require_primary_complete):
    assert re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', label), 'Use a simple snapshot label'
    base_path = resolve(str(base_path))
    assert base_path.is_file() and base_path.parent == RUN / 'budget'
    base_hash = sha(base_path)
    base = read(base_path)
    primary_totals, source_count = reconcile_primary(base)
    diagnostic, journal_hashes = reconcile_diagnostic()
    if require_primary_complete:
        assert base_path == RUN / 'budget/snapshot.json', 'Final mode uses the completed primary pipeline snapshot'
        finish = read(RUN / 'validation_finish_status.json')
        assert finish.get('complete') and not finish['active'], 'Primary validation/timing pipeline is not complete'
        checked_hashes(finish['output_hashes'])
        expected = [value for path,value in finish['output_hashes'].items() if resolve(path) == base_path]
        assert expected == [base_hash], 'Budget does not match completed pipeline'
    assert sha(base_path) == base_hash
    source_paths = [base_path] + [DIAGNOSTIC / name for name in ('budget.json','coverage_check.json','registration.json','results.json','status.json','audit.json','audit_status.json','smoke.json')]
    if require_primary_complete:
        source_paths.append(RUN / 'validation_finish_status.json')
    sources = {canonical(p):sha(p) for p in source_paths}
    dest = RUN / 'budget_supplement' / label
    if (dest / 'ledger.json').exists():
        previous = read(dest / 'ledger.json')
        assert previous['hashes'] == sources and previous['source_hash'] == sha(Path(__file__))
        assert previous['journal_hashes'] == journal_hashes
        for name, value in previous['delivery_hashes'].items():
            assert sha(dest / name) == value
        print(json.dumps(dict(reused=True, ledger=canonical(dest / 'ledger.json'), recorded_explicit_steps=previous['combined_recorded_explicit_step_attempts']), indent=2))
        return
    assert not dest.exists(), 'Inspect partial ledger before writing'
    dest.mkdir(parents=True)
    combined = primary_totals['explicit_step_attempts'] + diagnostic['total_new_simulated_steps']
    rows = [dict(source='primary:' + name, explicit_step_attempts=group['explicit_step_attempts'],
                 explicit_reset_attempts=group['explicit_reset_attempts'], raw_nlp_attempts=group['raw_solve_attempts'],
                 raw_nlp_completed=group['raw_solve_completed'], fixed_training_instrumented_steps=group['training_instrumentation_steps'])
            for name, group in base['groups'].items()]
    rows.append(dict(source='diagnostic:training_feasible_witness', explicit_step_attempts=diagnostic['total_new_simulated_steps'],
                     explicit_reset_attempts=0, raw_nlp_attempts=0, raw_nlp_completed=0, fixed_training_instrumented_steps=0))
    with (dest / 'all_sources.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scope = ('已完成主验证/串行计时流水线的预算，加已审计LQR训练诊断。' if require_primary_complete else
             '指定的历史主预算快照，加已审计LQR训练诊断；不是当前完整研究开销，也未纳入该快照之后的验证/计时或补训。')
    text = ['# 主实验与额外诊断的预算合并', '', scope, '',
        '合并前重算原快照全部%d个计数文件的分组与总量，并检查来源散列；另逐条核对810条LQR诊断轨迹及其逐步JSONL。原始快照和已冻结计数脚本没有改写。'%source_count, '',
        '|预算项目|数量|范围|', '|---|---:|---|',
        '|原快照显式环境步尝试|%d|保留原来的失败和中断尝试|'%primary_totals['explicit_step_attempts'],
        '|新增LQR诊断环境步|%d|12,331正式步及20合成冒烟步|'%diagnostic['total_new_simulated_steps'],
        '|上述已记录显式步合计|%d|不等同全部历史训练、reset或离线积分总量|'%combined,
        '|原快照reset尝试|%d|另列；不能再次作为计分步相加|'%primary_totals['explicit_reset_attempts'],
        '|原始NLP尝试/完成|%d / %d|诊断没有NLP调用|'%(primary_totals['raw_solve_attempts'],primary_totals['raw_solve_completed']),
        '|独立固定训练的仪表化步|%d|主快照可能同时包含训练和末次评估，单列避免混算|'%primary_totals['training_instrumentation_steps'], '',
        '原快照报告的继承终端训练为300,000步网格训练加60,000步补种子训练。本脚本保留该来源数字，没有在此重新核验全部继承模型，也不把它算作本次新增模拟。', '',
        'LQR诊断额外有988,080次RK4 rhs求值，12,331次正式独立积分（432,083次rhs求值）及20次冒烟独立积分。冒烟独立积分的rhs调用数未保存，保留为未知而非零。其他原实验的离线积分开销仍按原审计报告单列；不把这些不同操作单位合成一个计算预算。', '',
        '原快照中的构造内部调用、未安装求解包装器的场景生成reset、独立固定训练的原始NLP次数等覆盖限制，以及中断时两次NLP完成状态未知，均继续保留。没有用合并后的数字声称双方训练/搜索预算相同。', '',
        '## 完整验证结束后的汇总入口', '',
        '验证计时接续完成后，用最终主快照重新生成一个独立账本；此选项会要求接续complete且所有输出散列一致：', '',
        '    .venv/bin/python experiments/bohn2021_reproduction/gated_horizon_budget_supplement.py --base-snapshot research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25/budget/snapshot.json --label completed_validation --require-primary-complete', '',
        '该命令不等待、不重启主实验，也不读取效果分数或封存测试。若以后增加诊断，须显式纳入新预算，不能把本账本误称所有将来工作的完整清单。', '']
    (dest/'report_CN.md').write_text('\n'.join(text),encoding='utf-8')
    value=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),scope=scope,
        primary_complete_required=require_primary_complete,primary_snapshot=canonical(base_path),primary_snapshot_time=base['snapshot_time'],
        primary_counter_files=source_count,primary_totals=primary_totals,primary_groups=base['groups'],
        diagnostic_budget=diagnostic,combined_recorded_explicit_step_attempts=combined,
        inherited_terminal_training_as_reported_by_base=base['inherited_terminal_training'],
        unresolved_original_attempts=base['unresolved_or_current_solve_attempts'],original_caveats=base['caveats'],
        duplicate_count_check='Primary counter paths canonicalized unique; none inside training_feasible_witness; diagnostic journals separately reconciled.',
        diagnostic_trace_conditions=810,journal_hashes=journal_hashes,hashes=sources,source_hash=sha(Path(__file__)),
        delivery_hashes={n:sha(dest/n) for n in ('all_sources.csv','report_CN.md')})
    (dest/'ledger.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,primary_counter_files=source_count,diagnostic_conditions=810,recorded_explicit_steps=combined,ledger=canonical(dest/'ledger.json'),primary_complete_required=require_primary_complete),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-snapshot',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--require-primary-complete',action='store_true')
    args=parser.parse_args()
    main(args.base_snapshot,args.label,args.require_primary_complete)
