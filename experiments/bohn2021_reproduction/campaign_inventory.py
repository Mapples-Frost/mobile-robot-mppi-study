"""Inventory existing evidence without rerunning training or opening test outcomes.

Counts are explicitly lower bounds. A completion marker is not an audit verdict.
"""
import collections
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/campaign_inventory_2026-09-24'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    runs, failures, groups, hashes = [], [], {}, {}
    for p in sorted((ART / 'results').rglob('manifest.json')):
        spec = read(p)
        if not isinstance(spec, dict) or not {'task', 'seed', 'steps'} <= set(spec): continue
        dest = p.parent
        done_path, progress_path = dest / 'completed.json', dest / 'progress.json'
        done = read(done_path) if done_path.exists() else {}
        progress = read(progress_path) if progress_path.exists() else {}
        zip_path = dest / 'model.zip'
        complete = done.get('status') == 'complete' and zip_path.exists()
        row = {'path': str(dest.relative_to(ROOT)), 'task': spec['task'], 'seed': spec['seed'],
               'fixed_horizon': spec.get('fixed_horizon'), 'declared_steps': spec['steps'],
               'completed_marker_and_model': complete, 'archive': 'archive' in str(dest),
               'observed_steps': done.get('steps', 0) if complete else progress.get('steps', 0),
               'updates': done.get('updates') if complete else progress.get('updates'),
               'completion_sha256': sha(done_path) if done_path.exists() else None,
               'model_sha256': sha(zip_path) if zip_path.exists() else None,
               'parameters_hash': done.get('final_hash'), 'independent_audit_not_implied': True}
        hashes[str(p)] = sha(p)
        if done_path.exists(): hashes[str(done_path)] = sha(done_path)
        runs.append(row)
        name = dest.relative_to(ART / 'results').parts[0]
        g = groups.setdefault(name, {'manifests': 0, 'completed_markers_with_models': 0,
              'completed_declared_steps': 0, 'incomplete_observed_steps_lower_bound': 0})
        g['manifests'] += 1
        g['completed_markers_with_models'] += int(complete)
        g['completed_declared_steps' if complete else 'incomplete_observed_steps_lower_bound'] += row['observed_steps']
    for pattern in ('*failed*.json', '*failure*.json', '*error*.json'):
        for p in sorted((ART / 'results').rglob(pattern)):
            failures.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size})
    duplicates = collections.defaultdict(list)
    for r in runs:
        if r['parameters_hash']: duplicates[r['parameters_hash']].append(r['path'])
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    # Seal checking uses scenario identity only, never controller test outcomes.
    current = ART / 'results/branch_calibration_2026-09-24'
    scene_hashes = {}
    for p in sorted(current.glob('*_bank.json')):
        data = read(p)
        scene_hashes[p.name] = [hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest() for c in data['cases']]
    overlap = []
    names = sorted(scene_hashes)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            common = set(scene_hashes[a]) & set(scene_hashes[b])
            if common: overlap.append({'a': a, 'b': b, 'count': len(common)})
    assert not overlap, overlap
    result = {'created_unix': time.time(), 'scope': 'Historical author-format training manifests only; not total project compute.',
        'limitations': ['Counters from incomplete attempts are lower bounds; reset and unflushed work excluded.',
            'Copied or warm-started models may share parameters. Do not blindly sum these as independent experiments.',
            'PyTorch teacher/discrete/other extensions may not have this manifest schema and are outside this subtotal.',
            'File completion alone does not prove numerical or protocol audit success. Current running jobs may be absent.',
            'Failure-file list does not capture every failed process; logs and archived attempts remain authoritative.'],
        'groups': groups, 'runs': runs, 'duplicate_final_parameter_hashes': duplicates,
        'failure_artifacts': failures, 'input_hashes': hashes,
        'current_bank_overlap': {'identical_scenes_across_banks': overlap, 'scenario_hashes': scene_hashes,
            'test_outcomes_read': False, 'test_evaluations_exist': (current / 'evaluations/test').exists()}}
    (OUT / 'inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Bøhn 2021 历史训练文件清单', '',
        '只清点已有文件，不重新训练、不读取封存测试表现。完成标记不等于独立审计通过。', '',
        '|分组|manifest 数|完成标记且有模型|完成步数小计|未完成已记录步数下界|', '|---|---:|---:|---:|---:|']
    for name, g in sorted(groups.items()):
        lines.append('|%s|%d|%d|%d|%d|' % (name, g['manifests'], g['completed_markers_with_models'], g['completed_declared_steps'], g['incomplete_observed_steps_lower_bound']))
    lines += ['', '上述小计不覆盖所有 PyTorch 扩展、分支仿真、重置、未刷盘与失败工作；不能当作全项目总预算。',
        '完整逐运行路径、模型散列、重复参数提示和失败文件路径见 inventory.json。',
        '本次校准的十个场景库之间未发现完全相同场景；只比较场景散列，没有读取测试表现。']
    (OUT / 'inventory.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'training_manifests': len(runs), 'groups': len(groups), 'duplicate_parameter_groups': len(duplicates), 'bank_overlap': len(overlap)}))


if __name__ == '__main__': main()
