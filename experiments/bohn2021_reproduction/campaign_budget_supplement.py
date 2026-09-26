"""Reconcile training formats omitted by the author-manifest inventory.

Reads historical training records and existing budget fields only. No simulation,
model deserialization, evaluation outcomes or sealed scenario content is used.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results'
OUT = RESULTS/'campaign_inventory_2026-09-24/budget_supplement'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    hashes, runs, interruptions = {}, [], []

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    def model(path):
        assert path.is_file(), path
        hashes[str(path)] = sha(path)
        return str(path.relative_to(ROOT))

    def count_rows(path):
        # Validate record framing while counting; do not report controller outcomes.
        h, count = hashlib.sha256(), 0
        with path.open('rb') as f:
            for line in f:
                h.update(line)
                assert isinstance(json.loads(line), dict), path
                count += 1
        hashes[str(path)] = h.hexdigest()
        return count

    old = read(OUT.parent/'inventory.json')
    old_paths = {r['path'] for r in old['runs']}
    for group, arms in [('categorical_frozen', ('continuous','discrete')),
                        ('sac_teacher', ('plain','teacher')),
                        ('sac_preserve', ('free','rule','value'))]:
        for arm in arms:
            for seed in range(3):
                name = '%s_s%d' % (arm,seed)
                folder = RESULTS/group/(name if group=='categorical_frozen' else 'models/'+name)
                assert str(folder.relative_to(ROOT)) not in old_paths
                done = read(folder/'completed.json')
                spec = read(folder/'manifest.json')
                assert spec['seed']==seed
                if group=='categorical_frozen':
                    retained = sum(e['steps'] for e in read(folder/'training_episodes.json'))
                    progress = read(folder/'progress.json')
                    observed = done['steps']
                    assert observed==progress['steps']==15000 and 0<=observed-retained<100
                    evidence = 'Completion/progress count; completed-episode lengths exclude the final unfinished episode'
                    checkpoint = folder/'model.pt'
                else:
                    observed = count_rows(folder/'transitions.jsonl')
                    retained = observed
                    expected = done['online_steps']+done['teacher_steps'] if group=='sac_teacher' else done['steps']
                    assert observed==expected
                    evidence = 'Recounted parseable training transitions.jsonl records'
                    end = done['online_steps'] if group=='sac_teacher' else done['steps']
                    checkpoint = folder/('step_%05d.%s' % (end,'zip' if group=='sac_teacher' else 'pt'))
                runs.append({'group':group,'arm':arm,'seed':seed,'environment_training_steps':observed,
                    'retained_record_steps':retained,'steps_supported_only_by_metadata':observed-retained,
                    'updates':done['updates'],'kind':'online_or_teacher_collection',
                    'path':str(folder.relative_to(ROOT)),'checkpoint':model(checkpoint),'evidence':evidence})

    for rnd in range(2):
        for seed in range(3):
            stem = RESULTS/'teacher_value/models'/('round%d_s%d' % (rnd,seed))
            metrics = read(stem.with_name(stem.name+'_metrics.json'))
            assert metrics['updates']==3000
            runs.append({'group':'teacher_value','arm':'round%d'%rnd,'seed':seed,
                'environment_training_steps':0,'retained_record_steps':0,'steps_supported_only_by_metadata':0,
                'updates':metrics['updates'],'kind':'supervised_reuse_of_branch_labels',
                'path':str(stem.relative_to(ROOT)),'checkpoint':model(stem.with_suffix('.pt')),
                'evidence':'Saved update metadata and model; branch simulation budget listed separately'})

    categorical = read(RESULTS/'categorical_frozen/audit.json')
    cat_protocol = read(RESULTS/'categorical_frozen/protocol.json')
    cat_completed = read(RESULTS/'categorical_frozen/completed.json')
    cat_steps = sum(r['environment_training_steps'] for r in runs if r['group']=='categorical_frozen')
    assert cat_steps==cat_protocol['new_training_transitions']==cat_completed['new_training_transitions']==90000
    stationary = [r for r in old['runs'] if r['path'].split('/')[3]=='stationary_terminal' and r['observed_steps']==15000]
    assert len(stationary)==3 and sum(r['observed_steps'] for r in stationary)==45000
    assert categorical['new_formal_training_steps']==cat_steps+45000
    amendment = read(RESULTS/'categorical_frozen/resource_amendment.json')
    for item in amendment['interrupted_attempts']:
        folder = RESULTS/'categorical_frozen/interrupted_memory'/item['name']
        progress = read(folder/'progress.json')
        assert progress['steps']==item['steps']
        interruptions.append({'group':'categorical_frozen','path':str(folder.relative_to(ROOT)),
            'saved_steps_lower_bound':progress['steps'],'documented_upper_bound':item['transitions_upper_bound'],
            'source':'resource_amendment and archived progress; excludes unknown initialization/reset work'})

    teacher_audit = read(RESULTS/'sac_teacher/audit.json')
    for folder in sorted((RESULTS/'sac_teacher/interrupted_shutdown').iterdir()):
        if not folder.is_dir(): continue
        count = count_rows(folder/'transitions.jsonl')
        progress = read(folder/'progress.json')
        interruptions.append({'group':'sac_teacher','path':str(folder.relative_to(ROOT)),
            'saved_steps_lower_bound':count,'documented_upper_bound':None,
            'source':'Recounted archived transition records; unflushed shutdown work unknown'})
    assert sum(r['saved_steps_lower_bound'] for r in interruptions if r['group']=='sac_teacher')==teacher_audit['budget']['interrupted_saved_transitions']
    assert sum(r['environment_training_steps'] for r in runs if r['group']=='sac_teacher')==teacher_audit['budget']['formal_training_transitions']
    preserve = read(RESULTS/'sac_preserve/audit.json')
    assert sum(r['environment_training_steps'] for r in runs if r['group']=='sac_preserve')==preserve['training_steps']
    assert sum(r['updates'] for r in runs if r['group']=='sac_preserve')==preserve['joint_updates']
    shared = [read(RESULTS/'sac_preserve/pretrained'/('s%d.json'%s)) for s in range(3)]
    assert sum(r['imitation_updates'] for r in shared)==preserve['shared_imitation_updates']
    assert sum(r['critic_updates'] for r in shared)==preserve['shared_critic_updates']
    teacher_value = read(RESULTS/'teacher_value/audit.json')
    supervised = read(RESULTS/'teacher_value/completed.json')
    assert sum(r['updates'] for r in runs if r['group']=='teacher_value')==supervised['supervised_models']*supervised['supervised_updates_per_model']==18000

    source_budgets = {
        'teacher_value':teacher_value['budget'], 'sac_teacher':teacher_audit['budget'],
        'sac_preserve':{k:v for k,v in preserve.items() if k not in ('passed','audit_source_sha256')},
        'categorical_frozen':{k:categorical[k] for k in ('new_formal_training_steps','smoke_training_steps_separate',
            'physical_transitions_recomputed','resource_interrupted_extra_transitions_lower_bound',
            'resource_interrupted_extra_transitions_upper_bound')},
        'sac_teacher_cause':{k:v for k,v in read(RESULTS/'sac_teacher_cause/audit.json').items()
            if isinstance(v,(int,float)) and not isinstance(v,bool)},
        'paper_transfer_2026-09-23':read(RESULTS/'paper_transfer_2026-09-23/audit.json')['budget'],
        'plateau_screen':{k:v for k,v in read(RESULTS/'plateau_screen/audit.json').items()
            if k in ('episodes','transitions')},
    }
    coverage = []
    for folder in sorted(RESULTS.iterdir()):
        if not folder.is_dir() or folder.name=='campaign_inventory_2026-09-24': continue
        sources = [str(p.relative_to(ROOT)) for p in sorted(folder.glob('*.json'))
            if any(word in p.name for word in ('audit','budget','completed','protocol'))]
        coverage.append({'group':folder.name,'author_manifest_inventory':folder.name in old['groups'],
            'supplemental_training_recount':any(r['group']==folder.name for r in runs),
            'supplemental_budget_fields':folder.name in source_budgets,
            'top_level_metadata_sources':sources})

    result = {'scope':'Supplement to author-format inventory; no all-project grand total is claimed.',
        'no_training_or_simulation':True,'sealed_test_outcomes_read':False,
        'training_counts_reconciled':True,'physics_or_model_inference_reaudit':False,
        'runs':runs,'interruptions':interruptions,'source_budget_fields':source_budgets,'coverage':coverage,
        'overlap_explanations':[
            'categorical_frozen/audit.json 135000 = 90000 modern steps + 45000 stationary_terminal TF1 steps already in original inventory.',
            'sac_preserve inherited_branch_study_whole_physical_steps 217730 is the teacher_value study total, not new SAC-preserve work.',
            'sac_preserve inherited_teacher_transitions 18000 are reused from sac_teacher, not newly simulated transitions.',
            'Supervised updates, environmental transitions, evaluation steps, reset warmups and sampled branch calls have different units.',
            'Checkpoint files are repeated saves, not separate training runs. Historical report budgets can include old evaluation replay.'],
        'limits':['Categorical final partial episodes are absent from saved episode tables: 414 steps rely only on matching progress/completion metadata.',
            'Existing audit budget fields are preserved with hashes; this pass independently recounts specified training records only.',
            'Historical unflushed work and uninstrumented constructor, reset or diagnostic operations remain unknown.',
            'Remaining groups are indexed for coverage; metadata presence is not a complete compute ledger.',
            'No historical or current efficacy claim is upgraded by this bookkeeping audit.'],
        'hashes':hashes}
    result['additional_formal_environment_training_steps']=sum(r['environment_training_steps'] for r in runs)
    assert result['additional_formal_environment_training_steps']==351000
    result['steps_supported_only_by_metadata']=sum(r['steps_supported_only_by_metadata'] for r in runs)
    assert result['steps_supported_only_by_metadata']==414
    for name,rows in [('runs.csv',runs),('interruptions.csv',interruptions),('coverage.csv',coverage)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result['script_sha256']=sha(Path(__file__))
    (OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 历史预算补充：遗漏格式与重复计数核对','',
        '这次只核对已有训练文件和预算字段，没有训练、仿真、模型反序列化或读取封存测试表现。原作者格式清单保持不变。','',
        '|分组|原清单遗漏的环境训练步|监督/联合更新|证据与边界|',
        '|---|---:|---:|---|']
    for group in ('categorical_frozen','sac_teacher','sac_preserve','teacher_value'):
        rr=[r for r in runs if r['group']==group]
        lines.append('|%s|%d|%d|%d 个保存模型；逐运行路径、计数依据及散列见 runs.csv / inventory.json|' %
            (group,sum(r['environment_training_steps'] for r in rr),sum(r['updates'] for r in rr),len(rr)))
    lines += ['', '上述环境训练步合计 351,000，未包含在原 168 份作者格式 manifest 清单中。该数字不是全部物理仿真预算，也不包含本表单独列出的监督分支采集、验证、测试、重置与中断工作。', '',
        '其中 categorical 的六个末尾未完成回合共 414 步只由 progress/completed 两份元数据支持，完整回合表没有保存它们。其余 SAC 教师与保持方法的转移逐行重新解析计数。此证据强度差异在 runs.csv 单列；没有补造缺失逐步数据。','',
        '## 必须去重的继承预算','',
        '- categorical 报告的 135,000 步含 45,000 步 stationary_terminal TF1 训练，后者已在原清单中；本补充只增加 90,000。',
        '- sac_preserve 的 217,730 步是继承的整个 teacher_value 研究预算，18,000 条教师转移来自 sac_teacher；两者不是 sac_preserve 新仿真。',
        '- sac_preserve 另外有共享模仿 3,000 次、critic 预训练 3,000 次更新，未重复算作环境训练步。',
        '- teacher_value 的 18,000 次监督更新复用已保存分支；其采集、评价、重置与烟测合计 217,730 次物理步，不能用监督更新数代替仿真成本。','',
        '## 中断与额外仿真','',
        '- categorical 四次内存中断：额外训练步记录下界 14,700，上界 15,100；reset 和初始化另计。',
        '- sac_teacher 关机中断：重新逐行计得 19,348 条已保存转移，34 次观察到的 reset；未刷盘工作未知。其研究物理步下界为 301,753。',
        '- sac_preserve 已有审计记录新增物理步 344,928（含训练、评价、reset 和烟测），继承开销单列。',
        '- sac_teacher_cause 机制诊断为 183,144 物理步、零 SAC 更新；不能隐藏为免费诊断。',
        '- paper_transfer 验证与旧保留评价分别为 183,901 / 292,980 步，另有 1,824 烟测步及 1,166 reset；零新训练更新。',
        '- plateau_screen 记录 51,843 个评价转移；该计数不自动包含所有初始化和重置开销。','',
        '上述总计与子项不能再次相加。各原审计的预算字段原样保存在 source_budget_fields，并记录来源 SHA256。没有重新把历史评价结果作为新独立证据。','',
        '## 仍然存在的记录限制','',
        '全结果目录的 coverage.csv 标明每组是否由原清单、本补充或仅元数据索引覆盖。无法从文件恢复的未刷盘/未计量工作继续标为未知，不制造一个看似精确的全项目总数。此核对不升级任何控制效果或复现结论。','',
        '复运行：`.venv/bin/python experiments/bohn2021_reproduction/campaign_budget_supplement.py`。']
    (OUT/'report_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'additional_formal_training_steps':351000,'model_records':len(runs),'interrupted_attempts':len(interruptions),
        'coverage_groups':len(coverage),'sealed_test_read':False})


if __name__=='__main__':main()
