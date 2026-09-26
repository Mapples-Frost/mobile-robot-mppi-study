"""Reconcile current-study attempts/updates; inherited histories stay separate."""
import json
import hashlib
import time
from pathlib import Path
from conservative_iteration import OUT,verify
from paper_h_soft_probe import read,digest
from run import write


def main():
    verify();attempts=[];fits=[];extra=[];hashes={}
    def snapshot_read(path):
        raw=path.read_bytes()
        hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for path in sorted(OUT.rglob('attempt_*.json')):
        a=snapshot_read(path);rel=path.relative_to(OUT);top=rel.parts[0]
        if top=='reset_diagnosis':stage='reset_diagnosis'
        elif top.startswith('solver_initialization_probe'):stage='solver_initialization_probe'
        elif top.startswith('smoke_'):stage='smoke'
        elif top=='banks':stage='bank_generation'
        elif top=='evaluations':stage='evaluation_'+rel.parts[1]
        elif top.startswith('timing_'):stage=top
        else:stage='branch_collection'
        attempts.append(dict(path=str(path),stage=stage,step_calls=a['step_calls'],reset_calls=a['reset_calls'],
            environment_constructions=1,pid=a['pid']))
        # Snapshot hashes are evidence of the read counters, not a freeze on active files.
    for path in sorted(list(OUT.glob('*_s*_r*/fit_completed.json'))+list(OUT.glob('smoke_*/smoke_fit_completed.json'))):
        f=read(path)
        assert digest(path.parent/('smoke_policy.json' if f['smoke'] else 'policy.json'))==f['policy_hash']
        fits.append(dict(path=str(path),task=f['task'],seed=f['seed'],round=f['round'],smoke=f['smoke'],
            updates=f['training_updates'],groups=f['groups'],members=len(f['members'])))
        hashes[str(path)]=digest(path)
    for path in sorted(OUT.rglob('instrumentation_attempt_*.json')):
        a=snapshot_read(path);extra.append(dict(path=str(path),smoke='extra_fixed_smoke' in path.parts,**a))
    totals={}
    for a in attempts:
        group=totals.setdefault(a['stage'],dict(step_attempts=0,reset_attempts=0,environment_constructions=0))
        group['step_attempts']+=a['step_calls'];group['reset_attempts']+=a['reset_calls'];group['environment_constructions']+=1
    totals['supervised_fit']=dict(formal_completed_updates=sum(f['updates'] for f in fits if not f['smoke']),
        smoke_completed_updates=sum(f['updates'] for f in fits if f['smoke']),formal_models=sum(not f['smoke'] for f in fits),
        optimizer_members=sum(f['members'] for f in fits if not f['smoke']))
    totals['extra_fixed']=dict(step_attempts=sum(a['step_attempts'] for a in extra),
        step_completed=sum(a['step_completed'] for a in extra),reset_attempts=sum(a['reset_attempts'] for a in extra),
        reset_completed=sum(a['reset_completed'] for a in extra),environment_constructions=sum(a['environment_constructions'] for a in extra))
    completed_extra=[]
    for path in sorted(list(OUT.glob('extra_fixed/*/completed.json'))+list(OUT.glob('extra_fixed_smoke/*/completed.json'))):
        f=read(path);completed_extra.append(dict(path=str(path),smoke='extra_fixed_smoke' in path.parts,training_steps=f['steps'],updates=f['updates']))
    totals['extra_fixed'].update(formal_training_steps=sum(f['training_steps'] for f in completed_extra if not f['smoke']),
        smoke_training_steps=sum(f['training_steps'] for f in completed_extra if f['smoke']),
        formal_updates=sum(f['updates'] for f in completed_extra if not f['smoke']),
        smoke_updates=sum(f['updates'] for f in completed_extra if f['smoke']))
    numerical_audits=[]
    for path in sorted(OUT.glob('independent_dynamics/*/completed.json')):
        d=read(path)
        for p,h in d['hashes'].items():assert digest(Path(p))==h
        hashes[str(path)]=digest(path)
        numerical_audits.append(dict(path=str(path),smoke=d['smoke'],passed=d['passed'],
            recorded_transition_comparisons=d['recorded_transition_comparisons'],
            numerical_integrations=d['unique_numerical_integrations'],elapsed_s=d['elapsed_s']))
    solver_probe_budget=None
    probe_audit=OUT/'solver_initialization_probe/array_audit.json'
    if probe_audit.exists():
        probe=read(probe_audit);assert probe['passed']
        solver_probe_budget=probe['counts_including_interruption']
        hashes[str(probe_audit)]=digest(probe_audit)
    process_status={}
    for name in ('pipeline_status.json','posttrain_status.json','baseline_completion_status.json','validation_finish_status.json','validation_status.json','test_status.json'):
        if (OUT/name).exists():process_status[name]=read(OUT/name)
    partial=any(s.get('active',False) and not (name=='validation_finish_status.json' and s.get('stage')=='budget') for name,s in process_status.items())
    results=dict(recorded_unix=time.time(),partial_snapshot=partial,attempts=attempts,fits=fits,extra_fixed_attempts=extra,
        completed_extra_fixed=completed_extra,totals=totals,offline_numerical_audits=numerical_audits,solver_probe_budget=solver_probe_budget,
        inherited=dict(independent_fixed_H_grid_training_steps=300000,selected_fixed_extra_seeds_training_steps=60000,
                       single_step_pilot_formal_steps=19735,single_step_pilot_smoke_steps=1494,
                       single_step_pilot_formal_resets=264,single_step_pilot_smoke_resets=20,
                       note='These are inherited study budgets, not newly executed in this study. Prior SAC/min-Q/calibration/teacher campaigns remain in the historical budget inventory; do not add overlapping parent totals twice.'),
        caveats=['Counters are attempted explicit calls after construction, so interrupted calls remain charged.',
                 'Each reset includes one author H50 warmup; constructor initialization is separately counted as environment constructions, not inferred as an exact hidden step count.',
                 'During active collection a snapshot is a lower bound, not the final total. Completed-fit updates omit any currently running unsaved optimizer steps; completion is needed for exact final counts.',
                 'Branch collection is extra model-based simulation, not direct on-policy RL interaction. Prefix replay, sources and branch suffixes are included in step attempts.',
                 'Instrumentation of extra fixed training logs both training and final with/without-terminal validation; completed training steps/updates are separated.'],
        input_snapshot_hashes=hashes,source_hash=digest(Path(__file__)))
    dest=OUT/'budget';dest.mkdir(exist_ok=True);write(dest/'ledger.json',results)
    lines=['# 本轮预算账本','',
        ('这是运行中快照，未完成工作不当作最终精确总量。' if partial else '当前编排未报告 active；仍应对照最终审计确认各阶段是否齐全。'),'',
        '|阶段|显式 step 尝试|显式 reset 尝试|环境构造|','|---|---:|---:|---:|']
    for stage,t in totals.items():
        if stage=='supervised_fit':continue
        lines.append('|%s|%d|%d|%d|'%(stage,t['step_attempts'],t['reset_attempts'],t['environment_constructions']))
    f=totals['supervised_fit'];lines+=['','已保存正式模型 %d 份、成员 %d 个、监督更新 %d 次；冒烟更新 %d 次。'%(f['formal_models'],f['optimizer_members'],f['formal_completed_updates'],f['smoke_completed_updates']), '',
        '额外固定基线正式训练 %d 步/%d 次更新，记录器冒烟训练 %d 步/%d 次更新；其附带评估包含在上表 extra_fixed 显式调用中。'%(totals['extra_fixed']['formal_training_steps'],totals['extra_fixed']['formal_updates'],totals['extra_fixed']['smoke_training_steps'],totals['extra_fixed']['smoke_updates']), '',
        '继承固定 H 搜索训练300,000步，补足所选固定 H 独立种子60,000步。本轮复用这些终端模型，双方终端机会一致；自适应策略额外付出分支仿真与监督拟合预算，不将该开销隐去。','',
        '先行单步试验正式19,735步、冒烟1,494步单列继承。其他历史方法见原历史预算补充报告，不重复叠加已包含的父研究总数。每次reset内含H50热身，构造初始化另列，不能把H变短或失败提前终止解释为省时。','']
    if numerical_audits:
        lines+=['','额外离线动力学核验：正式与冒烟合计 %d 次独立积分计算，比较 %d 条保存转移，核验墙钟时间合计 %.3f 秒。相同输入缓存积分，所有记录输出仍逐项比较；这是额外数值计算，不是新增MPC调用或训练经验。'%(sum(a['numerical_integrations'] for a in numerical_audits),sum(a['recorded_transition_comparisons'] for a in numerical_audits),sum(a['elapsed_s'] for a in numerical_audits))]
    if solver_probe_budget:
        lines+=['','求解初值诊断额外原始NLP求解 %d 次（含中断1次），不推进物理系统；其7次环境步、7次reset和4次构造已在上表solver_initialization_probe项计入，不能重复相加。'%(solver_probe_budget['raw_solver_attempts'])]
    (dest/'report_CN.md').write_text('\n'.join(lines));print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
