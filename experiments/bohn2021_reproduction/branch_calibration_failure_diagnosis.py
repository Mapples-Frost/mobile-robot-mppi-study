"""Validation-only descriptive coverage/failure audit; no simulation or tuning."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24'
ARMS=('actor','raw_greedy','calibrated_greedy','fixed')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    gate=read(OUT/'validation_gate.json')
    assert gate['data_audit_passed']
    for name,h in gate['hashes'].items(): assert sha(Path(name))==h,name
    training=read(OUT/'training_audit.json')
    assert training['passed']
    for name,h in training['hashes'].items(): assert sha(Path(name))==h,name
    rows, summaries, contrasts, hashes=[],[],[],{}
    for task in ('vehicle','pendulum'):
        for seed in range(3):
            folder=OUT/('%s_s%d'%(task,seed))
            dp=folder/'dataset.json'; data=read(dp)
            hashes[str(dp)]=sha(dp)
            train=np.asarray([g['observation'] for g in data['groups']])
            groups=np.asarray([g['case'] for g in data['groups']])
            loo=[]
            for i,obs in enumerate(train):
                other=train[groups!=groups[i]]
                assert len(other)
                loo.append(float(np.min(np.linalg.norm(other-obs,axis=1))))
            reference=float(np.percentile(loo,95))
            by_arm={}
            for arm in ARMS:
                path=OUT/'evaluations/validation'/task/('%s_s%d'%(arm,seed))
                episodes=sorted(read(path/'summary.json')['episodes'],key=lambda e:e['episode'])
                per_arm=[]
                for e in episodes:
                    p=path/('trace_%02d.json'%e['episode']); trace=read(p)
                    hashes[str(p)]=sha(p)
                    obs=np.asarray([r['observation'] for r in trace])
                    distances=np.min(np.linalg.norm(obs[:,None,:]-train[None,:,:],axis=2),axis=1)
                    assert np.all(np.isfinite(distances))
                    failed=[i for i,r in enumerate(trace) if not r['solver_success']]
                    row={'task':task,'seed':seed,'arm':arm,'case':e['episode'],'cost':e['total_cost'],
                         'physical_cost':e['performance_cost'],'h_proxy_cost':e['computation_cost'],
                         'constraint_cost':e['constraint_cost'],
                         'steps':len(trace),'termination':e['termination'],'solver_failed_steps':len(failed),
                         'first_solver_failure_step':failed[0] if failed else None,
                         'h_at_first_solver_failure':trace[failed[0]]['horizon'] if failed else None,
                         'first_h':trace[0]['horizon'],'min_h':min(r['horizon'] for r in trace),
                         'max_h':max(r['horizon'] for r in trace),'mean_h':float(np.mean([r['horizon'] for r in trace])),
                         'fraction_h_below_10':float(np.mean([r['horizon']<10 for r in trace])),
                         'mean_nearest_training_anchor_distance':float(distances.mean()),
                         'initial_nearest_training_anchor_distance':float(distances[0]),
                         'train_leave_case_out_p95_distance':reference,
                         'fraction_distance_above_train_reference':float(np.mean(distances>reference))}
                    rows.append(row);per_arm.append(row)
                by_arm[arm]=per_arm
                summaries.append({'task':task,'seed':seed,'arm':arm,'mean_cost':float(np.mean([r['cost'] for r in per_arm])),
                    'successes':sum(r['termination']==('goal' if task=='vehicle' else 'steps') for r in per_arm),
                    'solver_failed_steps':sum(r['solver_failed_steps'] for r in per_arm),
                    'mean_episode_distance':float(np.mean([r['mean_nearest_training_anchor_distance'] for r in per_arm])),
                    'mean_fraction_above_train_reference':float(np.mean([r['fraction_distance_above_train_reference'] for r in per_arm]))})
            for comparator in ('actor','raw_greedy','fixed'):
                for calibrated,other in zip(by_arm['calibrated_greedy'],by_arm[comparator]):
                    assert calibrated['case']==other['case']
                    contrasts.append({'task':task,'seed':seed,'case':calibrated['case'],'comparator':comparator,
                        'cost_difference':calibrated['cost']-other['cost'],
                        'calibrated_termination':calibrated['termination'],'comparator_termination':other['termination'],
                        'solver_failed_step_difference':calibrated['solver_failed_steps']-other['solver_failed_steps'],
                        'initial_distance':calibrated['initial_nearest_training_anchor_distance']})
    assert len(rows)==240 and len(contrasts)==180
    failure_modes=[]
    for task in ('vehicle','pendulum'):
        for seed in range(3):
            for arm in ARMS:
                rr=[r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,arm)]
                assert len(rr)==10 and {r['case'] for r in rr}==set(range(10))
                failure_modes.append({'task':task,'seed':seed,'arm':arm,
                    'h_always_one_cases':[r['case'] for r in rr if r['max_h']==1],
                    'non_success_without_solver_failures':[r['case'] for r in rr
                        if r['termination']!=('goal' if task=='vehicle' else 'steps') and r['solver_failed_steps']==0],
                    'cost_share_from_solver_failure_episodes':sum(r['cost'] for r in rr if r['solver_failed_steps'])/sum(r['cost'] for r in rr),
                    'max_cost_case':max(rr,key=lambda r:r['cost'])['case'],
                    'max_case_share_of_total_cost':max(r['cost'] for r in rr)/sum(r['cost'] for r in rr)})
    result={'scope':'Post-hoc validation-only diagnostic. No retraining, new simulation, checkpoint or scene selection.',
        'definition':'Euclidean distance in the original fixed scaled observation coordinates to the same-seed calibration anchors. Reference is 95th percentile of training nearest-neighbor distances excluding the entire same source episode.',
        'limitations':['Distances depend on feature scaling and are descriptive, not a calibrated out-of-distribution detector.',
            'Fixed-H terminal value and reset warmup differ from adaptive arms; comparisons are not H-only causal effects.',
            'All three adaptive arms share terminal/actor weights and reset construction; later state differences result from their executed decisions.',
            'H, solver failures and coverage associations do not identify a unique cause. No scenes are dropped or acceptance criteria changed.',
            'No independent test outcomes are read.'],
        'summaries':summaries,'episodes':rows,'paired_contrasts':contrasts,'failure_modes':failure_modes,'hashes':hashes,
        'script_sha256':sha(Path(__file__)),
        'validation_gate_sha256':sha(OUT/'validation_gate.json'),
        'training_audit_sha256':sha(OUT/'training_audit.json')}
    (OUT/'failure_coverage_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in [('failure_coverage_episodes.csv',rows),('failure_coverage_paired.csv',contrasts),('failure_modes.csv',failure_modes)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 完整验证轨迹的失败与训练覆盖诊断', '',
        '这是事后描述，不改变本轮模型、场景、阈值或结论门槛；没有新增仿真、没有读取测试。', '',
        '|任务|种子|方法|平均成本|成功/10|求解失败步|平均最近训练锚点距离|超出训练距离参考的平均步比例|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|{task}|{seed}|{arm}|{mean_cost:.3f}|{successes}/10|{solver_failed_steps}|{mean_episode_distance:.3f}|{mean_fraction_above_train_reference:.1%}|'.format(**r))
    lines+=['','距离使用模型原固定缩放观测，不在验证集重新标准化。参考是训练锚点排除同源场景后最近邻距离的95百分位，只是描述参照，不是经过验证的分布外阈值。',
        '完整所有场景的 H 范围、短 H 比例、首次求解失败步、终止原因和成本配对差均在 CSV；不根据覆盖距离排除场景。',
        '这些关联无法独立证明分布偏移、某个 H 或求解器是唯一原因。固定 H 使用自己的终端函数，故与自适应方法比较不是只改 H 的因果实验。']
    lines+=['','## 校准策略的失败模式（全部种子）','','|任务|种子|全程 H1 的场景|无求解失败仍未成功的场景|最大成本场景|其成本占该种子总成本|含求解失败场景的成本占比|',
        '|---|---:|---|---|---:|---:|---:|']
    for r in failure_modes:
        if r['arm']=='calibrated_greedy':
            lines.append('|{task}|{seed}|{h_always_one_cases}|{non_success_without_solver_failures}|{max_cost_case}|{max_case_share_of_total_cost:.1%}|{cost_share_from_solver_failure_episodes:.1%}|'.format(**r))
    lines+=['','场景编号从 0 开始。成本集中度只用于描述，最大成本场景和失败场景全部继续计入预定均值；含求解失败场景的成本占比不能解释为求解失败导致的成本比例。']
    (OUT/'failure_coverage_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'episodes':len(rows),'paired_contrasts':len(contrasts),'test_read':False})


if __name__=='__main__':main()
