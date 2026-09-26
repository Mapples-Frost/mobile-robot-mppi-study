"""Summarize every predeclared mechanism probe, with descriptive limits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import ART, ROOT
from run import write
from mechanism_probe import OUT, HORIZONS


def load(path):
    return json.loads(path.read_text())


def stats(a):
    a=np.asarray(a,dtype=float)
    return {'mean':float(np.mean(a)), 'median':float(np.median(a)),
        'min':float(np.min(a)), 'max':float(np.max(a))}


def main():
    report=ART/'report/mechanism_probe';report.mkdir(exist_ok=True)
    result={}; total_rollouts=0; total_steps=0
    for task in ['vehicle','pendulum']:
        folders=[OUT/('%s_case%02d'%(task,i)) for i in range(10)]
        assert all((p/'completed.json').exists() for p in folders), 'Probe still running: '+task
        cases=[load(p/'completed.json') for p in folders]
        keys=list(cases[0]['sweeps'])
        costs={k:np.array([c['sweeps'][k]['total_cost'] for c in cases]) for k in keys}
        fixed=np.array([costs['h%d'%h] for h in HORIZONS]).T
        best=int(np.argmin(fixed.mean(axis=0)))
        rows=[]
        for k in keys:
            rows.append({'policy':k,**stats(costs[k]),
                'goals':sum(c['sweeps'][k]['termination']=='goal' for c in cases),
                'constraints':sum(c['sweeps'][k]['termination']=='constraint' for c in cases),
                'per_case':costs[k].tolist()})
        for folder,c in zip(folders,cases):
            assert c['weights_frozen'] and c['exact_prefix_verified'] and c['costs_independently_verified']
            initial=c['sweeps']['h10']['initial_state']
            for k in keys:
                if not k.startswith('exact_forecast'):
                    assert c['sweeps'][k]['initial_state']==initial, (task,folder.name,k)
            for p in folder.glob('*.json'):
                d=load(p)
                if isinstance(d,dict) and 'trace' in d:
                    total_rollouts+=1;total_steps+=len(d['trace'])
                    assert np.isclose(sum(t['cost'] for t in d['trace']),d['total_cost'])
                    assert np.isclose(sum(t['cost']*.97**i for i,t in enumerate(d['trace'])),d['discounted_cost'])
        branch_rows=[];one_switch=[];base_h=10 if task=='vehicle' else 30
        for case_id,c in enumerate(cases):
            baseline_trace=load(folders[case_id]/('h%d.json'%base_h))['trace']
            possibilities=[costs['h%d'%base_h][case_id]]
            for branch in c['branches']:
                cs=branch['candidates'];base=next(r for r in cs if r['horizon']==base_h)
                best_und=min(cs,key=lambda r:r['total_cost'])
                best_disc=min(cs,key=lambda r:r['discounted_cost'])
                lookup={r['horizon']:r for r in cs}
                regrets=[lookup[h]['discounted_cost']-best_disc['discounted_cost'] for h in branch['actor_horizons']]
                advantages=[base['discounted_cost']-lookup[h]['discounted_cost'] for h in branch['actor_horizons']]
                possibilities.append(sum(t['cost'] for t in baseline_trace[:branch['anchor']])+best_und['total_cost'])
                branch_rows.append({'case':case_id,'anchor':branch['anchor'],
                    'best_h_discounted':best_disc['horizon'],'best_h_undiscounted':best_und['horizon'],
                    'base_discounted':base['discounted_cost'],
                    'available_improvement_discounted':base['discounted_cost']-best_disc['discounted_cost'],
                    'actor_horizons':branch['actor_horizons'],'actor_regret_discounted':regrets,
                    'actor_advantage_over_base_discounted':advantages})
            one_switch.append(min(possibilities))
        gains=np.array([r['available_improvement_discounted'] for r in branch_rows])
        regrets=np.array([r['actor_regret_discounted'] for r in branch_rows])
        branch_summary={'n_states':len(branch_rows),'n_scenes':len(cases),'base_h':base_h,
            'available_discounted_improvement':stats(gains),'states_with_gain_gt_0_01':int(np.sum(gains>.01)),
            'actor_regret_means':regrets.mean(axis=0).tolist(),
            'actor_regret_medians':np.median(regrets,axis=0).tolist(),
            'actor_within_0_01_of_best_counts':np.sum(regrets<=.01,axis=0).tolist(),
            'best_h_counts':{str(h):sum(r['best_h_discounted']==h for r in branch_rows) for h in sorted(set(r['best_h_discounted'] for r in branch_rows))},
            'best_single_override_episode_costs':list(map(float,one_switch)),
            'best_single_override_mean':float(np.mean(one_switch)),
            'base_mean':float(np.mean(costs['h%d'%base_h])),
            'per_state':branch_rows}
        result[task]={'rows':rows,'best_fixed_h':HORIZONS[best],
            'best_fixed_mean':float(fixed[:,best].mean()),
            'per_scene_fixed_h_oracle_mean':float(fixed.min(axis=1).mean()),
            'per_scene_best_fixed_h':[HORIZONS[i] for i in np.argmin(fixed,axis=1)],
            'branch':branch_summary}
        if task=='vehicle':
            exact=np.array([costs['exact_forecast_h%d'%h] for h in HORIZONS]).T
            best_exact=int(np.argmin(exact.mean(axis=0)))
            result[task]['exact_forecast']={'best_h':HORIZONS[best_exact],
                'best_fixed_mean':float(exact[:,best_exact].mean()),
                'per_scene_fixed_h_oracle_mean':float(exact.min(axis=1).mean()),
                'per_scene_best_h':[HORIZONS[i] for i in np.argmin(exact,axis=1)]}
            result[task]['terminal_ablations']={}
            for mode in ['zero','learned','no_position','no_heading']:
                mat=np.array([costs['rl%d_%s'%(seed,mode)] for seed in range(3)])
                result[task]['terminal_ablations'][mode]={'mean':float(mat.mean()),
                    'seed_means':mat.mean(axis=1).tolist(),'scene_means':mat.mean(axis=0).tolist()}
    # Do not call paired policy evaluations independent scene replicates.
    bank_hashes={};all_keys=set()
    for task in result:
        new=load(OUT/(task+'_bank.json'))['cases']
        def key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        old=set(key(c) for name in ['test','validation','holdout'] for c in load(ART/'configs'/('%s_%s_bank.json'%(task,name)))['cases'])
        new_keys=[key(c) for c in new]
        assert len(set(new_keys))==10 and not set(new_keys)&old
        bank_hashes[task]=hashlib.sha256((OUT/(task+'_bank.json')).read_bytes()).hexdigest()
    audit={'completed_case_jobs':20,'independent_scenes_per_task':10,'saved_rollouts':total_rollouts,
        'saved_transition_rows_including_repeated_branch_suffixes':total_steps,
        'matched_terminal_initial_states':True,'frozen_model_weights':True,
        'exact_branch_prefix_and_baseline_suffix':True,'independent_step_cost_and_bounds_checks':True,
        'all_saved_return_sums_checked':True,'no_duplicate_with_old_banks':True,'bank_sha256':bank_hashes}
    write(OUT/'audit.json',audit);write(OUT/'summary.json',result)
    lines=['# Bøhn 2021：终端价值与时域选择机制诊断','',
        '## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run', '- Origin Date: 2026-09-17',
        '- Verification Status: VERIFIED（执行、配对和数值核验；不代表论文主结论已重现）',
        '- Version Label: mechanism_probe_v1','',
        '## 本轮结论','',
        '已确认当前车辆MPC存在显著的求解初值敏感性：同一状态、同一H10、完全相同的NLP参数，仅换一次初值，后续成本4552.06→27.69，两个解都报告成功。此结果解释了部分灾难性回合，不能推广为所有复现差异的唯一根因。',
        '终端表示限制与隐藏预测上下文也已通过符号/数值干预确认。车辆全部三个种子在新10场景上的平均成本为64.21（学习终端）与44.74（清零终端）；但种子1保留终端反而更好（18.94 vs 38.04）。删除位置项的平均成本4954.64，说明“有表示缺陷”不意味着任意删除相关项就会改善控制。',
        '零终端固定H的排名对场景敏感：新场景最佳为H5（16.11），H10均值468.39但中位数10.13，差异主要由一个停车陷阱场景拉大。逐场景事后选择固定H能降到11.43，但利用了未来结果，尚未证明可以由当前观测稳定学到。',
        '修复优先级：先验证不同初值能否一致改善MPC求解，再将终端函数改为适合相对参考误差的表示并保留同数据对照，同时检查是否应给RL增加可观测的预测不确定性/求解状态。上述修复方向尚未作为正式算法重训，不声称已恢复论文优势。','',
        '## 协议与范围','',
        '使用最终 paper_defaults 的全部三个 RL 种子；不训练、不挑检查点。每任务新生成10个诊断场景，种子26091703，与旧训练/验证/测试场景文件分开。所有声明的条件均报告。',
        '终端消融统一以零终端价值执行作者 reset 中的 H50 预热，再施加干预，各组正式起始物理状态完全相同。每回合新建MPC以排除跨回合求解器历史。预测噪声消融从预热前开始改变预测，属于预测器的总体干预，不声称预热后的状态完全相同。',
        '固定H扫描双方均无终端项，不能代替论文中每个固定H独立学习终端项的正式比较。RL原本与终端项联合训练，移除终端项会改变它所面对的闭环环境。','',
        '## 固定时域扫描与全部RL种子','']
    for task,title in [('vehicle','车辆'),('pendulum','倒立摆')]:
        d=result[task]
        lines += ['### '+title,'','|策略|平均成本|中位数|最小–最大|到达/约束终止|','|---|---:|---:|---:|---|']
        for r in d['rows']:
            lines.append('|%s|%.4f|%.4f|%.4f–%.4f|%d/%d|'%(r['policy'],r['mean'],r['median'],r['min'],r['max'],r['goals'],r['constraints']))
        lines+=['','最佳已测零终端固定时域为 H=%d，平均 %.4f。逐场景事后选择最优固定H的平均为 %.4f；它利用完整回合的未来结果，仅表示场景间差异，不能作为可部署策略。'%(d['best_fixed_h'],d['best_fixed_mean'],d['per_scene_fixed_h_oracle_mean']), '']
    lines+=['## 相同状态分支：只改一次H，再回到同一固定策略','',
        '车辆基线H10，倒立摆H30。锚点为第0、20、40、60步（已终止回合不再取样）。每个分支重新播放同一前缀，逐步断言物理状态完全一致；基线分支的后缀也与原轨迹完全一致。候选包括预先指定的H以及三个RL策略在该共同状态下实际选择的H。记录直至回合结束的完整回报，分别计算不折扣成本和0.97折扣成本。',
        '这是共同固定策略后续下的动作比较，不是RL critic真值：后续采用固定策略而非原RL策略，且这些状态可能偏离RL训练分布。因此局部动作劣势不能直接证明SAC优化器本身失效。','']
    for task,title in [('vehicle','车辆'),('pendulum','倒立摆')]:
        b=result[task]['branch']
        lines += ['### '+title,'',
            '- %d个共同状态，来自10个独立场景；其中%d个状态存在超过0.01的折扣成本改善。'%(b['n_states'],b['states_with_gain_gt_0_01']),
            '- 三个RL种子的平均局部遗憾值（所选H成本减候选中最小成本）：'+', '.join('%.4f'%v for v in b['actor_regret_means'])+'。',
            '- 三个种子的中位局部遗憾值：'+', '.join('%.4f'%v for v in b['actor_regret_medians'])+'。',
            '- 在0.01范围内选到候选最优的状态数：'+str(b['actor_within_0_01_of_best_counts'])+'。',
            '- 基线整回合平均成本 %.4f；每场景事后选取一次最有利H干预及其时刻的整回合平均成本 %.4f。'%(b['base_mean'],b['best_single_override_mean']),
            '- 候选最优H频数：'+json.dumps(b['best_h_counts'])+'。','']
    lines+=['这个一次干预的事后参考使用未来结果，不计其搜索计算成本，不能作为论文性能复现。如果它仍落后于最佳固定H，也不能排除更复杂的多次自适应策略有收益。','',
        '倒立摆逐场景成本最优H在三个全局困难场景中选到H5，但那些回合依然违反约束；有限惩罚下较低综合成本不能自动解释为成功率改善。全部成功/失败均保留。','',
        '## 终端表示审计','',
        '实际输入顺序为 theta, x, y, goal_x, goal_y。SAC观测缩放不会传到独立的终端经验池。作者poly包含线性项和逐维平方项，没有交叉项；CasADi符号微分确认所有状态–目标混合二阶导数恒为0，TF与显式多项式数值相符。',
        '恢复代码中相对目标坐标的处理仅见于注释。终端输入没有当前轨迹参考、障碍几何及时间；这限制了准确表达一般跟踪任务的长期价值。不能据此断定作者外部训练工程没有其他预处理。','']
    terminal=load(OUT/'terminal_structure_audit.json')
    for r in terminal['rows']:
        lines.append('- 种子%d：状态Hessian特征值 %s；旧20场景中有/无终端reset的最大位置差 %.6f m，最大航向差 %.6f rad。'%(r['seed'],str(r['state_hessian_eigenvalues']),r['max_reset_position_delta'],r['max_reset_heading_delta']))
    solver=load(OUT/'solver_case06/completed.json')
    alias=load(OUT/'observation_alias/completed.json')
    assert solver['same_nlp_parameters_verified'] and solver['same_prefix_verified']
    assert alias['same_observations_and_states_exact']
    lines+=['','## 补充诊断：相同NLP的局部解','',
        '这是主扫描发现困难场景后的事后机制检查，未用于选择正式策略。车辆场景6中，H5整回合成本4.757左右，H10为4574.235，H10每一步求解器均报告成功。固定H10与零终端价值，只在第20、30、40步之一改变NLP初值，然后恢复原控制流程。所有条件的NLP参数哈希完全相同，前缀状态完全相同。','',
        '|干预时刻|初值|当次NLP目标值|后续整段成本|最大约束残差|','|---|---|---:|---:|---:|']
    for r in solver['rows']:
        assert r['local']['solver_success']
        assert r['local']['constraint_max_violation']<1e-6
        lines.append('|%d|%s|%.6f|%.6f|%.2e|'%(r['anchor'],r['mode'],r['local']['objective'],r['suffix_cost'],r['local']['constraint_max_violation']))
    lines+=['','第30步从warm改为left初值，当次目标从46.69降到32.61，后续成本从4552.06降到27.69。相同问题有明显不同的可行局部解，原热启动被较差的停止解困住。第20步还出现反例：right初值的短期目标更差，但后续成本大幅更好，说明有限时域目标和长期闭环效果也可能不一致。不能声称简单冷启动或仅选最低当次目标就解决所有问题。',
        '', '## 补充诊断：RL看不到的预测上下文','',
        '作者控制器每回合固定object_noise_seed，RL当前14维观测包含真实障碍位置/半径，不包含该种子或完整预测。三个场景在第20步保持物理状态、RL观测、求解初值完全相同，只把预测扰动种子取原值、零、相反数。',
        '共27个单步条件，最大控制分量差 %.6f，最大下一位置差 %.6f m。'% (alias['max_input_delta'],alias['max_position_delta']),
        '这证明当前观测遗漏了会影响控制转移的持久上下文。它不是“有噪声就不满足Markov”的论证；关键在于该隐藏参数在整回合中持续存在。原始训练工程是否提供了额外预测特征仍未知，也未证明补充这些输入就能恢复论文优势。','']
    lines+=['','## 核验与解释边界','',
        json.dumps(audit,ensure_ascii=False,indent=2), '',
        '小样本、多策略配对、三个训练种子；不报告统计显著性，也不将多个回放当成独立场景。均值对极端困难场景敏感，表中同时报告中位数和范围。预测误差清零同时改变MPC可用信息；只能解释该机制的总体效应。',
        '首轮试运行在读取标量参考值时触发TypeError，未产出完整回合；修正为统一标量转换后完成运行。失败日志保留。原作者源码未修改。',
        '补充求解器诊断首次把CasADi结构底层向量设成NumPy数组，触发类型错误；改为CasADi DM后12个条件均完成。该失败未计入结果。',
        '', '## 复运行','', '```bash',
        '/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/mechanism_probe.py --workers 8',
        '/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/mechanism_terminal_audit.py',
        '/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/mechanism_solver_probe.py',
        '/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/mechanism_observation_probe.py',
        '.venv/bin/python experiments/bohn2021_reproduction/mechanism_report.py','```','']
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    with (report/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## 图形\n\n![相同优化问题，不同初值](solver_mechanism.png)\n\n![固定时域的场景敏感性](fixed_h_sensitivity.png)\n')
    print(json.dumps({task:{'best_fixed_h':d['best_fixed_h'],'best_fixed_mean':d['best_fixed_mean'],
        'per_scene_oracle':d['per_scene_fixed_h_oracle_mean'],'single_override':d['branch']['best_single_override_mean'],
        'branch_regrets':d['branch']['actor_regret_means']} for task,d in result.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
