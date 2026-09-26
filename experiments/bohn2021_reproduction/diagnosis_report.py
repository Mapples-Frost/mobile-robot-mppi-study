"""Report every diagnostic and refined outcome, including negative findings."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import ART

ap=argparse.ArgumentParser();ap.add_argument('--group',default='refined',choices=['refined','paper_defaults']);args=ap.parse_args()
dest=ART/'report'/('diagnosis' if args.group=='refined' else 'paper_defaults');dest.mkdir(parents=True,exist_ok=True)
diag=ART/'results/diagnosis';refined=ART/'results'/args.group
def read(p):return json.loads(p.read_text())
def metric(folder,mode):
    s=read(folder/mode/'summary.json')
    return {'name':folder.name,'mean_cost':s['mean_total_cost'],'goals':s['goal_episodes'],
            'constraints':s['constraint_episodes'],'episodes':len(s['episodes']),
            'mean_horizon':float(np.mean([r['mean_horizon'] for r in s['episodes']])),
            'solver_failures':sum(r['solver_failure_steps'] for r in s['episodes']),
            'mean_performance':float(np.mean([r['performance_cost'] for r in s['episodes']])),
            'mean_computation':float(np.mean([r['computation_cost'] for r in s['episodes']])),
            'mean_constraint':float(np.mean([r['constraint_cost'] for r in s['episodes']]))}

def vehicle_common_cost(ev):
    cases=read(ART/'configs/vehicle_validation_bank.json')['cases']
    totals=[]
    for j,case in enumerate(cases):
        trace=read(ev/('trace_%02d.json'%j))
        totals.append(sum((r['state']['x']-case['tvp']['trajectory_x'][t+2]['true'][0])**2+
                          (r['state']['y']-case['tvp']['trajectory_y'][t+2]['true'][0])**2+
                          r['compute']+r['constraint'] for t,r in enumerate(trace)))
    return float(np.mean(totals))

lines=['# 核心方法复现与故障定位结果','',
    '原26组结果保留。本报告对应追加的冻结干预、单因素重训以及时间对齐＋固定观测尺度的复现。原实验配置和原测试集未公开找回，不能称作者实验的逐数值复制。','',
    '## 已确认的定位结果','',
    '- 车辆后期退化主要随最终时域策略出现：早期策略替换成最终终端值仍可到达10/10，最终策略换回早期终端值仍失败。具体干预如下。',
    '- 同一782个观测上，种子2的H=1比例从0增至32.5%；种子1的H=50比例从0增至38.4%。边界聚集与退化同时出现，不能单凭相关性将其归因于熵系数。',
    '- 修正了重置预执行一步造成的TVP时间错位，以及终端TD训练使用下一状态配当前参考的阶段成本错位。两任务逐步数值检查通过。',
    '- 作者对角二次终端模型并不保证凸。将H5模型的负状态曲率截零后，旧测试成本由3407.941升至10410.753，到达1/10；未采用这种投影。','',
    '|种子|时域策略|终端价值|平均成本|到达/10|碰撞/10|',
    '|---|---|---|---:|---:|---:|']
if args.group=='paper_defaults':
    lines[2]+=' 本页正式结果使用引用链校正的SAC设置：batch=256、replay容量1000000、固定熵系数1.0。依据Bøhn第4节及Haarnoja 2018补充材料附录D表1；保留Bøhn明确修改的32单元actor、gamma=0.97和作者分母式reward scaling。'
cross=[]
for seed in [1,2]:
    p=diag/('crossover_s%d'%seed)/'completed.json'
    if p.exists():
        for r in read(p)['rows']:
            cross.append(dict(r,seed=seed))
            lines.append('|%d|%s|%s|%.3f|%d|%d|'%(seed,r['actor'],r['value'],r['mean_total_cost'],r['goals'],r['collisions']))
lines+=['','early=5000步，late=15000步；这些是事后诊断，未用它们选择测试检查点。','',
        '## 单因素重训（独立验证集）','',
        '|变更|统一时间基准成本|原记录成本|到达/10|碰撞/10|平均H|','|---|---:|---:|---:|---:|---:|']
diagnostics=[]
original=diag/'original_s1_validation'
if (original/'completed.json').exists():
    r=metric(diag,'original_s1_validation');r['variant']='original';r['common_cost']=vehicle_common_cost(original);diagnostics.append(r)
    lines.append('|original|%.3f|%.3f|%d|%d|%.2f|'%(r['common_cost'],r['mean_cost'],r['goals'],r['constraints'],r['mean_horizon']))
for name in ['aligned','scaled','fixed_entropy','no_online_value']:
    folder=diag/('vehicle_'+name+'_s1')
    mode='eval_no_value' if name=='no_online_value' else 'eval_value'
    if (folder/'completed.json').exists():
        r=metric(folder,mode);r['variant']=name;r['common_cost']=vehicle_common_cost(folder/mode);diagnostics.append(r)
        lines.append('|%s|%.3f|%.3f|%d|%d|%.2f|'%(name,r['common_cost'],r['mean_cost'],r['goals'],r['constraints'],r['mean_horizon']))
lines+=['','每组车辆种子1、15000条训练经验。为避免时间修正造成评分口径不同，统一时间基准成本从已保存轨迹、同一物理时刻参考重新计算；原记录成本亦保留。no_online_value在训练及此表评估时不反馈终端价值，仅作消融。其余组继续联合学习终端价值。验证场景与旧测试及下方保留测试均独立。',
        '', '## 最终方法复现（独立保留测试）','',
        '修正组保留作者SAC＋连续输出取整H＋非线性MPC＋32步二次终端价值学习。新增固定可逆的观测单位/相对坐标变换。每个RL种子和固定H都训练15000步；不按测试表现选检查点。',
        '', '|任务|方法|种子|终端值|平均成本|性能成本|H成本|约束成本|到达/20|约束/20|平均H|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
results=[]
for folder in sorted(p.parent for p in refined.glob('*/completed.json')):
    spec=read(folder/'manifest.json')
    for mode in ['holdout_value','holdout_no_value']:
        if not (folder/mode/'completed.json').exists():continue
        r=metric(folder,mode);r.update(task=spec['task'],seed=spec['seed'],horizon=spec['fixed_horizon'],value=mode=='holdout_value')
        results.append(r)
        lines.append('|%s|%s|%d|%s|%.3f|%.3f|%.3f|%.3f|%s|%d|%.2f|'%(r['task'],
            'RL' if r['horizon'] is None else 'H%d'%r['horizon'],r['seed'],'有' if r['value'] else '无',
            r['mean_cost'],r['mean_performance'],r['mean_computation'],r['mean_constraint'],
            str(r['goals']) if r['task']=='vehicle' else '不适用',r['constraints'],r['mean_horizon']))
conclusions=[]
for task in ['vehicle','pendulum']:
    rl=[r for r in results if r['task']==task and r['value'] and r['horizon'] is None]
    fixed=[r for r in results if r['task']==task and r['value'] and r['horizon'] is not None]
    if len(rl)==3 and len(fixed)==3:
        mean=float(np.mean([r['mean_cost'] for r in rl]));best=min(fixed,key=lambda r:r['mean_cost'])
        change=100*(best['mean_cost']-mean)/abs(best['mean_cost'])
        text='%s：RL三种子平均成本%.3f，测试均值最低的已测固定对照H%d为%.3f，相对成本降低率%.2f%%。'%(task,mean,best['horizon'],best['mean_cost'],change)
        text+=' 本组支持RL低于这些代表性固定对照的平均成本。' if mean<best['mean_cost'] else ' 本组没有重现RL优于最优已测固定对照。'
        conclusions.append({'task':task,'rl_mean':mean,'best_fixed':best,'reduction_percent':change})
        lines+=['',text]
        no=[r for r in results if r['task']==task and not r['value'] and r['horizon'] is None]
        if len(no)==3:
            no_mean=float(np.mean([r['mean_cost'] for r in no]))
            lines+=['','%s的RL终端价值冻结消融：有价值平均%.3f，清零价值平均%.3f；这里是同一已训练策略的干预，不等价于重新训练无终端价值策略。'%(task,mean,no_mean)]
if len(results)==24:
    for task in ['vehicle','pendulum']:
        fs=[r for r in results if r['task']==task and r['value'] and r['horizon'] is not None]
        sets=[]
        for r in fs:
            episodes=read(refined/r['name']/'holdout_value/summary.json')['episodes']
            sets.append({e['episode'] for e in episodes if e['termination']=='constraint'})
        common=sorted(set.intersection(*sets)) if sets else []
        if common:lines+=['','%s的三个固定对照都在场景%s发生约束终止。这提示重建初始分布中存在共同困难场景，但不能据此证明不可控；这些场景全部保留在主表，没有剔除。'%(task,','.join(map(str,common)))]
lines+=['','固定H仅一个训练种子、每任务三个代表性H，测试20回合；不能据此声称优于所有H或证明统计显著性。不同任务及旧/新测试集的绝对成本不可直接混比。倒立摆到达时间上限不是形式稳定性证明。',
        '', '## 核验与交付','',
        '验证/测试分别使用种子26091701/26091702。保存完整轨迹、模型、成本分解和实验清单。固定基线有独立终端模型，无终端值测试只清零MPC终端值。H成本为论文代理量；并行墙钟耗时不用于计算效率结论。']
audit=refined/'audit.json'
if audit.exists():
    a=read(audit);lines+=['','独立核验：%d组训练、%d个保留测试回合；逐步物理成本、时间索引、输入界限、约束终止和训练更新数检查通过。'%(a['training_runs'],a['holdout_episodes'])]
else:lines+=['','独立最终核验尚未完成；当前文件是进度报告。']
lines+=['','[实现与命令](../../../../experiments/bohn2021_reproduction/DIAGNOSIS.md) · [原始26组报告](../final_report.md)']
if args.group=='paper_defaults':lines+=['','[被引用SAC的公开超参数表](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf) · [batch64诊断对照报告](../diagnosis/report.md)']
if (dest/'case0_trajectories.png').exists():
    lines+=['','## 固定场景回放','',
        '以下始终使用预先指定的保留测试场景0，展示全部三个RL种子以及一个固定对照。没有按成功与否选择场景。',
        '', '![预选场景的轨迹和时域](case0_trajectories.png)']
if (dest/'vehicle_replay.gif').exists():
    lines+=['','[车辆种子0动画](vehicle_replay.gif) · [倒立摆种子0动画](pendulum_replay.gif)']

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
if cross:
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,seed in zip(axes,[1,2]):
        mat=np.array([[next(r['mean_total_cost'] for r in cross if r['seed']==seed and r['actor']==actor and r['value']==value) for value in ['early','late','zero']] for actor in ['early','late']])
        ax.imshow(np.log10(mat),vmin=1,vmax=4.3,cmap='YlOrRd',aspect='auto')
        for i in range(2):
            for j in range(3):ax.text(j,i,'%.1f'%mat[i,j],ha='center',va='center',color='white' if mat[i,j]>1500 else 'black')
        ax.set(xticks=range(3),xticklabels=['Early V','Final V','Zero V'],yticks=[0,1],yticklabels=['Early policy','Final policy'],title='Seed %d: mean cost (log color)'%seed)
    fig.savefig(dest/'crossover.png');fig.savefig(dest/'crossover.pdf');plt.close(fig)
    lines.insert(8,'\n![冻结策略与价值交叉干预](crossover.png)\n')
if len(conclusions)==2:
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,task in zip(axes,['pendulum','vehicle']):
        rs=[r for r in results if r['task']==task and r['value']]
        fs=sorted([r for r in rs if r['horizon'] is not None],key=lambda r:r['horizon'])
        rl=[r for r in rs if r['horizon'] is None]
        if fs:ax.scatter(range(len(fs)),[r['mean_cost'] for r in fs],s=70,color='#2878B5',label='Fixed H: seed 0')
        if rl:
            x=len(fs)
            ax.scatter(np.linspace(x-.1,x+.1,len(rl)),[r['mean_cost'] for r in rl],s=45,color='#C65C35',label='RL: seeds 0, 1, 2')
            ax.plot([x-.2,x+.2],[np.mean([r['mean_cost'] for r in rl])]*2,color='black',lw=2,label='RL seed mean')
        ax.set(xticks=range(len(fs)+1),xticklabels=['H%d'%r['horizon'] for r in fs]+['RL'],ylabel='Mean episode total cost',title=task.capitalize()+' / 20 held-out episodes')
        vals=[r['mean_cost'] for r in rs]
        if min(vals)>0 and max(vals)/min(vals)>20:
            ax.set_yscale('log');ax.set_ylabel('Mean episode total cost (log scale)')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(dest/'holdout.png');fig.savefig(dest/'holdout.pdf');plt.close(fig)
    lines+=['','![保留测试成本](holdout.png)']
(dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(dest/'results.json').write_text(json.dumps({'crossover':cross,'diagnostics':diagnostics,'refined':results,'comparisons':conclusions},indent=2)+'\n')
print(dest/'report.md')
