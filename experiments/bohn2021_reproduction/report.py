"""Generate source-grounded scene previews and partial/final reproduction results."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17'


def read(p):return json.loads(p.read_text())


def scenes(out):
    pcase=read(ART/'configs/pendulum_test_bank.json')['cases'][0]
    vcase=read(ART/'configs/vehicle_test_bank.json')['cases'][0]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    ax=axs[0];s=pcase['state'];x=s['pos'];theta=s['theta'];length=.25
    ax.plot([-1.5,1.5],[0,0],color='#64748b',lw=3)
    ax.add_patch(Rectangle((x-.11,0),.22,.09,facecolor='#2563eb'))
    tip=[x+length*np.sin(theta),.09+length*np.cos(theta)]
    ax.plot([x,tip[0]],[.09,tip[1]],color='#d97706',lw=4)
    ax.scatter(*tip,s=80,color='#d97706',zorder=4)
    ax.axvline(pcase['tvp']['pos_r'][0]['true'][0],ls='--',color='#16a34a',label='Initial position reference')
    ax.set(xlim=(-1.6,1.6),ylim=(-.08,.55),xlabel='Cart position (m)',ylabel='Height (m)',title='Cart-pendulum: stabilization + tracking')
    ax.text(.02,.92,'dt = 0.04 s | max 100 steps\nm = 0.2 kg | M = 0.8 kg | l = 0.25 m',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(loc='lower right',fontsize=8)
    ax=axs[1]
    ns=vcase['reference']['traj_steps'];tvp=vcase['tvp']
    tx=[r['true'][0] for r in tvp['trajectory_x'][:ns]];ty=[r['true'][0] for r in tvp['trajectory_y'][:ns]]
    ax.plot(tx,ty,'--',color='#d97706',label='Reference')
    for j in range(3):
        ox,oy,rad=[tvp['obj_%d_%s'%(j,n)][0]['true'][0] for n in ['x','y','r']]
        ax.add_patch(Circle((ox,oy),rad,facecolor='#64748b',alpha=.6))
        ax.add_patch(Circle((ox,oy),1.5*rad,fill=False,ls=':',edgecolor='#64748b'))
    ax.scatter(tx[0],ty[0],color='#16a34a',s=50,label='Start')
    ax.scatter(tx[-1],ty[-1],color='#dc2626',s=50,label='Goal')
    ax.set(xlabel='x (m)',ylabel='y (m)',title='Vehicle: random obstacles + uncertain forecasts')
    ax.axis('equal');ax.legend(fontsize=8)
    fig.suptitle('Reconstructed experiments — original test set unavailable',fontsize=12)
    fig.savefig(out/'scenes.png',dpi=180)
    fig.savefig(out/'scenes.pdf')
    plt.close(fig)


def comparisons(rows,out):
    if not rows:return
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ti,task in enumerate(['pendulum','vehicle']):
        for col,(field,title) in enumerate([('cost','Total cost'),('performance','Performance cost'),('constraint_episodes','Constraint terminations / 10')]):
            ax=axs[ti,col]
            for terminal,color,label in [(True,'#2563eb','With terminal value'),(False,'#d97706','Without terminal value')]:
                selected=[r for r in rows if r['task']==task and r['terminal']==terminal]
                fixed=sorted([r for r in selected if r['method']!='RL'],key=lambda r:int(r['method'][1:]))
                if fixed:ax.plot([int(r['method'][1:]) for r in fixed],[r[field] for r in fixed],'o-',color=color,label='Fixed H: '+label)
                rl=[r[field] for r in selected if r['method']=='RL']
                if rl:
                    ax.axhline(np.mean(rl),ls='--',color=color,label='RL: '+label)
                    ax.axhspan(min(rl),max(rl),color=color,alpha=.10)
            ax.set(xlabel='Fixed horizon H',ylabel=title,title=task.capitalize()+' — '+title,xlim=(3,52))
            if field=='constraint_episodes':ax.set_ylim(-.3,10.3)
            ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Reconstructed benchmark | RL shading: min–max across completed training seeds (not CI)',fontsize=11)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)


def trajectories(out):
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ti,task in enumerate(['pendulum','vehicle']):
        case=read(ART/'configs'/(task+'_test_bank.json'))['cases'][0]
        ax=axs[ti,0]
        if task=='vehicle':
            tvp=case['tvp'];ns=case['reference']['traj_steps']
            ax.plot([v['true'][0] for v in tvp['trajectory_x'][:ns]],[v['true'][0] for v in tvp['trajectory_y'][:ns]],'--',color='#64748b',label='Reference')
            for i in range(3):
                x,y,r=[tvp['obj_%d_%s'%(i,n)][0]['true'][0] for n in ['x','y','r']]
                ax.add_patch(Circle((x,y),r,color='#64748b',alpha=.4))
            ax.set(xlabel='x (m)',ylabel='y (m)');ax.axis('equal')
        else:
            ax.plot(np.arange(1,101)*.04,[v['true'][0] for v in case['tvp']['pos_r'][1:101]],'--',color='#64748b',label='Position reference')
            ax.set(xlabel='Time since reset (s)',ylabel='Cart position (m)')
        for method,color,label in [('rl_s0','#2563eb','RL seed 0'),('fixed_h25','#d97706','Fixed H=25')]:
            trace=ART/'results/full'/(task+'_'+method)/'eval_value/trace_00.json'
            if not trace.exists():continue
            t=read(trace);time=np.arange(1,len(t)+1)*(.04 if task=='pendulum' else .1)
            if task=='pendulum':ax.plot(time,[v['state']['pos'] for v in t],color=color,label=label)
            else:ax.plot([v['state']['x'] for v in t],[v['state']['y'] for v in t],color=color,label=label)
            axs[ti,1].step(time,[v['horizon'] for v in t],where='post',label=label,color=color)
        ax.set_title(task.capitalize()+' — frozen test case 0');ax.legend(fontsize=8)
        axs[ti,1].set(xlabel='Time since reset (s)',ylabel='Selected H',ylim=(0,52),title='Horizon over time')
    fig.suptitle('Preselected illustration: test case 0, RL seed 0 and fixed H=25; terminal value enabled',fontsize=10)
    fig.savefig(out/'trajectories.png',dpi=180);fig.savefig(out/'trajectories.pdf');plt.close(fig)


def conclusions(rows):
    lines=['','## 完整配对比较','']
    for task in ['pendulum','vehicle']:
        selection=[r for r in rows if r['task']==task and r['terminal']]
        fixed=[r for r in selection if r['method']!='RL'];rl=[r for r in selection if r['method']=='RL']
        if len(fixed)!=10 or len(rl)!=3:
            lines.append('- %s：完整比较尚未齐备（固定H %d/10，RL种子 %d/3）。'%(task,len(fixed),len(rl)))
            continue
        best=min(fixed,key=lambda r:r['cost']);costs=[r['cost'] for r in rl];mean=float(np.mean(costs))
        delta=best['cost']-mean
        relative=('%.2f%%'%(100*delta/abs(best['cost']))) if abs(best['cost'])>1e-12 else '不适用'
        lines.append('- %s：RL三种子平均成本 %.3f（种子范围 %.3f–%.3f）；测试集上最低固定成本为%s的 %.3f。RL相对该固定值的成本降低率 %s，负值表示RL更差。'%(task,mean,min(costs),max(costs),best['method'],best['cost'],relative))
    if len(rows)==52:
        lines+=['','**本次重建实验没有重现RL优于最佳固定H的性能结论。** 车辆学习曲线显示中后期退化；固定H=15可在10/10场景到达终点。原因尚未被隔离，不将配置补设、归一化、自举边界或算法实现差异中的任一项直接宣称为根因。']
    lines += ['','每个种子先在相同10个场景上求平均，再汇总3个RL种子。固定H仅1个训练种子，阴影是RL种子最小至最大值，不是置信区间。',
              '最低固定成本取自同一测试集，仅作描述性比较；没有据此重新训练或调参。10个测试场景与原论文不同，这些结果不能验证或否定原论文具体的4%/8%数值。']
    return lines


def paper_figures(rows,out):
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,task in zip(axs,['pendulum','vehicle']):
        selected=[r for r in rows if r['task']==task and r['terminal']]
        grouped={}
        for r in selected:grouped.setdefault(r['method'],[]).append(r)
        names=sorted(grouped,key=lambda n:np.mean([r['cost'] for r in grouped[n]]))
        bottom=np.zeros(len(names));y=np.arange(len(names))
        for field,color,label in [('performance','#2563eb','Performance'),('computation','#16a34a','Horizon proxy'),('constraint_cost','#d97706','Constraint')]:
            vals=np.array([np.mean([r[field] for r in grouped[n]]) for n in names])
            ax.barh(y,vals,left=bottom,label=label,color=color);bottom+=vals
        ax.set_yticks(y);ax.set_yticklabels(names);ax.invert_yaxis();ax.set(xlabel='Mean episode cost',title=task.capitalize())
    axs[0].legend(fontsize=8);fig.suptitle('Cost decomposition | learned terminal value | reconstructed benchmark')
    fig.savefig(out/'cost_components.png',dpi=180);fig.savefig(out/'cost_components.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,task in zip(axs,['pendulum','vehicle']):
        for seed,color in enumerate(['#2563eb','#d97706','#16a34a']):
            folder=ART/'results/full'/(task+'_rl_s'+str(seed))
            if not (folder/'completed.json').exists():continue
            horizons=[]
            for p in sorted((folder/'eval_value').glob('trace_*.json')):horizons.extend([t['horizon'] for t in read(p)])
            density=np.bincount(horizons,minlength=51)[1:51]/len(horizons)
            ax.step(np.arange(1,51),density,where='mid',color=color,label='Seed %d'%seed)
        ax.set(xlabel='Horizon H',ylabel='Fraction of recorded control steps',title=task.capitalize(),xlim=(.5,50.5))
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('RL horizon distribution on 10 frozen test cases, terminal value enabled')
    fig.savefig(out/'horizon_distribution.png',dpi=180);fig.savefig(out/'horizon_distribution.pdf');plt.close(fig)
    curve_rows=[]
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,task in zip(axs,['pendulum','vehicle']):
        point_values={}
        for seed in range(3):
            folder=ART/'results/full'/(task+'_rl_s'+str(seed));xs=[];ys=[]
            for step in range(2500,15001,2500):
                ev=folder/('eval_value' if step==15000 else 'learning_curve/'+str(step))
                completion=folder/'completed.json' if step==15000 else ev/'completed.json'
                if not completion.exists():continue
                cost=read(ev/'summary.json')['mean_total_cost'];xs.append(step);ys.append(cost)
                point_values.setdefault(step,[]).append(cost)
                curve_rows.append({'task':task,'seed':seed,'nominal_steps':step,'mean_test_cost':cost})
            if xs:ax.plot(xs,ys,alpha=.3,lw=1,label='Seed %d'%seed)
        xs=sorted([s for s in point_values if len(point_values[s])==3])
        if xs:
            means=np.array([np.mean(point_values[s]) for s in xs]);std=np.array([np.std(point_values[s],ddof=1) for s in xs])
            ax.plot(xs,means,'o-',color='#2563eb',label='3-seed mean');ax.fill_between(xs,means-std,means+std,color='#2563eb',alpha=.15)
        ax.set(xlabel='Nominal training transitions',ylabel='Mean frozen-test episode cost',title=task.capitalize())
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('Checkpoint evaluation | shading: ±1 sample SD across 3 seeds | no further training',fontsize=11)
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    (out/'learning_curves.json').write_text(json.dumps(curve_rows,indent=2)+'\n')
    ablation=[]
    fig,axs=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,task in zip(axs,['pendulum','vehicle']):
        methods=['H%d'%h for h in range(5,51,5)]+['RL'];names=[];improvements=[]
        for method in methods:
            subset=[r for r in rows if r['task']==task and r['method']==method]
            on=[r['cost'] for r in subset if r['terminal']];off=[r['cost'] for r in subset if not r['terminal']]
            if not on or len(on)!=len(off):continue
            baseline=float(np.mean(off));value=float(np.mean(on))
            pct=100*(baseline-value)/abs(baseline) if abs(baseline)>1e-12 else None
            if pct is not None:names.append(method);improvements.append(pct)
            ablation.append({'task':task,'method':method,'with_value_cost':value,'without_value_cost':baseline,
                             'cost_reduction_pct':pct,'training_seeds':len(on)})
        y=np.arange(len(names));ax.barh(y,improvements,color=['#2563eb' if v>=0 else '#d97706' for v in improvements])
        ax.set_yticks(y);ax.set_yticklabels(names);ax.invert_yaxis();ax.axvline(0,color='#64748b',lw=.8)
        ax.set(xlabel='Cost reduction with terminal value (%)',title=task.capitalize())
    fig.suptitle('Terminal-value removal on the same trained controllers | negative means adding value worsens cost',fontsize=10)
    fig.savefig(out/'terminal_ablation.png',dpi=180);fig.savefig(out/'terminal_ablation.pdf');plt.close(fig)
    (out/'terminal_ablation.json').write_text(json.dumps(ablation,indent=2)+'\n')


def main():
    out=ART/'report';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    scenes(out)
    rows=[]
    full=ART/'results/full'
    if full.exists():
        for p in sorted(full.glob('*/completed.json')):
            folder=p.parent;m=read(folder/'manifest.json')
            for enabled in [True,False]:
                e=read(folder/('eval_value' if enabled else 'eval_no_value')/'summary.json')
                eps=e['episodes']
                rows.append({'task':m['task'],'method':'RL' if m['fixed_horizon'] is None else 'H%d'%m['fixed_horizon'],
                    'seed':m['seed'],'terminal':enabled,'cost':e['mean_total_cost'],
                    'performance':float(np.mean([r['performance_cost'] for r in eps])),
                    'computation':float(np.mean([r['computation_cost'] for r in eps])),
                    'constraint_cost':float(np.mean([r['constraint_cost'] for r in eps])),
                    'constraint_episodes':e['constraint_episodes'],'goals':e['goal_episodes'],
                    'mean_h':float(np.mean([r['mean_horizon'] for r in eps])),
                    'solver_failure_steps':sum(r['solver_failure_steps'] for r in eps)})
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    comparisons(rows,out)
    trajectories(out)
    paper_figures(rows,out)
    status=read(full/'status.json') if (full/'status.json').exists() else {}
    finished=len(list(full.glob('*/completed.json')))
    audit_path=ART/'results/physical_constraint_audit.json'
    audit=read(audit_path) if audit_path.exists() else None
    text=['# 原论文复现进展与结果','',
          '## Material Passport','',
          '依据作者2021年同期SAC、Gym和do-mpc代码；配置与测试集重建，详见[复现说明](../../../experiments/bohn2021_reproduction/README.md)。',
          '生成时间：'+datetime.now(timezone.utc).isoformat(),'',
          '**完成 %d/26 组；失败 %d 组。**'%(finished,len(status.get('failed',[]))),
          '这里恢复作者核心算法并重建场景，原配置、测试集和模型未找回，不能称精确数值复现。','',
          '![重建场景](scenes.png)','',
          '|任务|方法|种子|终端值|平均总成本|因约束终止/10|到达终点/10|平均H|求解失败步|',
          '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append('|%s|%s|%d|%s|%.3f|%d|%s|%.2f|%d|'%(r['task'],r['method'],r['seed'],'有' if r['terminal'] else '无',r['cost'],r['constraint_episodes'],str(r['goals']) if r['task']=='vehicle' else '不适用',r['mean_h'],r['solver_failure_steps']))
    text+=['','倒立摆到达100步只表示未提前违反约束，不自动证明稳定性。车辆达到步数上限也不等于成功。',
           '表中平均H先按回合求均值再平均；分布图按实际控制步统计。所有成本均为完整回合总和再取测试均值。',
           '每个固定H具有独立训练的终端估计器。无终端值结果是对同一已训练模型清零终端值的干预。',
           '并行运行中的实测耗时不作为跨方法计算效率结论。','']
    if rows:text+=['![配对成本比较](comparison.png)','']
    text+=conclusions(rows)+['','![预选场景轨迹](trajectories.png)','']
    if rows:text+=['![成本分解](cost_components.png)','',
        '![时域分布](horizon_distribution.png)','',
        '![终端价值消融](terminal_ablation.png)','',
        '![学习曲线](learning_curves.png)','',
        '学习曲线来自训练完成后对已存检查点的冻结评估；阴影为3个种子的样本标准差。中途检查点保存于当步更新前，末点使用完成全部更新的最终模型。未使用测试曲线选择检查点或调整超参数。','']
    if audit:
        text+=['## 独立轨迹核验','',
            '已检查%d个最终测试回合，输入超界回合%d；状态越界但未按约束终止的回合%d（容差1e-6）。这是物理轨迹检查，不是稳定性证明。'%(len(audit['rows']),audit['episodes_with_input_violation'],len(audit['state_violations_without_constraint_termination'])),'']
        for row in audit['state_violations_without_constraint_termination']:
            text.append('- %s / %s / 场景%d：第%s步状态越界，环境标签为%s。原倒立摆环境先检查时间上限，末步可能漏标；主表保留标签，实际越界在此披露。'%(row['run'],row['mode'],row['case'],','.join(map(str,row['physical_state_violation_steps'])),row['termination']))
    text+=['## 场景回放','',
        '[倒立摆GIF](pendulum_replay.gif) · [车辆GIF](vehicle_replay.gif)。两者均预选种子0、测试场景0，可能包含失败，不能代表全部场景。','']
    if finished==26:
        text+=['## 执行与保真边界','',
            '26组共390,000条训练经验、387,426次梯度更新；520个最终测试回合＋300个检查点测试回合。模型权重在评估期间冻结。',
            '保存模型回放、终端值TensorFlow/CasADi/NumPy数值一致性、成本重算、配置/测试集散列均通过核验。作者源码工作树未修改，修正在独立适配层。',
            '原始实验配置、测试集、训练入口与SAC模型没有找回；测试场景为重建且仅10个，固定H仅一个训练种子。重建配置和已披露语义修正使本结果不能称逐数值精确复现。',
            '原AHMPC用50步NLP屏蔽后续阶段，H成本是代理量。并行执行记录的耗时不用于证明计算优势。','']
    (out/'progress.md').write_text('\n'.join(text))
    if finished==26:
        (out/'final_report.md').write_text('\n'.join(text).replace('# 原论文复现进展与结果','# 原论文复现实验报告',1))
    print(json.dumps({'completed':finished,'rows':len(rows),'report':str(out/'progress.md')}))


if __name__=='__main__':main()
