"""Independent physical/return audit and descriptive mechanism summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from sac_teacher_cause import OUT,STUDY,ROOT,GRID,read,write,verify,jobs,finals,summary,sha
from sac_teacher_report import audit_row


def phase_analysis():
    result={}
    bank=read(STUDY/'holdout_bank.json')['scenes']
    for name in ['plain_s%d_21000'%s for s in range(3)]+['teacher_s%d_15000'%s for s in range(3)]+['switch_5_30']:
        groups={k:[] for k in ['initial','after_first','after_second']}
        for p in (STUDY/'evaluation/holdout'/name).glob('scene_*.json'):
            r=read(p);switches=bank[r['summary']['scene']]['switches']
            for row in r['trace']:
                key=['initial','after_first','after_second'][sum(row['step']+1>=t for t in switches)]
                groups[key].append(row)
        result[name]={key:{'adjusted_cost_per_episode':sum(r['cost']+.4905 for r in part)/12,
            'mean_abs_error':float(np.mean([abs(r['state']['pos']-r['next_obs'][4]) for r in part])),
            'short_H_fraction':float(np.mean([r['horizon']<=3 for r in part])),
            'solver_failures':sum(not r['solver_success'] for r in part),'steps':len(part)} for key,part in groups.items()}
    return result


def training_dynamics():
    result={}
    for folder in sorted((STUDY/'models').iterdir()):
        failures=[]
        with (folder/'transitions.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['termination']=='constraint':
                    failures.append({'phase':row['phase'],'step':row['step'],
                        'cost':row['cost'],'scaled_reward':-row['cost']/.6})
        losses=read(folder/'losses.json')
        peak=max(losses,key=lambda r:r['values'][1])
        result[folder.name]={'physical_failures':failures,
            'max_logged_Q1_loss':peak['values'][1],'peak_update':peak['update'],
            'loss_logging_interval':100}
    return result


def switch_windows(folder,bank):
    result={}
    for condition in sorted(folder.iterdir()):
        if not condition.is_dir():continue
        windows={key:[] for key in ['before30','after30']}
        for path in sorted(condition.glob('scene_*.json')):
            episode=read(path);scene=bank[episode['summary']['scene']]
            for switch in scene['switches']:
                for key,lo,hi in [('before30',-30,0),('after30',0,30)]:
                    rows=[r for r in episode['trace'] if switch+lo<=r['step']+1<switch+hi]
                    if rows:windows[key].append(rows)
        result[condition.name]={key:{'observed_windows':len(parts),
            'complete_windows':sum(len(part)==30 for part in parts),
            'mean_H':float(np.mean([r['horizon'] for part in parts for r in part])),
            'mean_abs_error':float(np.mean([abs(r['state']['pos']-r['next_obs'][4]) for part in parts for r in part])),
            'mean_observed_window_cost':float(np.mean([sum(r['cost']+.4905 for r in part) for part in parts])),
            'solver_failures':sum(not r['solver_success'] for part in parts for r in part)}
            for key,parts in windows.items() if parts}
    return result


def analyze():
    groups={}
    for p in sorted((OUT/'evaluation').glob('*/completed.json')):
        rows=read(p)['episodes']
        groups[p.parent.name]={'cost':float(np.mean([r['adjusted_cost'] for r in rows])),
            'physical':float(np.mean([r['physical_cost'] for r in rows])),
            'H_cost':float(np.mean([r['H_cost'] for r in rows])),
            'failures':sum(r['termination']=='constraint' for r in rows),
            'solver_failures':sum(r['solver_failures'] for r in rows),'steps':sum(r['steps'] for r in rows),
            'scenes':[r['adjusted_cost'] for r in rows]}
    contrasts=[]
    for name in finals():
        if name+'__det' not in groups:continue
        base=np.asarray(groups[name+'__det']['scenes'])
        for mode in ['stoch0','stoch1','min5','q1']:
            if name+'__'+mode not in groups:continue
            other=np.asarray(groups[name+'__'+mode]['scenes'])
            contrasts.append({'name':name,'mode':mode,'cost_delta':float(np.mean(other-base)),
                'scene_deltas':(other-base).tolist(),'wins':int(sum(other<base))})
    branch_results=[]
    for folder in sorted((OUT/'branches').glob('*')):
        if not (folder/'completed.json').exists():continue
        rows=read(folder/'completed.json')['branches']
        for anchor in sorted({r['anchor'] for r in rows}):
            block={source:[r for r in rows if r['anchor']==anchor and r['candidate']==source] for source in ['actor','q1','rule']}
            record={'name':folder.name,'anchor':anchor,'candidates':{},'contrasts':{}}
            for source,part in block.items():
                prefix_returns=[]
                for branch in part:
                    trace=read(folder/('t%03d_%s_r%d.json'%(anchor,source,branch['replica'])))['trace']
                    prefix_returns.append(sum(.97**t*(-r['cost']/.6-(.01*r['decision']['logp'] if t else 0.)) for t,r in enumerate(trace[:50])))
                record['candidates'][source]={'H':part[0]['first_H'],'action':part[0]['first_action'],
                    'predicted_Q1':part[0]['predicted_Q1'],'predicted_Q2':part[0]['predicted_Q2'],
                    'soft_returns':[r['soft_return'] for r in part],
                    'first50_soft_returns':prefix_returns,
                    'mean_soft_return':float(np.mean([r['soft_return'] for r in part])),
                    'failures':sum(r['termination']=='constraint' for r in part)}
            actor=record['candidates']['actor']
            for source in ['q1','rule']:
                c=record['candidates'][source]
                delta=np.asarray(c['soft_returns'])-actor['soft_returns']
                predicted=c['predicted_Q1']-actor['predicted_Q1']
                record['contrasts'][source]={'predicted_advantage':predicted,'MC_advantage':float(delta.mean()),
                    'replica_deltas':delta.tolist(),'absolute_advantage_error':float(abs(predicted-delta.mean())),
                    'first50_replica_deltas':(np.asarray(c['first50_soft_returns'])-actor['first50_soft_returns']).tolist(),
                    'both_replicas_opposite_prediction':bool((predicted>0 and np.all(delta<0)) or (predicted<0 and np.all(delta>0)))}
            branch_results.append(record)
    result={'groups':groups,'interventions':contrasts,'branches':branch_results,'historical_phase':phase_analysis(),
        'training_dynamics':training_dynamics(),
        'historical_switch_windows':switch_windows(STUDY/'evaluation/holdout',read(STUDY/'holdout_bank.json')['scenes']),
        'diagnostic_switch_windows':switch_windows(OUT/'evaluation',read(OUT/'bank.json')['scenes'])}
    write(OUT/'analysis.json',result)
    return result


def audit():
    verify();assert (OUT/'completed.json').exists()
    sensitivity=read(OUT/'preview_sensitivity.json')
    assert sensitivity['physical_steps']==sensitivity['SAC_updates']==0
    assert sensitivity['source_sha256']==sha(Path(__file__).with_name('sac_teacher_preview_sensitivity.py'))
    assert len(sensitivity['samples'])==24 and len(sensitivity['models'])==15
    for sample in sensitivity['samples']:
        original=np.asarray(sample['obs']);held=np.asarray(sample['held_preview_obs'])
        row=read(OUT/'evaluation/rule__det'/('scene_%02d.json'%sample['scene']))['trace'][sample['switch']-sample['lead']-1]
        np.testing.assert_array_equal(original,row['obs'])
        np.testing.assert_array_equal(original[:5],held[:5]);assert original[55]==held[55]
        np.testing.assert_allclose(held[5:55],(original[4]-original[0]*1.5)/1.5,atol=1e-7)
    bank=read(OUT/'bank.json')['scenes'];initials={};steps=episodes=0
    expected={args[1]+'__'+args[3] for args in jobs()}
    assert {p.name for p in (OUT/'evaluation').iterdir() if p.is_dir()}==expected
    for condition in sorted(expected):
        folder=OUT/'evaluation'/condition
        completed=read(folder/'completed.json')['episodes']
        assert [r['scene'] for r in completed]==[0,1,2,3]
        for scene in bank:
            result=read(folder/('scene_%02d.json'%scene['id']));trace=result['trace']
            assert result['summary']==completed[scene['id']]
            if scene['id'] in initials:np.testing.assert_array_equal(result['initial'],initials[scene['id']])
            initials[scene['id']]=result['initial'];previous=None
            for i,row in enumerate(trace):
                assert row['step']==i+1
                audit_row(row,scene,i+1,previous);previous=row
            assert trace[-1]['done']
            calculated=summary(trace)
            for key,value in calculated.items():
                if isinstance(value,str):assert result['summary'][key]==value
                else:np.testing.assert_allclose(result['summary'][key],value)
            steps+=len(trace);episodes+=1
    branch_steps=prefix_steps=branch_resets=0
    reference=read(OUT/'evaluation/rule__det/scene_00.json')['trace']
    for name in finals():
        folder=OUT/'branches'/name
        completed=read(folder/'completed.json')['branches']
        assert len(completed)==24
        assert len({(r['anchor'],r['candidate'],r['replica']) for r in completed})==24
        for s in completed:
            result=read(folder/('t%03d_%s_r%d.json'%(s['anchor'],s['candidate'],s['replica'])))
            assert s==result['summary'];trace=result['trace'];previous=None
            np.testing.assert_array_equal(result['initial'],reference[s['anchor']]['obs'])
            for i,row in enumerate(trace):
                audit_row(row,bank[0],s['anchor']+i+1,previous);previous=row
                if i:assert np.isfinite(row['decision']['logp'])
            assert trace[-1]['done']
            soft=sum(.97**i*(-r['cost']/.6-(.01*r['decision']['logp'] if i else 0.)) for i,r in enumerate(trace))
            np.testing.assert_allclose(soft,s['soft_return'],atol=1e-9)
            assert len(trace)==s['branch_steps']
            branch_steps+=len(trace);prefix_steps+=s['prefix_steps'];branch_resets+=1
    result={'passed':True,'evaluation_episodes':episodes,'evaluation_steps':steps,
        'branch_rollouts':branch_resets,'branch_steps':branch_steps,'prefix_steps':prefix_steps,
        'reset_warmups':episodes+branch_resets,'smoke_physical_steps':read(OUT/'smoke.json')['physical_steps'],
        'SAC_updates':0,'source_sha256':sha(Path(__file__))}
    result['total_physical_steps']=steps+branch_steps+prefix_steps+episodes+branch_resets+result['smoke_physical_steps']
    write(OUT/'audit.json',result);return result


def plots(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    colors=['#0072B2','#D55E00','#009E73']
    modes=['det','stoch','min5','q1']
    for seed,color in enumerate(colors):
        name='teacher_s%d_step_15000'%seed
        values=[]
        for mode in modes:
            values.append(np.mean([a['groups'][name+'__'+m]['cost'] for m in (['stoch0','stoch1'] if mode=='stoch' else [mode])]))
        axes[0,0].scatter(np.arange(4)+(seed-1)*.09,values,color=color,label='Seed %d'%seed)
    axes[0,0].axhline(a['groups']['rule__det']['cost'],ls='--',color='#555555',label='Rule')
    axes[0,0].set(xticks=range(4),xticklabels=['Actor mean','Stochastic','H >= 5','Q1 grid'],ylabel='Mean episode cost',title='A  Frozen teacher-policy interventions')
    axes[0,0].legend(fontsize=8)
    for seed,color in enumerate(colors):
        values=[a['groups']['teacher_s%d_%s__det'%(seed,label)]['cost'] for label in ['pretrained','step_05000','step_10000','step_15000']]
        axes[0,1].plot([0,5,10,15],values,'o-',color=color,label='Seed %d'%seed)
    axes[0,1].set(xlabel='Online steps (thousands) after pretraining',ylabel='Mean episode cost',title='B  Timing of degradation; all checkpoints')
    for seed,color in enumerate(colors):
        rows=[r for r in a['branches'] if r['name']=='teacher_s%d_step_15000'%seed]
        for r in rows:
            c=r['contrasts']['rule'];x=c['predicted_advantage'];ys=c['replica_deltas']
            axes[1,0].plot([x,x],[min(ys),max(ys)],color=color,alpha=.6)
            axes[1,0].scatter(x,c['MC_advantage'],color=color,s=35)
    axes[1,0].axhline(0,color='#777777',lw=.8);axes[1,0].axvline(0,color='#777777',lw=.8)
    axes[1,0].set(xlabel='Predicted Q1(rule) - Q1(actor)',ylabel='Measured soft return difference',title='C  Same-state action advantages; 2 MC draws')
    phases=['initial','after_first','after_second'];width=.25
    for i,group in enumerate(['plain','teacher','rule']):
        names=['%s_s%d_%d'%(group,s,21000 if group=='plain' else 15000) for s in range(3)] if group!='rule' else ['switch_5_30']
        values=[np.mean([a['historical_phase'][name][phase]['adjusted_cost_per_episode'] for name in names]) for phase in phases]
        axes[1,1].bar(np.arange(3)+(i-1)*width,values,width,color=['#0072B2','#D55E00','#009E73'][i],label=group)
    axes[1,1].set(xticks=range(3),xticklabels=['Initial plateau','After switch 1','After switch 2'],ylabel='Cost per historical holdout episode',title='D  Retrospective location of excess cost')
    axes[1,1].legend(fontsize=8)
    dest=OUT.parents[1]/'report/sac_teacher_cause';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'diagnosis.png',dpi=180);fig.savefig(dest/'diagnosis.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    scene=read(OUT/'bank.json')['scenes'][0];switch=scene['switches'][0]
    for name,label,color in [('rule__det','Rule','#009E73'),
                            ('teacher_s2_step_15000__det','Teacher seed 2','#D55E00'),
                            ('teacher_s2_step_15000__min5','Teacher seed 2, H >= 5','#0072B2')]:
        rows=[r for r in read(OUT/'evaluation'/name/'scene_00.json')['trace'] if switch-65<=r['step']+1<=switch+90]
        x=[r['step']+1-switch for r in rows]
        axes[0].plot(x,[r['state']['pos'] for r in rows],color=color,label=label)
        axes[1].step(x,[r['horizon'] for r in rows],where='post',color=color,label=label)
    axes[0].step(x,[r['next_obs'][4] for r in rows],where='post',color='#555555',ls='--',label='Reference')
    for ax in axes:ax.axvline(0,color='#777777',lw=.8)
    axes[0].legend(fontsize=8,loc='upper left');axes[1].legend(fontsize=8,loc='upper right')
    axes[0].set(ylabel='Position (m)',title='Diagnostic scene 0, first switch: late horizon expansion')
    axes[1].set(xlabel='Reward reference clock relative to switch (steps)',ylabel='Executed H',ylim=(0,52))
    fig.savefig(dest/'switch_trace.png',dpi=180);fig.savefig(dest/'switch_trace.pdf');plt.close(fig)


def report(a,budget):
    lines=['# 教师回放 SAC 退化的机制诊断','',
        '这是前一轮结果之后登记的冻结模型干预，不是重新训练或新的性能优越性检验。使用4个全新诊断场景、全部3个训练种子；随机策略使用两个公共随机数重复。原训练、reward、终端及测试集保持冻结。','',
        '完整回合成本沿用原报告：原始总成本+294.3，越低越好；提前终止仍保留原失败罚项，平移常数也使未执行步的基准得到补齐。新4场景与原12场景分别报告，不能跨测试集比较绝对数值。soft-return使用原缩放奖励和熵项，数值与此成本不是同一尺度。','',
        '## 已定位的行为问题','',
        '教师策略在参考即将反转时没有保留教师的提前响应：切换前选择短H、维持当前位置，切换后才加长H追赶。旧12场景中，三个教师种子在每次切换前30步的平均H分别只有6.36、6.81、6.96；切换后30步平均绝对跟踪误差分别为0.616、0.609、0.638 m。表现较好的普通seed1切换前H为36.87，切换后误差为0.169 m。这里不是H越大越好，而是该在何时增加预测长度的问题。','',
        '旧holdout教师组相对普通组物理成本增加188.95，H代理成本只节省8.27，净成本增加180.68。额外损失集中在参考切换以后；教师组初始平台反而更好。这排除了“仅仅算力代理罚项增加”的解释。','',
        '|旧holdout模型|切换前30步H|切换后30步H|切换后平均绝对误差/m|切换后30步平均成本|','|---|---:|---:|---:|---:|']
    names=['plain_s%d_21000'%s for s in range(3)]+['teacher_s%d_15000'%s for s in range(3)]+['switch_5_30']
    for name in names:
        w=a['historical_switch_windows'][name];before=w['before30'];after=w['after30']
        assert before['complete_windows']==after['complete_windows']==24
        lines.append('|%s|%.3f|%.3f|%.4f|%.4f|'%(name,before['mean_H'],after['mean_H'],after['mean_abs_error'],after['mean_observed_window_cost']))
    lines+=['','窗口按reward参考时钟定义，每模型24次参考变化，全部窗口完整。窗口为事后描述性分析，不用于训练或选模型。','',
        '![参考变化轨迹](switch_trace.png)','',
        '上图展示新诊断场景0的seed2。规则在变化前提前移动；actor在变化前仍停在旧目标附近，之后才选择长H。H至少5的干预消除了该轨迹的求解失败，但仍明显迟到。此例用于展示机制，全种子与全场景统计如下。','',
        '## 完整策略干预','',
        '|模型|原actor|随机执行均值|H至少5|Q1网格贪心|','|---|---:|---:|---:|---:|']
    for name in finals():
        g=a['groups'];values=[g[name+'__det']['cost'],np.mean([g[name+'__stoch%d'%i]['cost'] for i in range(2)]),g[name+'__min5']['cost'],g[name+'__q1']['cost']]
        lines.append('|%s|%s|'%(name,'|'.join('%.4f'%v for v in values)))
    lines+=['','固定H25成本%.4f，H5/H30规则%.4f。'%(a['groups']['fixed25__det']['cost'],a['groups']['rule__det']['cost']),
        '', '|教师种子|预训练结束|在线5k|在线10k|在线15k|','|---|---:|---:|---:|---:|']
    for seed in range(3):
        values=[a['groups']['teacher_s%d_%s__det'%(seed,label)]['cost'] for label in ['pretrained','step_05000','step_10000','step_15000']]
        lines.append('|%d|%s|'%(seed,'|'.join('%.4f'%v for v in values)))
    lines+=['','解释：seed0/1预训练后成本分别170.67/146.07，在线训练后明显变差；seed2预训练已经有1/4物理失败（成本486.52），5k时改善到164.00，随后再次退化。在线退化和不稳定得到支持，但“预训练阶段全部可靠”不成立，也不是随更新次数单调恶化。',
        '', 'H至少5将三个最终教师模型的求解失败全部降为0，并消除本诊断集的物理终止，但成本仍450.29/452.37/462.32，约为规则的3.58至3.67倍。短H可靠性问题真实存在，却不是高成本的充分解释。Q1网格贪心仍为499.52/553.57/556.92；它救回部分失败回合，但不能恢复教师规则的提前响应。随机执行有时避免终止，也未恢复规则成本水平，因此确定性均值执行不是唯一问题。']
    sensitivity=read(OUT/'preview_sensitivity.json')
    lines+=['','## 同状态预览输入干预','',
        '此项是看到轨迹之后新增的只读网络诊断，独立标为事后分析。使用4个新场景、每场景2次变化前40/25/10步的共同规则状态，共24个状态；保持物理状态、当前参考和剩余时间完全相同，仅把未来50步参考替换为当前参考的常值。无环境推进、无网络更新。它识别网络输出对预览的响应，不等同于改变预览后的实际闭环成本。','',
        '|模型|提前25步：真实预览H|提前25步：恒定预览H|提前10步：真实预览H|提前10步：恒定预览H|','|---|---:|---:|---:|---:|']
    for r in sensitivity['models']:
        if r['name'].startswith('teacher') and not r['name'].endswith(('pretrained','15000')):continue
        x=r['by_lead'];lines.append('|%s|%.3f|%.3f|%.3f|%.3f|'%(r['name'],x['25']['actual_preview_mean_H'],x['25']['held_preview_mean_H'],x['10']['actual_preview_mean_H'],x['10']['held_preview_mean_H']))
    lines+=['','每格是8个相同物理状态的均值。预训练结束时，三个教师种子都能在25步提前量下对真实变化预览明显加长H；最终seed0/1在该提前量下响应方向反转，seed2在10步提前量下也反转（真实预览H=1，恒定预览H=14.125）。因此问题不只是接口没有提供预览，而是在线学习后对预览的动作映射退化。并非每个提前量、每个种子都反转，全部40/25/10步逐状态输出保存在preview_sensitivity.json。']
    lines+=['','同样8个提前25步的共同状态上，Q1(H30)-Q1(H5)均值也从预训练的+0.034/+0.226/+0.444变成最终的-0.998/-1.465/-0.797（三个种子顺序不变）。所以不仅actor的输出改变，critic对长短H的偏好也发生了反转。这是预测值的变化；它是否违反当前策略续控下的实际回报，必须用下面的完整分支验证，不能仅因与规则不一致便判定Q错误。']
    lines+=['','## 可靠性明细','','|条件|物理失败/4回合|求解失败步/总步|','|---|---:|---:|']
    for name,g in sorted(a['groups'].items()):lines.append('|%s|%d/4|%d/%d|'%(name,g['failures'],g['solver_failures'],g['steps']))
    lines+=['','## 同状态分支与价值排序','',
        '共同状态来自新场景0的规则轨迹，已执行步数为40、首次参考切换时钟减15、加15、加70；重置预热使观测参考时钟等于已执行步数加1。分支只改变首个动作，随后由相应模型的当前随机策略执行至真实终止；没有近似尾项。Q目标包含从第二步开始的熵项，奖励仅除以0.6。两个随机重复使用公共噪声，表中正差表示规则首动作更好。','',
        '这是给定该场景未来的条件回报估计，不能作为所有未来分布下的精确soft-Q；2次重复不足以提供高精度置信区间。两次重复均与预测符号相反的案例比仅比较均值更有说服力，但仍只适用于这些状态。','',
        '|模型|时刻|actor H|Q1 H|rule H|预测rule-actor|实测rule-actor|两次实测差|','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in a['branches']:
        c=r['candidates'];d=r['contrasts']['rule']
        lines.append('|%s|%d|%d|%d|%d|%.5f|%.5f|%s|'%(r['name'],r['anchor'],c['actor']['H'],c['q1']['H'],c['rule']['H'],d['predicted_advantage'],d['MC_advantage'],', '.join('%.5f'%v for v in d['replica_deltas'])))
    lines+=['','|模型|时刻|Q1动作预测优势|Q1动作实测优势|两次实测差|','|---|---:|---:|---:|---|']
    for r in a['branches']:
        d=r['contrasts']['q1']
        lines.append('|%s|%d|%.5f|%.5f|%s|'%(r['name'],r['anchor'],d['predicted_advantage'],d['MC_advantage'],', '.join('%.5f'%v for v in d['replica_deltas'])))
    for arm in ['plain','teacher']:
        for source in ['rule','q1']:
            distinct=[r for r in a['branches'] if r['name'].startswith(arm) and r['candidates'][source]['H']!=r['candidates']['actor']['H']]
            opposite=[r for r in distinct if r['contrasts'][source]['both_replicas_opposite_prediction'] and abs(r['contrasts'][source]['predicted_advantage'])>1e-6 and min(abs(v) for v in r['contrasts'][source]['replica_deltas'])>1e-6]
            lines.append('\n%s组：%s与actor执行不同H的%d个状态中，%d个状态两次回报差均与Q1预测符号相反（排除绝对值<=1e-6的数值零）。'%(arm,source,len(distinct),len(opposite)))
    anchor=read(OUT/'bank.json')['scenes'][0]['switches'][0]-15
    example=next(r for r in a['branches'] if r['name']=='teacher_s0_step_15000' and r['anchor']==anchor)['contrasts']['rule']
    lines+=['','一个明确反例是teacher_s0切换前的共同状态：Q1预测规则首动作相对actor的优势为%.4f，两次完整soft-return差却为%s。收益方向两次都相反，而且前50步已贡献%s，说明此例的差异主要发生在近段控制，而非很远的尾项。仍须保留“一个场景、两次随机重复、条件未来”的统计限制。'%(example['predicted_advantage'],', '.join('%.4f'%v for v in example['replica_deltas']),', '.join('%.4f'%v for v in example['first50_replica_deltas']))]
    contrary=next(r for r in a['branches'] if r['name']=='teacher_s2_step_15000' and r['anchor']==anchor)['contrasts']['rule']
    lines+=['','也有相反方向的错误：同一时刻teacher_s2预测规则首动作优势为%.4f，实测两次为%s。完整规则策略表现好，不等于只插入一次规则动作、随后恢复已退化的随机策略也必然更好。因此不能把所有教师动作都当成当前策略续控下的最优动作；本轮证据支持具体价值排序不可靠，而非“每次选长H都正确”。'%(contrary['predicted_advantage'],', '.join('%.4f'%v for v in contrary['replica_deltas']))]
    profile=read(OUT/'checkpoint_profile.json')
    lines+=['','## 固定教师状态上的训练诊断','',
        '每种子从已保存教师转移每10条取1条，共600条；所有检查点使用完全相同的状态、动作和下一状态。Bellman残差只衡量自洽性，不是实际价值误差。','',
        '|模型|教师样本TD RMSE|Q1候选跨度中位数|actor比记录动作的预测优势|actor平均H|','|---|---:|---:|---:|---:|']
    for r in profile['profiles']:
        lines.append('|%s|%.4f|%.4f|%.4f|%.2f|'%(r['name'],r['TD_RMSE_teacher_rows'],r['Q1_grid_span_median'],r['actor_minus_behavior_Q1_mean'],r['mean_actor_H']))
    shock=a['training_dynamics']['teacher_s1']
    lines+=['','## 为什么加入教师回放仍会丢失教师能力','',
        '1. **这里没有直接学习教师决策。** 教师只提供(s,a,r,s\')。作者实现的Q目标是 r + gamma V_target(s\')；V跟随当前actor的min(Q1,Q2)-alpha log pi，actor则最大化Q1-alpha log pi。因此教师动作没有模仿约束，教师后续控制也没有被固定为回报标签。6000次预训练同时更新actor、Q、V和target V，之后这些目标继续移动。教师数据保留，并不等于教师策略的价值或行为被保留。',
        '2. **教师轨迹对短H的联合状态动作覆盖有限。** 三个种子的教师数据中，80.1%至81.1%的动作是H5/H30，H<5仅1.35%至1.52%；在线阶段H<5增至37.4%至53.4%。20%随机动作提供了全域的边际覆盖，但不保证每个关键参考预览状态都有足够的反事实动作比较。只凭这些比例不能证明分布外估值就是唯一根因。',
        '最终教师数据占回放28.6%，但不能把这等同于累计训练占比。在当前均匀回放和更新时刻下，教师样本在全部batch条目中的期望占比为[6000 + sum(t=100..15000) 6000/(6000+t)]/20901 = 64.20%。这是由采样机制计算的期望，不是逐batch实测比例。教师数据并非没有被训练；问题是这些转移仍接受当前策略的移动自举目标。等总预算也不等于等在线状态覆盖。',
        '3. **延迟收益的学习不稳定。** 提前移动会先增加对旧参考的误差和H费用，收益在参考变化后才出现，需要critic正确传递。原12场景中教师初始平台成本较低、全回合成本却较高，甚至其从回合起点计算的折扣原始成本略优（-14.076对-13.951）。gamma=.97使30步后权重约0.401、200步后约0.00226；这一目标与600步未折扣评价存在侧重差异。但回放也抽取后期状态，所以不能把gamma单独当作退化证明。',
        '4. **极端失败罚项可能进一步放大误差。** teacher_s1在线第13026步单步成本1748.91，缩放后奖励约-2914.85。其后已记录的Q1损失峰值为%.2f（update %d）；teacher_s0/2没有训练期物理失败也发生退化，因此这属于seed1的额外放大因素。作者Q回归使用平方损失，没有在本轮添加鲁棒损失或梯度裁剪；不能未经消融就宣称修改它必然有效。'%(shock['max_logged_Q1_loss'],shock['peak_update']),
        '5. **动作参数化增加了价值排序的难度。** actor连续输出，环境舍入到整数H；同一舍入区间内实际控制相同，但神经Q的连续插值可以有斜率。actor利用这种梯度不保证跨整数H的排序正确。这是源码可确认的结构性风险，本轮未隔离其因果贡献。',
        '', '上述链条中，“预训练之后丢失性能”“提前响应失败”“特定执行干预是否救回”有检查点和真实轨迹证据；“自举误差、覆盖不足、连续插值各贡献多少”仍是待分离的训练机制。TD残差增大支持价值学习不稳定，但不等于已测得真实Q误差。',
        '', '## 为什么上轮监督教师法能成功','',
        '上轮成功的方法在同一状态比较多个H的分支，用固定规则续控生成相对代价标签，并做学生状态补充；它直接监督动作之间的差异，教师续控和目标相对固定。本轮SAC用行为轨迹做自举，后续回报来自不断变化的当前策略，没有相同的反事实监督。两者的数据内容、目标、动作集合、特征和预算都不同，不能把本轮结果解读为“同一种教师方法多训反而必然变差”。',
        '', '## 下一步最有区分力的实验','',
        '先冻结同一个预训练起点，匹配数据和更新预算，分离critic-only更新、联合SAC更新、以及带教师动作保持约束的联合更新；在未用于本轮诊断的新场景同时检查完整成本、变化前H、变化后误差、物理/求解失败。再按需要单独检验鲁棒Q损失和整数动作建模。这里提出的是待检验设计，尚未运行，不能写成已验证修复。']
    lines+=['','## 解释边界','',
        '- 时间序列能定位性能何时丢失；没有重新随机化训练因素，不能单独识别是actor更新、critic更新还是状态分布造成的训练因果效应。',
        '- H下限干预同时改变控制行为和H成本，不能把所有改善都归因于求解器失败被消除；直接比较实际失败计数。',
        '- Q1贪心的结果检验从critic提取动作是否足够；它并非重新训练，也不是全局最优H。',
        '- 教师数据在最终回放中仍保留6000/21000。性能遗失不等于教师数据被覆盖，数据保留也不保证移动Bellman目标正确。',
        '- 6000次预训练包括actor、Q1/Q2、V和target V的SAC更新。没有行为克隆；上一轮成功的相对分支价值监督与本轮算法不同。',
        '', '## 预算与审计','','```json',json.dumps(budget,indent=2),'```','',
        '逐步物理reward、观测、动作、有限终止、配对起始状态与soft-return求和均独立核验。所有分支前缀与同一规则轨迹逐步精确匹配。新诊断没有SAC训练更新；全部失败和检查点结果保留。','',
        '![机制诊断](diagnosis.png)','']
    dest=OUT.parents[1]/'report/sac_teacher_cause';dest.mkdir(parents=True,exist_ok=True)
    body='\n'.join(lines);(dest/'report.md').write_text(body)
    for name in ['diagnosis.png','switch_trace.png']:
        body=body.replace('](%s)'%name,'](../../research_artifacts/bohn2021_reproduction_2026-09-17/report/sac_teacher_cause/%s)'%name)
    (ROOT/'docs/reports/bohn2021_sac_teacher_cause_2026-09-19.md').write_text(body)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    if args.partial:
        a=analyze();print(json.dumps({'conditions':len(a['groups']),'branch_blocks':len(a['branches'])}))
    else:
        budget=audit();a=analyze();plots(a);report(a,budget);print(json.dumps(budget,indent=2))
