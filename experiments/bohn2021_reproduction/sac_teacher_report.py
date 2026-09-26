"""Independent accounting and descriptive reporting for matched SAC study."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sac_teacher_study import OUT, ROOT, read, write, verify, sha, SEEDS


def expected_obs(state, scene, elapsed):
    clock=elapsed+1
    refs=[v['true'][0] for v in scene['case']['tvp']['pos_r']]
    pos=state['pos']
    return np.asarray([pos/1.5,state['v']/5,state['theta']/(np.pi/2),state['omega']/10,refs[clock]]+
        [(refs[clock+k]-pos)/1.5 for k in range(1,51)]+[(600-elapsed)/600],np.float32)


def audit_row(row, scene, elapsed, previous=None):
    s=row['state'];v=s['v'];om=s['omega'];th=s['theta'];u=row['input']['u1']
    ref=scene['case']['tvp']['pos_r'][elapsed+1]['true'][0]
    perf=.4*v*v+.05*v*om*np.cos(th)+(1/120)*om*om-.4905*np.cos(th)+10*(s['pos']-ref)**2+.1*u*u
    fail=abs(s['pos'])>1.5 or abs(th)>np.pi/2
    penalty=10*(600-elapsed) if fail else 0.
    h=row['horizon']
    np.testing.assert_allclose([row['performance'],row['compute'],row['constraint'],row['cost']],
        [perf,.003*h,penalty,perf+.003*h+penalty],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(row['next_obs'],expected_obs(s,scene,elapsed),rtol=0,atol=1e-7)
    if previous is not None:
        assert not previous['done']
        np.testing.assert_array_equal(row['obs'],previous['next_obs'])
    # Unscaled continuous actions and physical integer execution must agree.
    assert h==int(np.clip(np.rint(row['unscaled_action']),1,50))
    assert abs((row['action']+1)*24.5+1-row['unscaled_action'])<5e-6
    assert row['done']==(fail or elapsed==600)
    assert (row['termination']=='constraint')==fail
    assert abs(u)<=5+1e-5
    assert np.isfinite(list(s.values())).all()


def train_audit(dest):
    manifest=read(dest/'completed.json');seed=manifest['seed'];banks={}
    for phase in ['online','teacher']:
        banks[phase]=read(OUT/('%s_s%d_bank.json'%(phase,seed)))['scenes']
    summaries=[];previous=None;elapsed=0;key=None;rows=[]
    resets=0;failures=0;solver=0;phasecounts={'teacher':0,'online':0}
    for line in (dest/'transitions.jsonl').open():
        row=json.loads(line);newkey=(row['phase'],row['episode'])
        if newkey!=key:
            key=newkey;elapsed=0;previous=None;resets+=1
        elapsed+=1
        audit_row(row,banks[row['phase']][row['episode']],elapsed,previous)
        previous=row;rows.append(row)
        phasecounts[row['phase']]+=1
        failures+=row['termination']=='constraint';solver+=not row['solver_success']
    assert phasecounts['online']==manifest['online_steps']
    assert phasecounts['teacher']==manifest['teacher_steps']
    if not manifest['smoke']:assert manifest['updates']==20901
    # Saved replay checks verify scaling once, true terminal masks and exact action storage.
    last='step_%05d'%manifest['online_steps']
    with (dest/(last+'_state.pkl')).open('rb') as f:state=pickle.load(f)
    replay=state['replay']._storage
    assert len(replay)==len(rows)
    for tr,row in zip(replay,rows):
        np.testing.assert_array_equal(tr[0],np.asarray(row['obs'],np.float32))
        np.testing.assert_array_equal(tr[3],np.asarray(row['next_obs'],np.float32))
        assert tr[2]==-row['cost']/.6 and tr[4]==float(row['done'])
        assert float(tr[1][0])==float(np.float32(row['action']))
    phase_stats={}
    for phase in ['teacher','online']:
        part=[r for r in rows if r['phase']==phase]
        if not part:continue
        groups={}
        for r in part:groups.setdefault(r['episode'],[]).append(r)
        full=[g for g in groups.values() if g[-1]['done']]
        phase_stats[phase]={'steps':len(part),'complete_episodes':len(full),'failures':sum(g[-1]['termination']=='constraint' for g in full),
            'mean_full_adjusted_cost':float(np.mean([sum(r['cost'] for r in g)+294.3 for g in full])) if full else None,
            'mean_H':float(np.mean([r['horizon'] for r in part]))}
    return {'arm':dest.name,'steps':len(rows),'resets':resets,'constraint_episodes':failures,'solver_failures':solver,
        'updates':manifest['updates'],'phase':phase_stats,'initial_hash':manifest['initial_hash'],
        'replay_verified':True,'physical_cost_and_observations_verified':True}


def audit_training_only():
    verify()
    dests=[p for p in (OUT/'models').glob('*') if (p/'completed.json').exists()]
    data=[train_audit(p) for p in sorted(dests)]
    for seed in SEEDS:
        paired=[d['initial_hash'] for d in data if d['arm'].endswith('_s%d'%seed)]
        assert len(set(paired))<=1
    write(OUT/'training_audit.json',data)
    return data


def audit_all():
    data=audit_training_only()
    assert len(data)==6
    completed=read(OUT/'completed.json')
    selected=read(OUT/'selection.json')['fixed']
    learned=['%s_s%d_%d'%(arm,seed,steps) for arm,steps in [('plain',15000),('plain',21000),('teacher',15000)] for seed in SEEDS]
    from sac_teacher_study import GRID
    expected={'validation':set(['fixed_%d'%h for h in GRID+read(OUT/'refinement.json')['extra']]+['switch_5_30']+learned),
              'holdout':set([selected,'fixed_30','switch_5_30']+learned)}
    assert set(completed['holdout_arms'])==expected['holdout']
    for split,arms in expected.items():
        folder=OUT/'evaluation'/split
        assert {p.name for p in folder.iterdir() if p.is_dir()}==arms
        bank=read(OUT/(split+'_bank.json'))['scenes']
        for arm in arms:
            summary=read(folder/arm/'summary.json')
            assert summary['arm']==arm
            assert [r['scene'] for r in summary['episodes']]==[r['id'] for r in bank]
            assert len(list((folder/arm).glob('scene_*.json')))==len(bank)
            for row in summary['episodes']:
                assert read(folder/arm/('scene_%02d.json'%row['scene']))['summary']==row
            np.testing.assert_allclose(summary['mean_adjusted_cost'],np.mean([r['adjusted_cost'] for r in summary['episodes']]))
    fixed_summaries=[read(OUT/'evaluation/validation'/arm/'summary.json') for arm in expected['validation'] if arm.startswith('fixed_')]
    assert selected==min(fixed_summaries,key=lambda r:(r['mean_adjusted_cost'],int(r['arm'].split('_')[1])))['arm']
    eval_steps=0;eval_resets=0;initials={};counts={}
    for split in ['validation','holdout']:
        bank=read(OUT/(split+'_bank.json'))['scenes']
        counts[split]=0
        for p in sorted((OUT/'evaluation'/split).glob('*/scene_*.json')):
            result=read(p);summary=result['summary'];scene=summary['scene'];trace=result['trace']
            key=(split,scene)
            if key in initials:np.testing.assert_array_equal(initials[key],result['initial'])
            initials[key]=result['initial']
            previous=None
            for i,row in enumerate(trace):
                audit_row(row,bank[scene],i+1,previous);previous=row
            assert trace[-1]['done']
            np.testing.assert_allclose(summary['adjusted_cost'],sum(r['cost'] for r in trace)+294.3,atol=1e-8)
            assert summary['steps']==len(trace)
            assert summary['termination']==trace[-1]['termination']
            assert summary['solver_failures']==sum(not r['solver_success'] for r in trace)
            np.testing.assert_allclose([summary['physical_cost'],summary['H_cost'],summary['constraint_cost'],summary['mean_H'],summary['discounted_cost']],
                [sum(r['performance']+.4905 for r in trace),sum(r['compute'] for r in trace),sum(r['constraint'] for r in trace),
                 np.mean([r['horizon'] for r in trace]),sum(.97**t*r['cost'] for t,r in enumerate(trace))])
            # Independent exact-reference RMSE; stored float32-reference metric can differ slightly.
            rmse=np.sqrt(np.mean([(r['state']['pos']-bank[scene]['case']['tvp']['pos_r'][i+2]['true'][0])**2 for i,r in enumerate(trace)]))
            assert abs(rmse-summary['rmse'])<1e-6
            eval_steps+=len(trace);eval_resets+=1;counts[split]+=1
    smoke=train_audit(OUT/'smoke/teacher_s0')
    budget={'formal_training_transitions':sum(d['steps'] for d in data),'formal_SAC_updates':sum(d['updates'] for d in data),
        'evaluation_transitions':eval_steps,'evaluation_episodes':counts,'reset_warmup_steps':sum(d['resets'] for d in data)+eval_resets+smoke['resets'],
        'smoke_transitions':smoke['steps'],'smoke_updates':smoke['updates']}
    budget['total_physical_steps']=budget['formal_training_transitions']+eval_steps+smoke['steps']+budget['reset_warmup_steps']
    recovery=None
    if (OUT/'recovery.json').exists():
        from sac_teacher_recovery import audit as recovery_audit
        recovery=recovery_audit()
        interrupted=read(OUT/'recovery.json')['interrupted']
        for name,entry in interrupted.items():
            dest=OUT/'interrupted_shutdown'/name
            seed=read(dest/'manifest.json')['seed']
            banks={phase:read(OUT/('%s_s%d_bank.json'%(phase,seed)))['scenes'] for phase in ['online','teacher']}
            previous=None;key=None;elapsed=0
            with (dest/'transitions.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line);newkey=(row['phase'],row['episode'])
                    if newkey!=key:previous=None;elapsed=0;key=newkey
                    elapsed+=1
                    audit_row(row,banks[row['phase']][row['episode']],elapsed,previous)
                    previous=row
        budget['interrupted_saved_transitions']=sum(r['saved_transitions'] for r in interrupted.values())
        budget['interrupted_observed_reset_warmups']=sum(r['observed_reset_warmups'] for r in interrupted.values())
        budget['interrupted_SAC_updates_lower_bound']=sum(read(OUT/'interrupted_shutdown'/name/'progress.json')['updates'] for name in interrupted)
        budget['total_SAC_updates_lower_bound']=budget['formal_SAC_updates']+budget['smoke_updates']+budget['interrupted_SAC_updates_lower_bound']
        budget['completed_study_physical_steps']=budget.pop('total_physical_steps')
        budget['total_physical_steps_lower_bound']=budget['completed_study_physical_steps']+budget['interrupted_saved_transitions']+budget['interrupted_observed_reset_warmups']
        budget['interrupted_unflushed_steps']='Unknown; total including shutdown work is a lower bound.'
    result={'passed':True,'training':data,'budget':budget,'identical_evaluation_starts':True,
        'paired_initial_networks':True,'finite_endpoint_no_bootstrap':True,'reward_scaled_once':True,'registered_hashes_verified':True,
        'all_registered_evaluations_present':True,'recovery':recovery,'report_source_sha':sha(Path(__file__))}
    write(OUT/'audit.json',result)
    return result


def analyze():
    selected=read(OUT/'selection.json')['fixed']
    raw={p.parent.name:read(p) for p in (OUT/'evaluation/holdout').glob('*/summary.json')}
    result={}
    for group,arms in [(selected,[selected]),('fixed_30',['fixed_30']),('rule',['switch_5_30']),
        ('plain15k',['plain_s%d_15000'%s for s in SEEDS]),('plain21k',['plain_s%d_21000'%s for s in SEEDS]),
        ('teacher',['teacher_s%d_15000'%s for s in SEEDS])]:
        means={k:[float(np.mean([r[k] for r in raw[a]['episodes']])) for a in arms] for k in
            ['adjusted_cost','raw_cost','physical_cost','H_cost','mean_H','rmse','discounted_cost']}
        result[group]={'arms':arms,'means':{k:float(np.mean(v)) for k,v in means.items()},'per_seed':means,
            'constraints':sum(r['termination']=='constraint' for a in arms for r in raw[a]['episodes']),
            'solver_failures':sum(r['solver_failures'] for a in arms for r in raw[a]['episodes']),
            'steps':sum(r['steps'] for a in arms for r in raw[a]['episodes']),
            'episodes':sum(len(raw[a]['episodes']) for a in arms)}
    comparisons={}
    for other in [selected,'rule','plain15k','plain21k']:
        before=result[other]['means']['adjusted_cost'];after=result['teacher']['means']['adjusted_cost']
        comparisons[other]={'teacher_minus_comparator':after-before,'reduction_pct':100*(before-after)/before}
    paired=[]
    for seed in SEEDS:
        teacher=np.array([r['adjusted_cost'] for r in raw['teacher_s%d_15000'%seed]['episodes']])
        row={'seed':seed,'teacher_mean':float(teacher.mean())}
        for name,arm in [('fixed',selected),('rule','switch_5_30'),('plain15k','plain_s%d_15000'%seed),('plain21k','plain_s%d_21000'%seed)]:
            other=np.array([r['adjusted_cost'] for r in raw[arm]['episodes']]);delta=teacher-other
            row[name]={'mean_delta':float(delta.mean()),'wins':int(sum(delta<0)),'scene_deltas':delta.tolist()}
        paired.append(row)
    bank=read(OUT/'holdout_bank.json')['scenes']
    behavior={}
    for arm in raw:
        phases={'plateau':[],'transition':[]};fail_h={};residuals=[];hist={}
        for p in sorted((OUT/'evaluation/holdout'/arm).glob('scene_*.json')):
            record=read(p);scene=bank[record['summary']['scene']]
            for row in record['trace']:
                clock=row['step']
                phase='transition' if any(t-50<=clock<t+100 for t in scene['switches']) else 'plateau'
                h=row['horizon'];phases[phase].append(h);hist[h]=hist.get(h,0)+1
                if not row['solver_success']:fail_h[h]=fail_h.get(h,0)+1
                residuals.append(row['max_constraint_residual'])
        behavior[arm]={'phase_mean_H':{p:float(np.mean(v)) for p,v in phases.items()},
            'H_histogram':hist,'failed_solver_H_histogram':fail_h,'max_constraint_residual':max(residuals)}
    output={'selected_fixed':selected,'groups':result,'teacher_comparisons':comparisons,'paired_by_seed':paired,'behavior':behavior,
        'percentages_use_offset_cost':True,'raw':raw}
    write(OUT/'analysis.json',output)
    return output


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    groups=[a['selected_fixed'],'rule','plain15k','plain21k','teacher']
    labels=['Selected fixed H','H5/H30 rule','SAC 15k','SAC 21k','Teacher + SAC']
    colors=['#666666','#009E73','#56B4E9','#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,g in enumerate(groups):
        r=a['groups'][g];values=r['per_seed']['adjusted_cost']
        offsets=np.linspace(-.12,.12,len(values)) if len(values)>1 else np.zeros(1)
        axes[0,0].scatter(i+offsets,values,color=colors[i],s=45)
        axes[0,0].plot([i-.2,i+.2],[np.mean(values)]*2,color=colors[i],lw=2)
    axes[0,0].set(xticks=range(5),xticklabels=labels,ylabel='Offset-adjusted episode cost',title='A  All training seeds; lower is better')
    axes[0,0].tick_params(axis='x',rotation=20)
    for s in SEEDS:
        row=a['paired_by_seed'][s]
        axes[0,1].scatter(np.arange(12)+(s-1)*.16,row['plain21k']['scene_deltas'],label='Seed %d'%s,alpha=.8)
    axes[0,1].axhline(0,color='black',lw=1)
    axes[0,1].set(xlabel='Fresh holdout scene',ylabel='Teacher cost minus SAC 21k',title='B  Matched total simulation/update budget')
    axes[0,1].legend()
    for i,g in enumerate(groups):
        r=a['groups'][g]['means']
        axes[1,0].bar(i,r['physical_cost'],color=colors[i],alpha=.55)
        axes[1,0].bar(i,r['H_cost'],bottom=r['physical_cost'],color=colors[i],label='H proxy' if i==0 else None)
        remainder=r['adjusted_cost']-r['physical_cost']-r['H_cost']
        if remainder>1e-6:
            axes[1,0].bar(i,remainder,bottom=r['physical_cost']+r['H_cost'],color='#CC79A7',hatch='//')
    axes[1,0].set(xticks=range(5),xticklabels=labels,ylabel='Mean cost components',title='C  Cost decomposition, including failures')
    axes[1,0].legend(handles=[Patch(facecolor='#777777',alpha=.55,label='Physical'),
        Patch(facecolor='#777777',label='H proxy'),Patch(facecolor='#CC79A7',hatch='//',label='Failure + offset')],fontsize=8,loc='upper left')
    axes[1,0].tick_params(axis='x',rotation=20)
    for arm,color in [('plain','#0072B2'),('teacher','#D55E00')]:
        for seed in SEEDS:
            losses=read(OUT/'models'/('%s_s%d'%(arm,seed))/'losses.json')
            axes[1,1].plot([r['update'] for r in losses],[r['values'][1] for r in losses],color=color,alpha=.45,label=arm if seed==0 else None)
    axes[1,1].set(yscale='log',xlabel='Cumulative SAC updates',ylabel='Sampled Q1 squared-error loss',title='D  Training loss is not value calibration')
    axes[1,1].legend()
    report=ART_REPORT();report.mkdir(parents=True,exist_ok=True)
    fig.savefig(report/'overview.png',dpi=170);fig.savefig(report/'overview.pdf');plt.close(fig)


def ART_REPORT():
    return OUT.parents[1]/'report/sac_teacher'


def report(a,audit):
    g=a['groups'];fixed=a['selected_fixed'];comparison=a['teacher_comparisons']
    better_fixed=g['teacher']['means']['adjusted_cost']<g[fixed]['means']['adjusted_cost']
    better_matched=g['teacher']['means']['adjusted_cost']<g['plain21k']['means']['adjusted_cost']
    seed_wins=sum(r['plain21k']['mean_delta']<0 for r in a['paired_by_seed'])
    verdict=('教师辅助组平均成本低于最佳已测固定H。' if better_fixed else '教师辅助组平均成本未低于最佳已测固定H。')
    verdict+=('相同总预算下，它的平均成本低于普通SAC。' if better_matched else '相同总预算下，它的平均成本未低于普通SAC。')
    verdict+='三个种子中有%d个在配对测试平均成本上胜过普通21k。'%seed_wins
    if g['teacher']['constraints'] or g['teacher']['solver_failures']:
        verdict+='教师组有%d/%d次物理约束终止、%d/%d个求解失败步，成本结果必须结合可靠性解读。'%(g['teacher']['constraints'],g['teacher']['episodes'],g['teacher']['solver_failures'],g['teacher']['steps'])
    else:
        verdict+='教师组在本测试集上无物理约束终止、无求解器失败。'
    lines=['# 作者 SAC 与教师数据辅助 SAC：倒立摆实验','',verdict,'',
        '本轮实际运行作者 TF1 SAC，保留 actor、双 Q、V/target V 及熵正则更新。教师数据只进入 replay，不使用监督 H 分类器或近似 soft-Q 标签。终端仍冻结，未运行联合终端学习。','',
        '## 新测试场景结果','',
        '3个训练种子、12个全新配对场景。成本越低越好；下表总成本与物理成本移除平衡点势能常数，百分比依赖该明确零点。原始总成本也完整保存。','',
        '|方法|平移总成本|物理成本|H代理成本|平均H|物理失败回合|求解失败步/总步|', '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [(fixed,'验证选定 '+fixed),('fixed_30','固定 H30'),('rule','H5/H30 规则'),('plain15k','普通 SAC 15k'),('plain21k','普通 SAC 21k'),('teacher','教师6k + SAC在线15k')]:
        r=g[name];m=r['means']
        lines.append('|%s|%.4f|%.4f|%.4f|%.2f|%d/%d|%d/%d|'%(label,m['adjusted_cost'],m['physical_cost'],m['H_cost'],m['mean_H'],r['constraints'],r['episodes'],r['solver_failures'],r['steps']))
    lines+=['','教师辅助相对普通21k的成本降低率为 **%.2f%%**；相对最佳已测固定H为 **%.2f%%**。负值表示退化，不改写为成功。'%(comparison['plain21k']['reduction_pct'],comparison[fixed]['reduction_pct']),
        '', '|种子|普通15k|普通21k|教师辅助|教师胜普通21k场景数|教师胜固定H场景数|','|---|---:|---:|---:|---:|---:|']
    for i,row in enumerate(a['paired_by_seed']):
        lines.append('|%d|%.4f|%.4f|%.4f|%d/12|%d/12|'%(i,g['plain15k']['per_seed']['adjusted_cost'][i],g['plain21k']['per_seed']['adjusted_cost'][i],row['teacher_mean'],row['plain21k']['wins'],row['fixed']['wins']))
    lines+=['','## 实验解释与边界','',
        '- 普通最终组与教师辅助组每种子均使用21,000条真实转移、20,901次SAC更新。教师预采样和预训练都计入预算。普通15k只作在线步数对照。',
        '- 教师为80%因果H5/H30规则、20%随机连续H。它改变了数据分布和更新安排，因此结果不单独识别“数据质量”的因果效应。没有复用旧监督模型或其分支计算预算。',
        '- 两组共用56维可见参考预览、原reward、固定alpha=.01、gamma=.97和冻结Riccati终端；600步末端不自举。动作仍为连续策略输出后舍入到H1..50。自定义循环从第100条转移起有放回采样batch256；作者learn原循环还检查buffer>=batch，故这也是一个明确区别。',
        '- 每个训练种子的初始网络相同，所有最终种子都报告。固定H只在4个验证场景上选择，未根据12个最终测试场景追加变体。',
        '- 平台/参考反转分布、完整预览、冻结终端、有限回合处理和自定义采样循环是与论文的明确差异；本轮是核心机制研究，不是原论文性能复现。',
        '- H成本是计算代理。只降低它不意味着物理控制质量同时提升，也不能直接宣称CPU加速。',
        '- 提前终止时总成本还包含失败罚项及未执行步的势能常数补齐；此时表内物理成本与H成本之和不等于总成本。图中用斜线单独标出这部分。',
        '- 训练中已观察到H2/H3求解器失败及非零约束残差；作者环境仍应用求解器返回控制。求解失败与实际物理约束终止分别计数，不能因物理回合完成便忽略优化器问题。',
        '- 3个种子不能支持强显著性结论；配对场景不是额外独立训练种子。所有失败保留。',
        '', '普通21k / 教师辅助的折扣原始成本均值分别为 %.6f / %.6f，位置RMSE为 %.6f / %.6f m。'%(g['plain21k']['means']['discounted_cost'],g['teacher']['means']['discounted_cost'],g['plain21k']['means']['rmse'],g['teacher']['means']['rmse']),
        '', '## 审计与预算','', '```json',json.dumps(audit['budget'],ensure_ascii=False,indent=2),'```','',
        '逐步独立重算物理reward、H罚项、失败罚项、观测预览与终止；检查最终replay的奖励仅除.6一次、动作与下一状态一致、有限末端不自举；验证配对初始网络与评价重置状态、协议散列。源文件审计结果见 results/sac_teacher/audit.json。',
        '', '关机中断的两个目录已完整归档，按原种子从头重跑；已完成的seed2模型逐文件散列验证未改变。中断日志中的19,348条转移及34次预热另计。关机瞬间未刷盘步数未知，因此含中断开销的总预算是可验证下界。恢复检查详见 recovery_audit.json。',
        '', '检查点保留网络、优化器变量、回放及Python/NumPy RNG；未声称精确恢复TF随机流或中途MPC内部状态。训练loss仅为诊断，不能作为真实Q已校准的证明。',
        '', '![全部种子与预算对照](overview.png)','']
    if (OUT/'critic_diagnostic.json').exists():
        diagnostic=read(OUT/'critic_diagnostic.json')
        lines+=['## Critic排序诊断','',
            '仅在上一轮48个验证锚点上检查冻结模型，不用于选择或调参。参考标签是固定规则续控的近似代价，不含SAC熵项；以下regret不能解释为真实soft-Q误差。Actor列将动作映射到最近候选，不能替代实际actor动作的分支评价。','',
            '|模型|Q1选择的规则参考regret|minQ参考regret|最近actor候选regret|actor/Q1候选不一致|',
            '|---|---:|---:|---:|---:|']
        for r in diagnostic['results']:
            lines.append('|%s|%.6f|%.6f|%.6f|%d/48|'%(r['arm'],r['Q1_policy_specific_regret'],r['minQ_policy_specific_regret'],r['nearest_candidate_to_actor_regret'],r['nearest_actor_Q1_disagreements']))
    lines+=['','## 分阶段H与求解残差','',
        '过渡阶段定义为每次参考切换前50步至后100步，其余为平台。它是事后统一统计窗口，不是提供给actor的事件日程。','',
        '|模型|平台平均H|过渡平均H|最大优化约束残差|','|---|---:|---:|---:|']
    for name,b in sorted(a['behavior'].items()):
        lines.append('|%s|%.3f|%.3f|%.3g|'%(name,b['phase_mean_H']['plateau'],b['phase_mean_H']['transition'],b['max_constraint_residual']))
    if not better_matched:
        successful=[arm for arm in g['plain21k']['arms'] if a['raw'][arm]['mean_adjusted_cost']<g[fixed]['means']['adjusted_cost']]
        lines+=['','## 结论与后续','',
            '这轮不支持教师回放预填和预训练能够改善作者SAC。教师组相对普通21k的物理成本增加%.4f，H代理成本变化%.4f；退化主要来自控制性能，不能解释成只多用了计算。'%(g['teacher']['means']['physical_cost']-g['plain21k']['means']['physical_cost'],g['teacher']['means']['H_cost']-g['plain21k']['means']['H_cost']),
            '', '相对普通15k的%.2f%%均值改善不构成同预算优势，因为教师组额外使用6000条采样和6000次预训练更新。原始起点折扣成本与全回合未折扣成本的排序不同，也不能据前者撤回预先指定主指标上的退化。'%comparison['plain15k']['reduction_pct'],
            '', '普通21k中低于固定H的种子数为%d/3，尚未形成稳定的跨种子优势。'%len(successful)]
        for arm in successful:
            rows=a['raw'][arm]['episodes']
            lines.append('%s虽成本较低，仍有%d/%d个求解失败步。'%(arm,sum(r['solver_failures'] for r in rows),sum(r['steps'] for r in rows)))
        lines+=['','后续建议先在新的预先固定诊断状态上，用当前冻结策略续控的实际H分支成本核对critic排序和actor决策，区分价值估计与策略利用问题；另行检查短H求解失败。该建议尚未执行，不据本轮holdout调参或追加变体。获得可靠、跨种子的H学习后，再恢复联合终端学习及固定H独立终端基线。','']
    body='\n'.join(lines)
    p=ART_REPORT();p.mkdir(parents=True,exist_ok=True);(p/'report.md').write_text(body)
    (ROOT/'docs/reports/bohn2021_sac_teacher_2026-09-19.md').write_text(body.replace('](overview.png)','](../../research_artifacts/bohn2021_reproduction_2026-09-17/report/sac_teacher/overview.png)'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--training-only',action='store_true');args=ap.parse_args()
    if args.training_only:print(audit_training_only())
    else:
        audit=audit_all();a=analyze();plot(a);report(a,audit)
        print(json.dumps({'audit_passed':True,'comparisons':a['teacher_comparisons'],'budget':audit['budget']},indent=2))
