"""Physical, replay and frozen-policy audit plus complete optimization report."""
import argparse
import json
from pathlib import Path
import sys

# Check before importing PyTorch or scanning traces: measured-delay rollouts
# use wall-clock compute as plant latency, so even read-only audits interfere.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'icra_scene_design'))
    from sweep_guard import require_no_live_sweep
    require_no_live_sweep('SAC preservation report and full trace audit')

import numpy as np
import torch
from sac_preserve_common import OUT,OLD,ROOT,ARMS,HS,read,write,verify,metric,state_features,rule_index,sha
from sac_preserve_agent import load,digest,teacher_data,ValuePrior
from sac_preserve_inference import HorizonPolicy
from sac_teacher_report import audit_row


def aggregate(episodes,bank=None,traces=None):
    out={key:float(np.mean([r[key] for r in episodes])) for key in ['adjusted_cost','physical_cost','H_cost','mean_H']}
    out.update(failures=sum(r['termination']=='constraint' for r in episodes),episodes=len(episodes),
               solver_failures=sum(r['solver_failures'] for r in episodes),steps=sum(r['steps'] for r in episodes),
               scene_costs=[r['adjusted_cost'] for r in episodes])
    if traces is not None:
        pre=[];post=[];complete=0;observed=0
        for e,rows in zip(episodes,traces):
            for switch in bank[e['scene']]['switches']:
                before=[r for r in rows if switch-30<=r['step']+1<switch]
                after=[r for r in rows if switch<=r['step']+1<switch+30]
                pre+=before;post+=after;complete+=len(before)==len(after)==30;observed+=bool(before or after)
        out['pre30_H']=float(np.mean([r['horizon'] for r in pre])) if pre else None
        out['post30_error']=float(np.mean([abs(r['state']['pos']-r['next_obs'][4]) for r in post])) if post else None
        out['switch_windows_complete']=complete;out['switch_windows_observed']=observed
    return out


def analyze():
    result={}
    for split in ['validation','holdout']:
        groups={};bank=read(OUT/(split+'_bank.json'))['scenes']
        for p in sorted((OUT/'evaluation'/split).glob('*/completed.json')):
            eps=read(p)['episodes'];traces=[read(p.parent/('scene_%02d.json'%e['scene']))['trace'] for e in eps]
            groups[p.parent.name]=aggregate(eps,bank,traces)
        result[split]=groups
    contrasts=[];g=result['holdout']
    for arm in ARMS:
        for seed in range(3):
            name='%s_s%d_15000'%(arm,seed)
            if name not in g:continue
            for comparator in ['free_s%d_15000'%seed,'plain_s%d'%seed,'teacher_s%d'%seed,'rule','fixed25','value_teacher']:
                if comparator not in g or comparator==name:continue
                delta=np.asarray(g[name]['scene_costs'])-g[comparator]['scene_costs']
                contrasts.append({'name':name,'comparator':comparator,'mean_cost_delta':float(delta.mean()),
                    'paired_scene_deltas':delta.tolist(),'wins':int(sum(delta<0))})
    result['contrasts']=contrasts;write(OUT/'analysis.json',result);return result


def audit():
    verify();assert read(OUT/'completed.json')['models']==9
    assert read(OUT/'smoke.json')['passed'] and read(OUT/'checks.json')['passed']
    exports=read(OUT/'export_audit.json');assert len(exports['models'])==9
    assert exports['inference_sha256']==sha(Path(__file__).with_name('sac_preserve_inference.py'))
    for item in exports['models']:assert item['sha256']==sha(OUT/'export'/(item['model']+'.json'))
    training_steps=training_resets=0;teacher_sizes={};initials={};updates=0
    for seed in range(3):
        teacher=teacher_data(seed);teacher_sizes[str(seed)]=len(teacher.rows)
        pre=OUT/'pretrained'/('s%d.pt'%seed);pa=load(pre);initials[seed]=digest(pa)
        pm=read(pre.with_suffix('.json'));assert pm['imitation_updates']==pm['critic_updates']==1000
    trainstats={}
    for arm in ARMS:
        for seed in range(3):
            name='%s_s%d'%(arm,seed);folder=OUT/'models'/name;m=read(folder/'completed.json')
            assert m['steps']==15000 and m['updates']==14901 and m['initial_hash']==initials[seed]
            bank=read(OUT/('train_s%d_bank.json'%seed))['scenes'];rows=[];prev=None;episode=-1;resets=0
            for line in (folder/'transitions.jsonl').open():
                r=json.loads(line)
                if r['episode']!=episode:
                    assert r['episode']==episode+1
                    if prev is not None:assert prev['done']
                    episode=r['episode'];prev=None;resets+=1
                assert r['online_step']==len(rows)+1
                audit_row(r,bank[episode],r['step'],prev);assert HS[r['action_index']]==r['horizon']
                if prev:assert r['step']==prev['step']+1
                else:assert r['step']==1
                rows.append(r);prev=r
            assert len(rows)==15000 and resets==m['resets']
            path=folder/'step_15000.pt';a=load(path);state=torch.load(path,weights_only=False,map_location='cpu')
            assert digest(a)==m['final_hash'];assert len(state['replay'])==len(rows)
            for actual,r in zip(state['replay'],rows):
                expected=(state_features(r['obs']),HS.index(r['horizon']),-r['cost']/.6,state_features(r['next_obs']),float(r['done']),rule_index(r['obs']))
                for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
            for step in [5000,10000,15000]:assert (folder/('step_%05d.pt'%step)).exists()
            trainstats[name]={'steps':len(rows),'physical_failures':sum(r['termination']=='constraint' for r in rows),
                'solver_failures':sum(not r['solver_success'] for r in rows),'resets':resets,'replay_verified':True}
            training_steps+=len(rows);training_resets+=resets;updates+=m['updates']
    evalsteps=evaleps=0;starts={};split_counts={}
    for split in ['validation','holdout']:
        bank=read(OUT/(split+'_bank.json'))['scenes']
        names=['%s_s%d_15000'%(arm,s) for arm in ARMS for s in range(3)]+['rule','fixed25','value_teacher']
        if split=='validation':names+=['pretrained_s%d'%s for s in range(3)]+['%s_s%d_%d'%(arm,s,step) for arm in ARMS for s in range(3) for step in [5000,10000]]
        else:names+=['%s_s%d'%(arm,s) for arm in ['plain','teacher'] for s in range(3)]
        folder=OUT/'evaluation'/split
        assert set(names)=={p.name for p in folder.iterdir() if p.is_dir()}
        split_counts[split]=0
        for name in names:
            eps=read(folder/name/'completed.json')['episodes'];assert [r['scene'] for r in eps]==list(range(len(bank)))
            agent=None;prior=None;portable=None
            if name.startswith('pretrained'):agent=load(OUT/'pretrained'/('s%s.pt'%name.split('_s')[1]))
            # A trained arm is arm_seed_step (three tokens). The bare name 'rule'
            # is the hand-written rule BASELINE, not the 'rule' training arm, and
            # 'rule' is in ARMS -- so a prefix test alone sends it into the
            # three-way unpack and raises. Require the full form.
            elif len(name.split('_'))==3 and name.split('_')[0] in ARMS:
                arm,seed,step=name.split('_');agent=load(OUT/'models'/(arm+'_'+seed)/('step_%05d.pt'%int(step)))
                if int(step)==15000:portable=HorizonPolicy(OUT/'export'/(arm+'_'+seed+'.json'))
            elif name=='value_teacher':prior=ValuePrior()
            for e in eps:
                d=read(folder/name/('scene_%02d.json'%e['scene']));assert d['summary']==e;trace=d['trace'];previous=None
                key=(split,e['scene'])
                if key in starts:np.testing.assert_array_equal(starts[key],d['initial'])
                starts[key]=d['initial'];np.testing.assert_array_equal(d['initial'],trace[0]['obs'])
                for i,r in enumerate(trace):audit_row(r,bank[e['scene']],i+1,previous);previous=r
                assert trace[-1]['done']
                for k,v in metric(trace).items():
                    if isinstance(v,str):assert e[k]==v
                    else:np.testing.assert_allclose(e[k],v,atol=1e-9)
                x=torch.tensor(np.asarray([state_features(r['obs']) for r in trace]))
                with torch.no_grad():
                    if agent:hs=np.asarray(HS)[agent.actor(x).argmax(-1).numpy()]
                    elif prior:hs=np.asarray(HS)[prior.costs(x).argmin(-1).numpy()]
                    elif name=='rule':hs=[HS[rule_index(r['obs'])] for r in trace]
                    elif name=='fixed25':hs=[25]*len(trace)
                    else:hs=None
                if hs is not None:np.testing.assert_array_equal(hs,[r['horizon'] for r in trace])
                if portable:np.testing.assert_array_equal([portable.predict(r['obs']) for r in trace],hs)
                evalsteps+=len(trace);evaleps+=1;split_counts[split]+=1
    smoke=read(OUT/'smoke.json');checks=read(OUT/'checks.json')
    budget={'passed':True,'training_steps':training_steps,'training_resets':training_resets,
        'joint_updates':updates,'shared_imitation_updates':3000,'shared_critic_updates':3000,
        'evaluation_steps':evalsteps,'evaluation_episodes':evaleps,'evaluation_by_split':split_counts,
        'smoke_physical_steps':smoke['transitions']+smoke['resets']+checks['full_switch_physical_steps'],
        'smoke_network_updates':8+8+21+1+20,'inherited_teacher_transitions':18000,
        'teacher_rows_retained_by_seed':teacher_sizes,'inherited_branch_study_whole_physical_steps':217730,
        'inherited_scope':'Historical full-study cost, not marginal training-only dependency cost; not equal total budget vs old SAC.',
        'audit_source_sha256':sha(Path(__file__))}
    budget['new_physical_steps']=training_steps+training_resets+evalsteps+evaleps+budget['smoke_physical_steps']
    write(OUT/'training_audit.json',trainstats);write(OUT/'audit.json',budget);return budget


def deliver(a,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=OUT.parents[1]/'report/sac_preserve';dest.mkdir(parents=True,exist_ok=True)
    g=a['holdout'];labels=['plain','teacher']+ARMS
    display_names={'plain':'plain_sac','teacher':'teacher_sac','free':'free_arm',
                   'rule':'rule_arm','value':'value_arm','fixed25':'fixed25',
                   'value_teacher':'value_teacher'}
    means={}
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for i,arm in enumerate(labels):
        names=['%s_s%d%s'%(arm,s,'_15000' if arm in ARMS else '') for s in range(3)]
        values=[g[n]['adjusted_cost'] for n in names];means[arm]=float(np.mean(values))
        axs[0].scatter(np.repeat(i,3),values,s=40,label=arm);axs[0].plot([i-.2,i+.2],[np.mean(values)]*2,color='black',lw=2)
    for n,c in [('fixed25','#666666'),('rule','#009E73'),('value_teacher','#CC79A7')]:
        axs[0].axhline(g[n]['adjusted_cost'],ls='--',c=c,label=display_names[n] if n != 'rule' else 'rule_baseline')
    axs[0].set(xticks=range(5),xticklabels=['Old plain','Old teacher','Free SAC','Rule prior','Value prior'],ylabel='Mean holdout cost (lower is better)',title='A  All three seeds; shared 12 fresh scenes')
    axs[0].tick_params(axis='x',rotation=18);axs[0].legend(fontsize=7)
    colors={'free':'#0072B2','rule':'#D55E00','value':'#009E73'}
    for arm in ARMS:
        for seed in range(3):
            names=['pretrained_s%d'%seed]+['%s_s%d_%d'%(arm,seed,t) for t in [5000,10000,15000]]
            values=[a['validation'][n]['adjusted_cost'] for n in names]
            axs[1].plot([0,5,10,15],values,'o-',color=colors[arm],alpha=.6,label=arm if seed==0 else None)
    axs[1].set(xlabel='Online transitions (thousands)',ylabel='Mean validation cost',title='B  Every checkpoint, no selection');axs[1].legend()
    fig.savefig(dest/'optimization.png',dpi=180);fig.savefig(dest/'optimization.pdf');plt.close(fig)
    lines=['# 教师保持 SAC 优化结果','',
        '三组各3个配对种子，统一19维因果预览特征、七个整数H候选、规则模仿初始化、critic-only预训练、固定40%教师回放与原奖励/解析终端。free不加在线策略先验，rule加入规则动作交叉熵，value加入冻结分支价值教师集成交叉熵。先验不写入SAC Q目标。','',
        '这是显式保留教师的混合算法。动作集合、网络、表示、鲁棒Q损失等共同改动与旧作者SAC不同，不能把对旧模型的差异归因于单一因素；三新组之间共享这些设置。旧模型在同一新测试集重新评价，不跨场景比较旧数字。','',
        '|方法|3种子平均成本|相对新free成本降低率|','|---|---:|---:|']
    for arm in labels:lines.append('|%s|%.4f|%.2f%%|'%(display_names[arm],means[arm],100*(means['free']-means[arm])/means['free']))
    for n in ['fixed25','rule','value_teacher']:
        label='rule_baseline' if n=='rule' else display_names[n]
        lines.append('|%s|%.4f|—|'%(label,g[n]['adjusted_cost']))
    lines+=['','上述成本=原始总成本+294.3，比例依赖此零点；失败保留原罚项与未执行步的展示补齐。固定H25是历史锁定参照，不声称是新场景的最优固定H。','',
        '|模型|成本|物理成本|H成本|物理失败/回合|求解失败/步|切换前30步H|切换后误差/m|完整切换窗口|','|---|---:|---:|---:|---|---|---:|---:|---:|']
    for name,v in sorted(g.items()):
        label='rule_baseline' if name=='rule' else name
        lines.append('|%s|%.3f|%.3f|%.3f|%d/%d|%d/%d|%.2f|%.4f|%d/%d|'%(label,v['adjusted_cost'],v['physical_cost'],v['H_cost'],v['failures'],v['episodes'],v['solver_failures'],v['steps'],v['pre30_H'] or 0,v['post30_error'] or 0,v['switch_windows_complete'],v['switch_windows_observed']))
    lines+=['','## 配对结论','']
    for arm in ['rule','value']:
        for comp in ['free','plain','teacher','rule','fixed25','value_teacher']:
            cname=lambda s:('%s_s%d%s'%(comp,s,'_15000' if comp=='free' else '')) if comp in ['free','plain','teacher'] else comp
            deltas=[g['%s_s%d_15000'%(arm,s)]['adjusted_cost']-g[cname(s)]['adjusted_cost'] for s in range(3)]
            comp_display = 'rule_baseline' if comp == 'rule' else display_names.get(comp, comp)
            lines.append('- %s相对%s：各种子成本差%s；均值差%.4f；%d/3种子改善。'%(display_names[arm],comp_display,', '.join('%.4f'%d for d in deltas),np.mean(deltas),sum(d<0 for d in deltas)))
    lines+=['','## 全部验证检查点','', '|模型|预训练|5k|10k|15k|','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in range(3):
            names=['pretrained_s%d'%seed]+['%s_s%d_%d'%(arm,seed,t) for t in [5000,10000,15000]]
            lines.append('|%s_s%d|%s|'%(arm,seed,'|'.join('%.3f'%a['validation'][n]['adjusted_cost'] for n in names)))
    profile=read(OUT/'preservation_profile.json')
    assert len(profile['profiles'])==30
    assert profile['source_sha256']==sha(Path(__file__).with_name('sac_preserve_profile.py'))
    lines+=['','## 同一批教师状态上的能力保持','',
        '每种子从历史6000教师转移每20条取1条，固定300个状态比较各检查点。可见变化状态是输入预览中确有参考变化的子集。恒定预览只修改未来参考输入，保持物理状态、当前参考和时间；这不是物理闭环评价。教师参考regret来自冻结教师模型，不能解释为真实SAC Q误差。','',
        '|检查点|规则动作一致率|有变化预览时H|对应恒定预览H|冻结教师参考regret|','|---|---:|---:|---:|---:|']
    for r in profile['profiles']:
        lines.append('|%s|%.3f|%.2f|%.2f|%.5f|'%(r['checkpoint'],r['rule_agreement'],r['visible_change_mean_H'],r['held_preview_mean_H_on_visible_states'],r['frozen_teacher_regret']))
    lines+=['','## 方法与边界','',
        '- 预训练先做1000次规则动作模仿，再冻结actor做1000次critic更新；三个组从同种子的完全相同权重和优化器状态启动。在线14901次更新，先验权重从约2降至.5，未衰减到零。',
        '- value先验来自此前全部3个round1分支价值模型的等权集成，softmax温度.1。模型只读取当前可见预览，不读取未来事件日程。其训练标签为规则续控、截断和近似尾项的相对成本，不能视为当前策略真实soft-Q。',
        '- 原6000教师转移只保留执行整数H属于候选集合的记录；保留比例与继承预算见下表。没有把其他H动作篡改成候选动作。',
        '- 明确加入冻结value_teacher参照：若SAC不胜过该教师，改善只能归结为保住或利用教师，不能宣称在线RL进一步提升。',
        '- 完整回合物理失败及求解失败独立报告。窗口遇到提前终止按实际观察计数，不填造缺失轨迹。',
        '- 三个训练种子不足以支持广泛显著性推断；12个场景是配对评价，不是36个独立训练重复。所有预定模型和检查点保留，没有根据holdout调参。',
        '- 七个候选排除H1..4是明确的动作空间限制，不是求解器修复；H费用仍是计算代理，不能直接称为CPU加速。',
        '', '## 推理入口','',
        '全部九个最终actor已导出到`results/sac_preserve/export/`。`sac_preserve_inference.HorizonPolicy(path).predict(obs56)`只依赖NumPy，返回整数H；不需要critic或教师在线运行。导出后的动作在全部最终模型验证/测试轨迹上与PyTorch实现逐步一致。仍要求使用本实验的56维观测定义与冻结Riccati终端，未验证其他被控对象。',
        '', '## 审计与预算','', '```json',json.dumps(budget,indent=2,ensure_ascii=False),'```','',
        '独立逐步重算reward、动作映射、预览、终止、最终replay；验证公共初始权重、评估起始状态、冻结源码散列和模型重载后动作。继承的分支预算是历史完整研究开销，不能当作新算法免费获得的数据，也不是本轮新增开销。',
        '', '![优化结果](optimization.png)','']
    body='\n'.join(lines);(dest/'report.md').write_text(body)
    (ROOT/'docs/reports/bohn2021_sac_preserve_2026-09-19.md').write_text(body.replace('](optimization.png)','](../../research_artifacts/bohn2021_reproduction_2026-09-17/report/sac_preserve/optimization.png)'))
    write(OUT/'comparison.json',{'mean_costs':means,'controls':{n:g[n]['adjusted_cost'] for n in ['rule','fixed25','value_teacher']}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    if args.partial:
        a=analyze();print({k:len(a[k]) for k in ['validation','holdout']})
    else:
        b=audit();a=analyze();deliver(a,b);print(json.dumps(b,indent=2))

