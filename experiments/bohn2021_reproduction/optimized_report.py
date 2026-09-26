"""Complete, matched reporting for the improved reconstruction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import ART,ROOT
from run import write

OUT=ART/'results/optimized';DEST=ART/'report/optimized'


def read(p):return json.loads(p.read_text())


def main():
    assert read(OUT/'training_completed.json')['complete']
    assert read(OUT/'holdout_completed.json')['complete']
    DEST.mkdir(exist_ok=True)
    selected=read(OUT/'validation_selected_fixed.json')
    training_jobs=read(OUT/'training_completed.json')['jobs']
    assert len(training_jobs)==30 and len({j['name'] for j in training_jobs})==30
    assert all(j['exit_code']==0 for j in training_jobs)
    holdout_jobs=read(OUT/'holdout_completed.json')['jobs']
    assert len(holdout_jobs)==60 and all(j['exit_code']==0 for j in holdout_jobs)
    rows=[];audit=[]
    for j in training_jobs:
        folder=OUT/j['name'];spec=read(folder/'manifest.json');task=spec['task']
        assert 'optimized_reconstruction' in spec
        assert hashlib.sha256((ART/'configs'/(task+'.json')).read_bytes()).hexdigest()==spec['config_sha256']
        assert hashlib.sha256((ART/'configs'/(task+'_validation_bank.json')).read_bytes()).hexdigest()==spec['test_bank_sha256']
        assert read(folder/'completed.json')['steps']==15000
        assert read(folder/'completed.json')['updates']==14745
        for mode in ['value','no_value']:
            ev=folder/('holdout_'+mode);s=read(ev/'summary.json')
            assert read(ev/'completed.json')['frozen'] and s['physical_cost_and_bounds_verified']
            assert read(ev/'completed.json')['weights_sha256']==read(folder/'completed.json')['final_hash']
            assert len(s['episodes'])==20
            for k,ep in enumerate(s['episodes']):
                trace=read(ev/('trace_%02d.json'%k))
                assert len(trace)==ep['steps']
                assert np.isclose(sum(t['cost'] for t in trace),ep['total_cost'])
                assert all(t['solver_calls']==(3 if task=='vehicle' else 1) for t in trace)
                assert all(1<=t['horizon']<=50 for t in trace)
            row={'name':folder.name,'task':task,'seed':spec['seed'],'fixed_h':spec['fixed_horizon'],'terminal':mode,
                'mean_cost':s['mean_total_cost'],'median_cost':float(np.median([e['total_cost'] for e in s['episodes']])),
                'goals':s['goal_episodes'],'constraints':s['constraint_episodes'],
                'mean_h':float(np.mean([e['mean_horizon'] for e in s['episodes']])),
                'performance':float(np.mean([e['performance_cost'] for e in s['episodes']])),
                'compute':float(np.mean([e['computation_cost'] for e in s['episodes']])),
                'constraint_cost':float(np.mean([e['constraint_cost'] for e in s['episodes']])),
                'mean_step_s':float(np.mean([e['mean_step_s'] for e in s['episodes']])),
                'solver_failures':sum(e['solver_failure_steps'] for e in s['episodes']),
                'costs':[e['total_cost'] for e in s['episodes']]}
            rows.append(row);audit.append({'name':folder.name,'mode':mode,'passed':True})
    # All terminal conditions and policies must share the same formal initial state.
    for task in ['vehicle','pendulum']:
        def case_key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        new=read(OUT/(task+'_holdout_bank.json'))['cases']
        existing=[c for n in ['test','validation','holdout'] for c in read(ART/'configs'/('%s_%s_bank.json'%(task,n)))['cases']]
        existing+=read(ART/'results/mechanism_probe'/(task+'_bank.json'))['cases']
        assert len(set(map(case_key,new)))==20
        assert not set(map(case_key,new)) & set(map(case_key,existing))
        names=[j['name'] for j in read(OUT/'training_completed.json')['jobs'] if j['name'].startswith(task)]
        starts=None
        for name in names:
            for mode in ['value','no_value']:
                current=[e['initial_state'] for e in read(OUT/name/('holdout_'+mode)/'summary.json')['episodes']]
                if starts is None:starts=current
                assert current==starts,(task,name,mode,'initial states mismatch')
    hashes=read(OUT/'source_hashes.json')
    implementation_update=json.loads((OUT/'implementation_update.json').read_text(encoding='utf-8-sig')) if (OUT/'implementation_update.json').exists() else None
    for path,expected in hashes.items():
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        if implementation_update and path==implementation_update['path']:
            assert expected==implementation_update['old_sha256'] and actual==implementation_update['new_sha256']
            assert hashlib.sha256((OUT/'optimized_runtime_before_vectorization.py').read_bytes()).hexdigest()==expected
            assert read(OUT/'initialization_equivalence.json')['exact']
        else:assert actual==expected,path
    comparison={}
    for task in ['vehicle','pendulum']:
        comparison[task]={}
        for mode in ['value','no_value']:
            subset=[r for r in rows if r['task']==task and r['terminal']==mode]
            rl=[r for r in subset if r['fixed_h'] is None]
            h=selected[task]['h'];fixed=[r for r in subset if r['fixed_h']==h]
            assert len(rl)==len(fixed)==3
            rc=np.array([r['costs'] for r in rl]);fc=np.array([r['costs'] for r in fixed])
            # Average over training seeds before paired scene comparisons.
            delta=rc.mean(axis=0)-fc.mean(axis=0)
            grid=[r for r in subset if r['fixed_h'] is not None and r['seed']==0]
            assert sorted(r['fixed_h'] for r in grid)==list(range(5,51,5))
            best=min(grid,key=lambda r:r['mean_cost'])
            comparison[task][mode]={'selected_fixed_h':h,'rl_mean':float(rc.mean()),'fixed_mean':float(fc.mean()),
                'relative_improvement_pct':float(100*(fc.mean()-rc.mean())/abs(fc.mean())),
                'paired_scene_delta_rl_minus_fixed':delta.tolist(),'rl_better_scene_count':int(np.sum(delta<0)),
                'rl_seed_means':rc.mean(axis=1).tolist(),'fixed_seed_means':fc.mean(axis=1).tolist(),
                'best_holdout_grid_h_descriptive':best['fixed_h'],'best_holdout_grid_cost_descriptive':best['mean_cost']}
    write(OUT/'summary.json',{'rows':rows,'comparison':comparison})
    write(OUT/'audit.json',{'passed':True,'training_runs':30,'training_transitions':450000,
        'holdout_episode_evaluations':1200,'independent_holdout_scenes_per_task':20,
        'frozen':True,'initial_states_match_across_all_conditions':True,'source_hashes_match':True,'rows':audit})
    lines=['# Bøhn 2021 核心假设：优化重建对照','',
        '## Material Passport','', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run','- Origin Date: 2026-09-17','- Verification Status: VERIFIED（执行与核验）',
        '- Version Label: optimized_reconstruction_v1','',
        '## 方法与原文的一致性','',
        '保留作者SAC连续动作缩放取整得到H∈[1,50]、MPC动力学和约束、原论文reward公式及权重、32步联合终端学习、15k训练预算。采用引用链核实的batch256、buffer1e6、actor32×32、critic256×256、Adam3e-4、tau0.005、固定熵1、reward除数0.6/0.3、gamma=rho=0.97。',
        '显式扩展：终端价值使用相对当前参考误差的全二次模型，以L Lᵀ保证半正定曲率；MPC末端参考使用所选H的参考点；车辆增加MPC已知的预测偏移/参考预览、目标偏移与时间，倒立摆增加参考预览和时间；车辆每步固定三个初值求解，倒立摆单个确定性初值。reset统一使用零终端H50预热。不是逐项原样复现。',
        '车辆三次求解的成本未假装包含在原H代理中；双方使用相同求解流程，表中另报墙钟耗时。并行运行的墙钟测量仅供资源核算，不是专用平台的实时性基准。',
        '每任务3个RL种子；固定H5..50每档独立训练终端价值，先各1个种子。在原验证集上按平均成本选择固定H，再补训另外2个种子，之后才统一评估新20场景保留集（seed26091704）。主比较使用这个验证选择的固定H的3种子均值；另完整列出全部固定H以免隐去强对照。',
        '没有按保留测试选择种子、检查点或删除失败场景。学习启动100步、初始状态分布和观测特征仍是重建选择。三个扩展同时实施，正式结果不能单独归因于其中某一项。','',
        '执行中对初值构造做了等价的批量赋值优化：15组H/初值向量逐元素完全相同。已运行任务保留原实现，后续任务使用向量化实现；原版本、两个散列和等价核验均保留。该变化不涉及算法或超参数。',
        '## 主比较','', '|任务|终端|RL均值|验证选固定H|固定均值|RL成本改善|RL更好场景|',
        '|---|---|---:|---:|---:|---:|---:|']
    for task in comparison:
        for mode,c in comparison[task].items():
            lines.append('|%s|%s|%.4f|%d|%.4f|%.2f%%|%d/20|'%(task,mode,c['rl_mean'],c['selected_fixed_h'],c['fixed_mean'],c['relative_improvement_pct'],c['rl_better_scene_count']))
    lines+=['','正改善表示RL成本更低；负改善表示固定H更好。仅20个独立场景/任务与3个训练种子，不据此声称普遍优势或统计显著性。零终端是冻结消融，不是重新训练的独立无终端策略。','',
        '## 全部条件','', '|模型|终端|平均成本|中位成本|性能|H成本|约束成本|平均H|到达|违规|求解失败步|毫秒/步|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|%s|%s|%.3f|%.3f|%.3f|%.3f|%.3f|%.2f|%d|%d|%d|%.2f|'%(r['name'],r['terminal'],r['mean_cost'],r['median_cost'],r['performance'],r['compute'],r['constraint_cost'],r['mean_h'],r['goals'],r['constraints'],r['solver_failures'],1000*r['mean_step_s']))
    lines+=['','## 文件与重跑','',
        '- 协议、模型、manifest、逐步轨迹、源文件散列与核验位于 results/optimized。',
        '- 运行：`python experiments/bohn2021_reproduction/optimized_suite.py --workers 8`（遗留Python3.7）。',
        '- 汇总：`.venv/bin/python experiments/bohn2021_reproduction/optimized_report.py`。','']
    (DEST/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(comparison,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
