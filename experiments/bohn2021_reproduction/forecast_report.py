"""Frozen matched comparison for full-reference preview."""
import hashlib
import json
import numpy as np
from runtime import ART,ROOT
from run import write

OUT=ART/'results/forecast_refinement';DEST=ART/'report/forecast_refinement'
def read(p):return json.loads(p.read_text())


def main():
    import sys
    global OUT,DEST
    confirmation='--confirmation' in sys.argv
    if confirmation:OUT=ART/'results/forecast_confirmation';DEST=ART/'report/forecast_confirmation'
    source=ART/'results/forecast_refinement'
    assert read(source/'training_completed.json')['complete']
    ncases=40 if confirmation else 20
    jobs=read(OUT/'evaluation_completed.json')['jobs'];assert len(jobs)==18
    for path,digest in read(source/'source_hashes.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    key=lambda c:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    bank=read(OUT/'holdout_bank.json')['cases'];old=[]
    for p in list((ART/'configs').glob('pendulum_*bank.json'))+[ART/'results/optimized/pendulum_holdout_bank.json',ART/'results/policy_refinement/holdout_bank.json',ART/'results/mechanism_probe/pendulum_bank.json']:old+=read(p)['cases']
    if confirmation:old+=read(source/'holdout_bank.json')['cases']
    assert len(set(map(key,bank)))==ncases and not set(map(key,bank))&set(map(key,old))
    rows=[];starts=None
    for job in jobs:
        assert job['exit_code']==0
        name,mode=job['name'],job['mode']
        group='forecast' if 'forecast' in name else 'sparse' if 'alpha' in name else 'fixed30'
        folder=source/name if group=='forecast' else ART/'results'/('policy_refinement' if group=='sparse' else 'optimized')/name
        spec=read(folder/'manifest.json');complete=read(folder/'completed.json');assert complete['steps']==15000 and complete['updates']==14745
        ev=OUT/'evaluations'/name/mode;s=read(ev/'summary.json');frozen=read(ev/'completed.json')
        assert frozen['frozen'] and frozen['weights_sha256']==complete['final_hash']
        assert s['physical_cost_and_bounds_verified'] and len(s['episodes'])==ncases
        initial=[e['initial_state'] for e in s['episodes']]
        if starts is None:starts=initial
        assert initial==starts
        for i,e in enumerate(s['episodes']):
            trace=read(ev/('trace_%02d.json'%i));assert len(trace)==e['steps'] and np.isclose(sum(t['cost'] for t in trace),e['total_cost'])
        rows.append({'name':name,'mode':mode,'group':group,'cost':s['mean_total_cost'],'constraints':s['constraint_episodes'],
            'mean_h':float(np.mean([e['mean_horizon'] for e in s['episodes']])),'costs':[e['total_cost'] for e in s['episodes']],
            'terminations':[e['termination'] for e in s['episodes']],'steps':[e['steps'] for e in s['episodes']]})
    comparison={}
    for mode in ['value','no_value']:
        comparison[mode]={}
        fixed=np.array([r['costs'] for r in rows if r['mode']==mode and r['group']=='fixed30']);assert fixed.shape==(3,ncases)
        for group in ['sparse','forecast']:
            x=np.array([r['costs'] for r in rows if r['mode']==mode and r['group']==group]);assert x.shape==(3,ncases)
            relevant=[r for r in rows if r['mode']==mode and r['group'] in [group,'fixed30']]
            full=np.all([[t=='steps' for t in r['terminations']] for r in relevant],axis=0)
            failed=np.all([[t=='constraint' for t in r['terminations']] for r in relevant],axis=0)
            comparison[mode][group]={'mean':float(x.mean()),'seed_means':x.mean(axis=1).tolist(),'fixed_mean':float(fixed.mean()),
                'improvement_pct':float(100*(fixed.mean()-x.mean())/abs(fixed.mean())),
                'better_scenes':int(np.sum(x.mean(axis=0)<fixed.mean(axis=0))),
                'all_models_full_length_scenes':int(full.sum()),
                'full_length_mean_delta':float((x[:,full]-fixed[:,full]).mean()) if full.any() else None,
                'all_models_failed_scenes':int(failed.sum()),
                'failed_scene_mean_delta':float((x[:,failed]-fixed[:,failed]).mean()) if failed.any() else None}
    write(OUT/'summary.json',{'rows':rows,'comparison':comparison});write(OUT/'audit.json',{'passed':True,'new_training_runs':0 if confirmation else 3,'new_training_transitions':0 if confirmation else 45000,'episodes':18*ncases,'independent_scenes':ncases,'source_hashes_match':True,'initial_states_match':True})
    DEST.mkdir(exist_ok=True);lines=['# 完整参考预览单因素实验','',
        '只把倒立摆策略参考预览从3点扩展到50点，其余采用alpha0.01优化重建设置。3个新模型各15k步，所有最终种子均报告。与稀疏预览alpha0.01三种子、固定H30三种子在新20场景(seed26091706)配对评估；不是原文逐项复现。','',
        '|终端|策略输入|RL均值|固定H30|RL改善|更好场景|','|---|---|---:|---:|---:|---:|']
    for mode,groups in comparison.items():
        for group,r in groups.items():lines.append('|%s|%s|%.4f|%.4f|%.2f%%|%d/20|'%(mode,group,r['mean'],r['fixed_mean'],r['improvement_pct'],r['better_scenes']))
    lines+=['','360次回合评估仅对应20个独立场景。未选择最佳种子或检查点，不声称统计显著性。',
        '', '**不能把综合成本正改善解释为控制质量提高。** 场景分解显示收益主要来自同样失败但更早终止的回合；所有模型完整运行100步的场景中，完整预览模型平均成本反而更高。失败惩罚没有完全抵消缩短回合后少累计的性能成本。原始场景和成本均保留，此结果不算复现控制性能优势。',
        '', '|终端|策略输入|共同完整回合数|完整回合RL−固定|共同失败回合数|失败回合RL−固定|',
        '|---|---|---:|---:|---:|---:|']
    for mode,groups in comparison.items():
        for group,r in groups.items():lines.append('|%s|%s|%d|%.4f|%d|%.4f|'%(mode,group,r['all_models_full_length_scenes'],r['full_length_mean_delta'],r['all_models_failed_scenes'],r['failed_scene_mean_delta']))
    if confirmation:
        lines=[line.replace('新20场景(seed26091706)','额外40场景(seed26091707)').replace('360次回合评估仅对应20个独立场景','720次回合评估仅对应40个独立场景').replace('/20|','/40|') for line in lines]
        lines.insert(2,'本次没有新增训练；使用全部冻结模型确认，原20场景结果单独保留。')
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
