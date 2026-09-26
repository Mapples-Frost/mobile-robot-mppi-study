"""Audit the changed initial-distribution experiment without mixing old scenes."""
import hashlib
import json
import numpy as np
from runtime import ART,ROOT
from run import write

OUT=ART/'results/recoverable_distribution';DEST=ART/'report/recoverable_distribution'
def read(p):return json.loads(p.read_text())


def main():
    import sys
    global OUT,DEST
    prior='--prior' in sys.argv
    if prior:OUT=ART/'results/prior_refinement';DEST=ART/'report/prior_refinement'
    jobs=read(OUT/'training_completed.json')['jobs'];assert len(jobs)==15
    evaluations=read(OUT/'evaluation_completed.json')['jobs'];assert len(evaluations)==30
    for path,digest in read(OUT/'source_hashes.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    key=lambda c:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    cases=read(OUT/'holdout_bank.json')['cases'];assert len(set(map(key,cases)))==30
    assert not set(map(key,cases))&set(map(key,read(OUT/'validation_bank.json')['cases']))
    bounds={'pos':.3,'v':.5,'theta':.35,'omega':.5}
    for c in cases:assert all(abs(c['state'][k])<=v for k,v in bounds.items())
    selected=read(OUT/'selected_fixed.json')['h'];rows=[];starts=None
    for job in jobs:
        folder=OUT/job['name'];spec=read(folder/'manifest.json');done=read(folder/'completed.json')
        assert done['steps']==15000 and done['updates']==14745
        assert spec['config_sha256']==hashlib.sha256((OUT/'pendulum.json').read_bytes()).hexdigest()
        for mode in ['value','no_value']:
            ev=folder/('holdout_'+mode);s=read(ev/'summary.json');frozen=read(ev/'completed.json')
            assert frozen['frozen'] and frozen['weights_sha256']==done['final_hash']
            assert len(s['episodes'])==30 and s['physical_cost_and_bounds_verified']
            initial=[e['initial_state'] for e in s['episodes']]
            if starts is None:starts=initial
            assert starts==initial
            for i,ep in enumerate(s['episodes']):
                trace=read(ev/('trace_%02d.json'%i));assert len(trace)==ep['steps'] and np.isclose(sum(t['cost'] for t in trace),ep['total_cost'])
            rows.append({'name':folder.name,'mode':mode,'h':spec['fixed_horizon'],'seed':spec['seed'],'cost':s['mean_total_cost'],
                'constraints':s['constraint_episodes'],'performance':float(np.mean([e['performance_cost'] for e in s['episodes']])),
                'compute':float(np.mean([e['computation_cost'] for e in s['episodes']])),
                'mean_h':float(np.mean([e['mean_horizon'] for e in s['episodes']])),
                'costs':[e['total_cost'] for e in s['episodes']],'terminations':[e['termination'] for e in s['episodes']]})
    comparison={}
    for mode in ['value','no_value']:
        rr=[r for r in rows if r['mode']==mode and r['h'] is None];ff=[r for r in rows if r['mode']==mode and r['h']==selected]
        rc=np.array([r['costs'] for r in rr]);fc=np.array([r['costs'] for r in ff]);assert rc.shape==fc.shape==(3,30)
        full=np.all([[t=='steps' for t in r['terminations']] for r in rr+ff],axis=0)
        comparison[mode]={'rl_mean':float(rc.mean()),'fixed_mean':float(fc.mean()),'selected_h':selected,
            'rl_seed_means':rc.mean(axis=1).tolist(),'fixed_seed_means':fc.mean(axis=1).tolist(),
            'improvement_pct':float(100*(fc.mean()-rc.mean())/abs(fc.mean())),
            'better_scenes':int(np.sum(rc.mean(axis=0)<fc.mean(axis=0))),
            'rl_constraints':[r['constraints'] for r in rr],'fixed_constraints':[r['constraints'] for r in ff],
            'joint_full_length_scenes':int(full.sum()),'joint_full_length_cost_delta':float((rc[:,full]-fc[:,full]).mean()) if full.any() else None}
    write(OUT/'summary.json',{'rows':rows,'comparison':comparison})
    write(OUT/'audit.json',{'passed':True,'runs':15,'training_transitions':225000,'episodes':900,'independent_scenes':30,'frozen':True,'matched_starts':True})
    DEST.mkdir(exist_ok=True);lines=['# 倒立摆初始分布重建实验','',
        '只收窄初值：位置±0.3、速度±0.5、角度±0.35rad、角速度±0.5。动力学、输入界限、reward、参考过程不变；使用完整参考预览和alpha0.01。不是原作者已核实的初始化，也不声称整个分布有形式化可恢复保证。原困难分布的全部失败结果仍保留。',
        '15个模型各15000步；固定H5..50网格各自训练终端价值，并给验证选中的固定H补齐三种子。新30个保留场景seed26091709，900次配对评估。','',
        '|终端|RL均值|固定H|固定均值|改善|RL失败/30|固定失败/30|','|---|---:|---:|---:|---:|---|---|']
    for mode,c in comparison.items():lines.append('|%s|%.4f|%d|%.4f|%.2f%%|%s|%s|'%(mode,c['rl_mean'],selected,c['fixed_mean'],c['improvement_pct'],c['rl_constraints'],c['fixed_constraints']))
    lines+=['','|模型|终端|成本|性能|H成本|平均H|违规|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|%s|%s|%.3f|%.3f|%.3f|%.2f|%d|'%(r['name'],r['mode'],r['cost'],r['performance'],r['compute'],r['mean_h'],r['constraints']))
    if prior:
        lines[0]='# Riccati终端初始化：同预算联合学习'
        lines.insert(2,'相对较小扰动版本，只把终端价值的初始权重换为模型线性化的离散折扣Riccati解；随后照常32步联合训练。RL和所有固定H均用相同先验初始化。新保留集seed26091710，不与上轮场景混合。')
        lines=[x.replace('seed26091709','seed26091710') for x in lines]
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
