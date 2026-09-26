"""Audit all entropy-only refinement conditions, including unchanged comparators."""
import hashlib
import json
import numpy as np
from runtime import ART,ROOT
from run import write

OUT=ART/'results/policy_refinement';DEST=ART/'report/policy_refinement'


def read(p):return json.loads(p.read_text())


def main():
    assert read(OUT/'training_completed.json')['complete']
    jobs=read(OUT/'evaluation_completed.json')['jobs'];assert len(jobs)==24
    assert all(j['exit_code']==0 for j in jobs)
    for path,expected in read(OUT/'source_hashes.json').items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    bank=read(OUT/'holdout_bank.json')['cases']
    key=lambda c:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    prior=[]
    for p in list((ART/'configs').glob('pendulum_*bank.json'))+[ART/'results/optimized/pendulum_holdout_bank.json',ART/'results/mechanism_probe/pendulum_bank.json']:
        prior.extend(read(p)['cases'])
    assert len(set(map(key,bank)))==20 and not set(map(key,bank))&set(map(key,prior))
    rows=[];starts=None
    for j in jobs:
        name,mode=j['name'],j['mode'];folder=OUT/name
        if not (folder/'manifest.json').exists():folder=ART/'results/optimized'/name
        spec=read(folder/'manifest.json');completed=read(folder/'completed.json')
        assert completed['steps']==15000 and completed['updates']==14745
        alpha=float(spec['adaptations']['ent_coef']);group='fixed30' if spec['fixed_horizon'] else 'alpha'+str(alpha)
        ev=OUT/'evaluations'/name/mode;s=read(ev/'summary.json');frozen=read(ev/'completed.json')
        assert frozen['frozen'] and frozen['weights_sha256']==completed['final_hash']
        assert s['physical_cost_and_bounds_verified'] and len(s['episodes'])==20
        initial=[e['initial_state'] for e in s['episodes']]
        if starts is None:starts=initial
        assert initial==starts
        for i,ep in enumerate(s['episodes']):
            tr=read(ev/('trace_%02d.json'%i));assert len(tr)==ep['steps']
            assert np.isclose(sum(r['cost'] for r in tr),ep['total_cost'])
        rows.append({'name':name,'group':group,'mode':mode,'seed':spec['seed'],
            'cost':s['mean_total_cost'],'constraints':s['constraint_episodes'],
            'performance':float(np.mean([e['performance_cost'] for e in s['episodes']])),
            'compute':float(np.mean([e['computation_cost'] for e in s['episodes']])),
            'constraint_cost':float(np.mean([e['constraint_cost'] for e in s['episodes']])),
            'costs':[e['total_cost'] for e in s['episodes']]})
    comparison={}
    for mode in ['value','no_value']:
        selected=[r for r in rows if r['mode']==mode]
        baseline=np.array([r['costs'] for r in selected if r['group']=='fixed30'])
        assert baseline.shape==(3,20)
        comparison[mode]={}
        for group in ['alpha1.0','alpha0.1','alpha0.01']:
            a=np.array([r['costs'] for r in selected if r['group']==group]);assert a.shape==(3,20)
            comparison[mode][group]={'mean':float(a.mean()),'seed_means':a.mean(axis=1).tolist(),
                'fixed_mean':float(baseline.mean()),'improvement_pct':float(100*(baseline.mean()-a.mean())/abs(baseline.mean())),
                'paired_scene_differences':(a.mean(axis=0)-baseline.mean(axis=0)).tolist(),
                'better_scenes':int(np.sum(a.mean(axis=0)<baseline.mean(axis=0)))}
    result={'rows':rows,'comparison':comparison}
    write(OUT/'summary.json',result)
    write(OUT/'audit.json',{'passed':True,'new_runs':6,'new_training_transitions':90000,
        'evaluated_models':12,'episodes':480,'independent_scenes':20,'starts_match':True,'frozen':True,'source_hashes_match':True})
    DEST.mkdir(exist_ok=True)
    lines=['# 倒立摆：熵权重单因素改进','',
        '保留reward、15k训练预算、控制器、终端结构和联合更新，只将SAC固定熵系数从1改为0.1或0.01。每项3种子；所有最终模型均报告。明确偏离论文引用设置，不称原样复现。',
        '在预先冻结的新20场景（seed26091705）上同时评估原alpha1三种子与固定H30三种子。480次配对回合只对应20个独立场景。零终端是冻结后消融。','',
        '|终端|熵系数|RL均值|固定H30均值|RL改善|更好场景|','|---|---:|---:|---:|---:|---:|']
    for mode,groups in comparison.items():
        for group,r in groups.items():lines.append('|%s|%s|%.4f|%.4f|%.2f%%|%d/20|'%(mode,group[5:],r['mean'],r['fixed_mean'],r['improvement_pct'],r['better_scenes']))
    lines+=['','改善为正代表RL成本更低。样本小，不据此宣称普遍优势或显著性；新的场景分布与作者原始测试仍不等同。','',
        '|模型|终端|成本|性能|H成本|约束成本|违规回合|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|%s|%s|%.3f|%.3f|%.3f|%.3f|%d|'%(r['name'],r['mode'],r['cost'],r['performance'],r['compute'],r['constraint_cost'],r['constraints']))
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
