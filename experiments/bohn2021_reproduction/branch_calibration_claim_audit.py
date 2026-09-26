"""Check user-level claims without opening a test bank unless already evaluated."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def gate(split):
    p=OUT/(split+'_gate.json')
    d=read(p)
    assert d['data_audit_passed']
    for name,h in d['hashes'].items(): assert sha(Path(name))==h, name
    return d


def main():
    requirement=read(OUT/'claim_requirements.json')
    assert requirement['original_gate_required']
    result={'core_reproduction_achieved':False, 'strict_original_numerical_reproduction':False,
        'reason_original':'Original complete author experiment configs/test data remain unavailable.',
        'claim_requirements_sha256':sha(OUT/'claim_requirements.json'),
        'measured_independent_acceleration_supported':False, 'checks':{}, 'test_rows':[]}
    if not (OUT/'validation_gate.json').exists():
        result['status']='Training or validation incomplete; no efficacy claim.'
    else:
        v=gate('validation')
        result['checks']['registered_validation_gate']=v['passed']
        if not v['passed']:
            assert not (OUT/'evaluations/test').exists()
            result['status']='Validation gate failed; independent test sealed and not evaluated.'
        elif not (OUT/'test_gate.json').exists():
            result['status']='Validation gate passed; independent test evidence incomplete.'
        else:
            test=gate('test')
            result['checks']['registered_test_gate']=test['passed']
            good=True
            for task in ('vehicle','pendulum'):
                lookup={(r['seed'],r['arm']):r for r in test['rows'] if r['task']==task}
                for seed in range(3):
                    counts={}
                    for arm in ('actor','raw_greedy','calibrated_greedy','fixed'):
                        p=OUT/'evaluations/test'/task/('%s_s%d'%(arm,seed))/'summary.json'
                        summary=read(p)
                        assert len(summary['episodes'])==20
                        counts[arm]=sum(e['termination']==('goal' if task=='vehicle' else 'steps') for e in summary['episodes'])
                        r=lookup[seed,arm]
                        result['test_rows'].append({'task':task,'seed':seed,'arm':arm,'total_cost':r['cost'],
                            'successes':counts[arm],'episodes':20,'constraints':r['constraints'],'solver_failed_steps':r['solver_failures']})
                    passed=all(counts['calibrated_greedy']>=counts[a] for a in ('actor','raw_greedy','fixed'))
                    result['checks']['%s_s%d_success_noninferiority'%(task,seed)]=passed
                    good &= passed
            result['control_effect_supported']=bool(test['passed'] and good)
            result['status']='Independent control gate %s; computation tradeoff and complete delivery still require review.' % ('passed' if result['control_effect_supported'] else 'failed')
            # Even a control pass cannot certify the entire goal or a speed claim.
    timing=OUT/'serial_timing/timing_audit.json'
    result['validation_timing_audited']=timing.exists() and read(timing).get('passed',False)
    result['scope']='This audit is necessary but does not replace the complete user-goal requirement audit. No goal status is changed automatically.'
    (OUT/'claim_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 本轮可支持的结论边界', '',result['status'], '',
        '当前不能把整个复现目标标为完成。原始精确配置缺失、独立证据完整性、计算代价与全部交付仍须逐项核实。', '',
        '|任务|种子|方法|独立测试总成本|成功/20|约束回合|求解失败步|',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in result['test_rows']:
        lines.append('|{task}|{seed}|{arm}|{total_cost:.4f}|{successes}/20|{constraints}|{solver_failed_steps}|'.format(**r))
    if not result['test_rows']: lines += ['', '没有已审计的本轮独立测试结果；不读取封存场景来补表。']
    lines += ['', '验证计时是否已审计：%s。它不是独立加速证据。' % result['validation_timing_audited'],
        '完整逐种子判据及本文件对应的要求散列见 claim_audit.json。']
    (OUT/'claim_audit_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(result['status'])


if __name__=='__main__': main()
