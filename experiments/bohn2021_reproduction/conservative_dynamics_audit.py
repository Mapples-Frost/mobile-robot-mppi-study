"""Independent one-step integration of every saved training source and branch.

No controller invocation, fit, validation access or threshold tuning. Exact
repeated transition inputs share an integration calculation, but every recorded
output is compared. Counts are numerical-audit work, not new MPC interactions.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from conservative_iteration import OUT, TASKS, verify
from conservative_fixed_log_audit import integrate
from paper_h_soft_probe import read, digest
from run import write


def main(smoke=False):
    verify()
    dest=OUT/'independent_dynamics'/('smoke' if smoke else 'formal');dest.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),Path(__file__).with_name('conservative_fixed_log_audit.py').resolve()]
    registration=dict(source_hashes={str(p):digest(p) for p in sources},rtol=1e-7,atol=1e-7,
        vehicle='Analytic constant-input unicycle integration at dt=0.1',
        pendulum='SciPy DOP853, rtol=atol=1e-12, dt=0.04; independent from author simulator integration',
        scope='Every saved training source and branch, all seeds and rounds. Exact transition inputs cached per condition.',
        no_control_or_model_updates=True,held_out_data_access=False)
    rp=dest/'registration.json'
    if rp.exists():assert read(rp)==registration
    else:write(rp,registration)
    if (dest/'completed.json').exists():
        old=read(dest/'completed.json')
        for p,h in old['hashes'].items():assert digest(Path(p))==h
        assert old['passed'];print('Already complete; unchanged audit verified');return
    conditions=[(task,0,0) for task in TASKS] if smoke else [(task,seed,r) for task in TASKS for seed in range(3) for r in range(2)]
    results=[];failures=[];hashes={str(rp):digest(rp)};started=time.time()
    for task,seed,round_id in conditions:
        folder=OUT/('smoke_'+task if smoke else '%s_s%d_r%d'%(task,seed,round_id))
        done_path=folder/'collection_completed.json';done=read(done_path)
        assert done['audit_passed'] and done['smoke']==smoke
        hashes[str(done_path)]=digest(done_path)
        cases=read(folder/'train_bank.json')['cases']
        paths=[folder/('source_%02d.json'%cid) for cid in range(len(cases))]
        paths += [folder/('case%02d_t%03d_h%02d.json'%(g['case'],g['anchor'],int(h))) for g in done['groups'] for h in g['branches']]
        assert len(paths)==len(set(paths))
        actual=set(folder.glob('source_*.json'))|set(folder.glob('case*_t*_h*.json'))
        assert set(paths)==actual,'Unexpected or missing persisted training trace'
        cache={};rows_checked=0;max_error=0.;local_failures=0;began=time.time()
        per_file=[]
        for path in paths:
            assert digest(path)==done['hashes'][str(path)],str(path)
            data=read(path);trace=data if isinstance(data,list) else data['trace']
            file_error=0.;file_failures=0
            for index,row in enumerate(trace):
                state=row['previous_state'];control=row['input']
                key=tuple(sorted(state.items()))+tuple((k,tuple(np.asarray(v).reshape(-1))) for k,v in sorted(control.items()))
                if key not in cache:cache[key]=integrate(task,state,control)
                expected=cache[key]
                assert set(expected)==set(row['state'])
                for name,value in expected.items():
                    actual_value=row['state'][name];error=float(abs(actual_value-value))
                    assert np.isfinite(error),(path,index,name)
                    file_error=max(file_error,error)
                    if error>1e-7+1e-7*abs(value):
                        failures.append(dict(path=str(path),row=index,state=name,observed=actual_value,independent=float(value),absolute_error=error))
                        file_failures+=1
                rows_checked+=1
            per_file.append(dict(path=str(path),rows=len(trace),max_absolute_state_error=file_error,failed_components=file_failures))
            max_error=max(max_error,file_error);local_failures+=file_failures
        result=dict(task=task,seed=seed,round=round_id,source_episodes=len(cases),branch_traces=len(paths)-len(cases),
            rows_checked=rows_checked,unique_transition_integrations=len(cache),max_absolute_state_error=max_error,
            failed_state_components=local_failures,elapsed_s=time.time()-began,files=per_file)
        target=dest/('%s_s%d_r%d.json'%(task,seed,round_id));write(target,result);hashes[str(target)]=digest(target)
        results.append({k:v for k,v in result.items() if k!='files'})
        print(json.dumps(results[-1]),flush=True)
    expected_branches=sum(sum(len(g['branches']) for g in read(OUT/('smoke_'+task if smoke else '%s_s%d_r%d'%(task,seed,r))/'collection_completed.json')['groups']) for task,seed,r in conditions)
    assert sum(r['branch_traces'] for r in results)==expected_branches
    with (dest/'conditions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    write(dest/'failures.json',failures)
    for p in (dest/'conditions.csv',dest/'failures.json'):hashes[str(p)]=digest(p)
    summary=dict(passed=not failures,smoke=smoke,conditions=results,hashes=hashes,elapsed_s=time.time()-started,
        source_episodes=sum(r['source_episodes'] for r in results),branch_traces=expected_branches,
        recorded_transition_comparisons=sum(r['rows_checked'] for r in results),
        unique_numerical_integrations=sum(r['unique_transition_integrations'] for r in results),
        controller_step_calls=0,model_updates=0,failed_state_components=len(failures),
        limitations='Checks one-step dynamics conditional on logged prior state and control. Does not independently optimize controls, validate solver-success flags or reintegrate reset warmups. Exact repeated inputs are cached, overlapping suffixes are not independent scientific samples. This supplementary numerical audit cannot establish efficacy.')
    write(dest/'completed.json',summary)
    lines=['# 已保存训练轨迹的独立动力学积分核验','',
        '车辆使用解析积分，倒立摆使用独立DOP853积分；每个保存状态都参与比较，相同状态与输入只缓存积分计算。预先固定比较容差为rtol=atol=1e-7，DOP853内部容差均为1e-12。','',
        '|任务|种子|轮|源回合|分支|记录转移核验|独立积分计算|最大绝对状态差|超容差分量|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('|%s|%d|%d|%d|%d|%d|%d|%.3g|%d|'%(r['task'],r['seed'],r['round'],r['source_episodes'],r['branch_traces'],r['rows_checked'],r['unique_transition_integrations'],r['max_absolute_state_error'],r['failed_state_components']))
    lines+=['','审计通过：'+str(summary['passed'])+'。没有新MPC控制调用或模型更新；独立积分计算属于额外离线核验计算，已单列数量与耗时，不并入RL经验步数。',
        '此审计以记录的前一状态和控制输入为条件，不重解优化、不核实求解器成功标志、不覆盖reset热身转移。与既有成本/约束/观测审计互补，不替代闭环验证或证明收益。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    assert summary['passed'],'Independent dynamics comparison failed; preserve all failures and investigate without loosening tolerance'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
