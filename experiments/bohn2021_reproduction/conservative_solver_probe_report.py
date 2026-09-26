"""Audit saved solver-probe arrays and report all conditions, without new solves."""
from pathlib import Path
import json
import numpy as np
from conservative_iteration import OUT
from paper_h_soft_probe import read,digest
from run import write


def main():
    dest=OUT/'solver_initialization_probe';done=read(dest/'completed.json')
    for p,h in done['hashes'].items():assert digest(Path(p))==h
    checked=[]
    for seed in range(3):
        for h in (10,25):
            arrays=np.load(str(dest/('s%d_h%d_arrays.npz'%(seed,h))),allow_pickle=False)
            report=read(dest/('s%d_h%d.json'%(seed,h)))
            for row in report['results']:
                name=row['guess'];x=arrays[name+'_solution'];g=arrays[name+'_constraints']
                gc=float(np.maximum.reduce([np.zeros_like(g),arrays['lbg']-g,g-arrays['ubg']]).max())
                xc=float(np.maximum.reduce([np.zeros_like(x),arrays['lbx']-x,x-arrays['ubx']]).max())
                np.testing.assert_allclose([gc,xc],[row['max_constraint_residual'],row['max_bound_residual']],rtol=0,atol=1e-12)
                if name=='original_initial':np.testing.assert_allclose(x,arrays['recorded_solution'],rtol=0,atol=1e-10)
                checked.append(row)
    assert len(checked)==24
    failed=OUT/'solver_initialization_probe_failed_dmstruct'
    overhead=read(failed/'attempts.json')
    note=dict(reason='DMStruct bounds must be converted through .cat before NumPy residual arithmetic. Only representation conversion changed; tolerances unchanged.',
        interrupted_counts=overhead,raw_result_not_persisted=True,
        limitation='First interrupted raw solve completed but its arrays were not persisted; counted as overhead, not silently treated as recoverable data.',
        archived_source_hash=digest(failed/'source_before_fix.py'),current_source_hash=digest(Path(__file__).with_name('conservative_solver_initialization_probe.py')),
        prior_protocol_hash=digest(failed/'protocol.json'),current_protocol_hash=digest(dest/'protocol.json'))
    write(dest/'recovery_note.json',note)
    totals={k:done['counts'][k]+overhead[k] for k in done['counts']}
    result=dict(passed=True,conditions=6,raw_solve_results=24,counts_including_interruption=totals,
        source_completion_hash=digest(dest/'completed.json'),source_hash=digest(Path(__file__)),
        scope='Saved numeric arrays have no pickle objects; constraint/bound residuals recomputed and original-initial replay compared. Does not reconstruct the NLP constraint function or certify global optimality.',
        local_finding='Seed2 H10 original and returned-solution initial guesses failed, both constant-state alternatives found a feasible solver-success solution. Thus infeasibility detection here did not establish global infeasibility.',
        efficacy_claim=False,test_access=False)
    write(dest/'array_audit.json',result)
    lines=['# 同一NLP的求解初值机制诊断','',
        '事后诊断已暴露的车辆验证场景2：三个独立终端模型，首步H10/H25，四类初值。其余NLP参数、边界与求解器完全相同；所有条件保留。重试结果没有施加到物理系统。','',
        '|种子|H|初值|求解成功|最大约束残差|迭代次数|NLP目标|',
        '|---|---:|---|---|---:|---:|---:|']
    for r in checked:lines.append('|%d|%d|%s|%s|%.3g|%d|%.6f|'%(r['seed'],r['h'],r['guess'],r['success'],r['max_constraint_residual'],r['iterations'],r['objective']))
    lines+=['','种子2、H10的原初值和失败返回解均报告不可行，约束残差分别约0.7007和0.6819；两类当前状态常值初值均成功，残差约9.98e-9。参数保持相同时找到可行解，否定了把原求解器的局部不可行报告当作该NLP无可行解的解释。它不证明这就是之后150步失败的全部原因，也不证明闭环恢复或泛化有效。',
        '其他五个种子/H组合的四类初值均成功。原初值重解与保存解在1e-10内一致；NPZ全部以allow_pickle=False读取，约束和变量边界残差独立重算一致。NLP目标不是实际回合成本，不能跨H按该列选择最佳策略。',
        '一次残差计算类型错误中断已归档，未放宽任何容差；其1次环境步、1次reset、1次原始求解全部计费，未保存的首个原始解不冒称可恢复数据。合计4次环境构造、7次reset、7次环境步、25次额外原始求解。每次reset仍含作者H50热身。',
        '下一步需要检验失败发生时同一步重试的闭环作用，并同时给固定H比较器相同恢复机会、记录原始失败次数与重试开销。该干预必须独立记录，不更改已完成验证；新的效果验证使用新场景。','']
    (dest/'report_CN.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
