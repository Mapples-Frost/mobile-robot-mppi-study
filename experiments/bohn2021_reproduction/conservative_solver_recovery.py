"""Bounded same-NLP primal retries; independent extension, disabled at reset."""
import copy
import json
import time
import numpy as np
from run import write,serial


def array(v):return np.asarray(v.cat if hasattr(v,'cat') else v,dtype=float)


def install(mpc,folder):
    from casadi import DM
    original=mpc.solve
    state=dict(enabled=False,events=[],counts=dict(solve_attempts=0,solve_completed=0,warmup_attempts=0,retry_attempts=0),step=None,case=None)
    attempts=folder/'solver_attempts.json'
    assert not attempts.exists(),'Inspect interrupted solver attempt before resuming'
    write(attempts,state['counts'])
    def call(kind):
        c=state['counts'];c['solve_attempts']+=1
        if kind=='warmup':c['warmup_attempts']+=1
        if kind.startswith('retry'):c['retry_attempts']+=1
        write(attempts,c)
        start=time.perf_counter();original();elapsed=time.perf_counter()-start
        c['solve_completed']+=1;write(attempts,c)
        x=array(mpc.opt_x_num);g=array(mpc.opt_g_num)
        finite=bool(np.isfinite(x).all() and np.isfinite(g).all())
        residual=lambda v,lo,hi:float(np.maximum.reduce([np.zeros_like(v),array(lo)-v,v-array(hi)]).max())
        gr=residual(g,mpc.cons_lb,mpc.cons_ub) if finite else None
        xr=residual(x,mpc.lb_opt_x,mpc.ub_opt_x) if finite else None
        stats=mpc.solver_stats
        row=dict(case=state['case'],step=state['step'],kind=kind,success=bool(stats['success']),
            return_status=stats.get('return_status'),iterations=int(stats.get('iter_count',-1)),
            finite=finite,constraint_residual=gr,bound_residual=xr,solver_s=elapsed,
            accepted=finite and bool(stats['success']) and max(gr,xr)<=1e-5)
        with (folder/'solver_calls.jsonl').open('a') as f:f.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
        return row
    def solve():
        if not state['enabled']:call('warmup');return
        records=[call('initial')]
        if not records[0]['accepted']:
            saved={name:copy.deepcopy(getattr(mpc,name)) for name in ('opt_g_num','opt_f_num','lam_g_num','lam_x_num','solver_stats')}
            initial_solution=array(mpc.opt_x_num).copy()
            for name,previous in [('retry_zero',False),('retry_previous',True)]:
                guess=mpc.opt_x(0);guess['_x']=mpc.opt_p_num['_x0']/mpc._x_scaling
                guess['_u']=mpc.opt_p_num['_u_prev']/mpc._u_scaling if previous else 0
                mpc.opt_x_num.master=guess.cat
                records.append(call(name))
                if records[-1]['accepted']:break
            if not records[-1]['accepted']:
                mpc.opt_x_num.master=DM(initial_solution)
                mpc.opt_x_num_unscaled.master=DM(initial_solution*array(mpc.opt_x_scaling))
                for name,value in saved.items():setattr(mpc,name,value)
                mpc.calculate_aux_num()
        state['events'].append(dict(attempts=records,recovered=not records[0]['accepted'] and records[-1]['accepted'],
            final_success=bool(mpc.solver_stats['success']),restored_original=not records[-1]['accepted']))
    mpc.solve=solve
    return state
