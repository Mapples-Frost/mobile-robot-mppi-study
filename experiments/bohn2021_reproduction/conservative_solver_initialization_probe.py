"""Post-hoc solver-initialization mechanism probe; no policy efficacy claim.

Run only after registered serial timing exits. Reuse exposed vehicle validation
case2 at seed0/1/2, H10/25. Hold NLP parameters fixed while changing only x0.
No retraining, test access, or application of retry controls to the plant.
"""
import copy
import fcntl
import json
import os
import time
from pathlib import Path
from runtime import imports, ART
import numpy as np
from conservative_iteration import OUT, verify
from conservative_canonical_reset import make_env
from conservative_iteration_timing import idle
from branch_calibration_run import meter, observed_step
from relative_policy_features import context
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read, digest
from run import write, weights_hash


def numeric(value):
    return np.asarray(value.cat if hasattr(value,'cat') else value,dtype=float)


def main():
    verify();idle()
    assert read(OUT/'validation_finish_status.json').get('complete'),'Do not overlap serial timing'
    dest=OUT/'solver_initialization_probe';dest.mkdir(exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    bank=OUT/'banks/vehicle_validation.json'
    files=[Path(__file__).resolve(),bank,
        ART/'sources/do-mpc-horizon/do_mpc/optimizer.py',ART/'sources/gym-horizon/gym_let_mpc/controllers.py']
    files += [model_dir('vehicle','fixed',seed)/'model.zip' for seed in range(3)]
    spec=dict(design='Post-hoc mechanistic probe on exposed validation case2; all3 independent terminal seeds, H10 and H25 at first scored step.',
        hypothesis='Failed primal iterate reused as warm start may sustain solver failure; test same NLP parameters with alternative initial guesses.',
        guesses=['original_initial','returned_solution','constant_current_state_zero_input','constant_current_state_previous_input'],
        measurements=['success','return_status','iterations','objective','max_constraint_residual','max_bound_residual','first_control','solver_elapsed_s'],
        criterion='Report every result. Recovery means success=True and primal constraint/bound residual<=1e-5; this is not a control or safety guarantee.',
        controls='Same opt_p_num, bounds and solver. Original-initial repeat must reproduce recorded x vector within1e-10 and same status.',
        selection='Case2 identified after validation because seed2 first H10 failed. Conditional diagnosis only; no population efficacy claim.',
        budget='3 environments,6 resets with warmup,6 plant steps,24 extra raw NLP solves; constructor initialization separately disclosed. Retries never advance plant.',
        limitations='First-step case2 only; does not explain no-solver-failure degradation, test long-run recovery or certify optimality. New methods need new validation/test.',
        hashes={str(p):digest(p) for p in files},test_access=False)
    reg=dest/'protocol.json'
    if reg.exists():assert read(reg)==spec
    else:write(reg,spec)
    if (dest/'completed.json').exists():
        done=read(dest/'completed.json')
        for p,h in done['hashes'].items():assert digest(Path(p))==h
        print('Completed probe verified; no repeat solves');return
    # A partially executed raw-solver experiment must be reviewed before resume.
    assert not (dest/'attempts.json').exists(),'Interrupted probe: inspect saved results; do not silently repeat'
    counts=dict(environment_constructions=0,raw_solver_attempts=0,raw_solver_completed=0,plant_steps=0,resets=0)
    write(dest/'attempts.json',counts)
    _,SAC,_=imports();case=read(bank)['cases'][2];rows=[];hashes={str(reg):digest(reg)}
    for seed in range(3):
        counts['environment_constructions']+=1;write(dest/'attempts.json',counts)
        env=make_env('vehicle',seed,aligned=True,scaled_obs=True);meter(env,dest)
        source=model_dir('vehicle','fixed',seed);model=SAC.load(str(source/'model.zip'))
        assert weights_hash(model)==read(source/'completed.json')['final_hash']
        env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
        ctrl=env.control_system.controller;mpc=ctrl.mpc
        for h in (10,25):
            counts['resets']+=1;write(dest/'attempts.json',counts);env.reset(**copy.deepcopy(case))
            ctx=context(env,'vehicle');original=mpc.solve;captured={}
            def capture():
                captured['initial']=np.asarray(mpc.opt_x_num.cat).copy()
                captured['parameters']=np.asarray(mpc.opt_p_num.cat).copy()
                return original()
            mpc.solve=capture
            counts['plant_steps']+=1;write(dest/'attempts.json',counts)
            _,_,trace=observed_step(env,'vehicle',h,case,0);mpc.solve=original
            returned=np.asarray(mpc.opt_x_num.cat).copy();original_stats=copy.deepcopy(mpc.solver_stats)
            primary=OUT/'evaluations/validation/vehicle'/('primary_h25_s%d'%seed)/'trace_02.json'
            assert ctx['state']==read(primary)[0]['previous_state']
            if h==25:assert trace=={k:v for k,v in read(primary)[0].items() if k!='policy_context'}
            if seed==2 and h==10:
                reference=OUT/'evaluations/validation/vehicle/round0_h0_s2/trace_02.json'
                assert trace=={k:v for k,v in read(reference)[0].items() if k!='policy_context'}
                hashes[str(reference)]=digest(reference)
            hashes[str(primary)]=digest(primary)
            p=mpc.opt_p(captured['parameters'])
            guesses=dict(original_initial=captured['initial'],returned_solution=returned)
            for name,previous in [('constant_current_state_zero_input',False),('constant_current_state_previous_input',True)]:
                guess=mpc.opt_x(0)
                guess['_x']=p['_x0']/mpc._x_scaling
                guess['_u']=p['_u_prev']/mpc._u_scaling if previous else 0
                guesses[name]=np.asarray(guess.cat).copy()
            raw=dict(parameters=captured['parameters'],recorded_solution=returned,
                lbx=numeric(mpc.lb_opt_x),ubx=numeric(mpc.ub_opt_x),lbg=numeric(mpc.cons_lb),ubg=numeric(mpc.cons_ub))
            local=[]
            for name,guess in guesses.items():
                counts['raw_solver_attempts']+=1;write(dest/'attempts.json',counts)
                started=time.perf_counter()
                result=mpc.S(x0=guess,lbx=mpc.lb_opt_x,ubx=mpc.ub_opt_x,lbg=mpc.cons_lb,ubg=mpc.cons_ub,p=p)
                elapsed=time.perf_counter()-started;stats=mpc.S.stats()
                counts['raw_solver_completed']+=1;write(dest/'attempts.json',counts)
                x=np.asarray(result['x']);g=np.asarray(result['g']);f=float(result['f'])
                residual=lambda values,lo,hi:float(max(0.,np.max(numeric(lo)-values),np.max(values-numeric(hi))))
                gc=residual(g,mpc.cons_lb,mpc.cons_ub);xc=residual(x,mpc.lb_opt_x,mpc.ub_opt_x)
                assert np.isfinite(x).all() and np.isfinite(g).all() and np.isfinite([f,gc,xc]).all()
                if name=='original_initial':
                    np.testing.assert_allclose(x,returned,rtol=0,atol=1e-10)
                    assert stats['success']==original_stats['success']
                control=np.asarray(mpc.opt_x(x)['_u',0,0]*mpc._u_scaling).reshape(-1).tolist()
                row=dict(seed=seed,h=h,guess=name,success=bool(stats['success']),return_status=stats.get('return_status'),
                    iterations=int(stats.get('iter_count',-1)),objective=f,max_constraint_residual=gc,max_bound_residual=xc,
                    first_control=control,solver_elapsed_s=elapsed,primal_recovery=bool(stats['success']) and max(gc,xc)<=1e-5)
                local.append(row);rows.append(row)
                raw[name+'_initial']=guess;raw[name+'_solution']=x;raw[name+'_constraints']=g
            raw_path=dest/('s%d_h%d_arrays.npz'%(seed,h));np.savez_compressed(str(raw_path),**raw)
            report_path=dest/('s%d_h%d.json'%(seed,h));write(report_path,dict(context=ctx,original_first_step=trace,results=local))
            for path in (raw_path,report_path):hashes[str(path)]=digest(path)
            print(json.dumps(dict(seed=seed,h=h,results=local)),flush=True)
    assert counts['raw_solver_completed']==24 and counts['plant_steps']==6
    write(dest/'completed.json',dict(completed=True,rows=rows,counts=counts,hashes=hashes,test_access=False,efficacy_claim=False))


if __name__=='__main__':main()
