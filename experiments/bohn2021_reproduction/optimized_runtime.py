"""Explicit reconstruction improvements; pinned author repositories stay intact."""
import copy
import types
import time
import numpy as np
from runtime import ART, imports, make_env as base_env


def install_terminal(task):
    """Same affine error features and PSD quadratic in TensorFlow and CasADi."""
    imports()
    import tensorflow as tf
    import casadi as ca
    from stable_baselines.sac.policies import AHMPCPolicy
    from gym_let_mpc.utils import casadiNNVF
    n=3 if task=='vehicle' else 4

    def features(z, cat):
        if task=='vehicle':
            return cat([z[0]/np.pi,(z[1]-z[3])/5.,(z[2]-z[4])/5.])
        return cat([z[0]/10.,(z[1]-z[4])/1.5,z[2]/(np.pi/2),z[3]/5.])

    def tf_value(self,state,reuse=False,scope='mpc_value_fns'):
        with tf.variable_scope(scope,reuse=reuse):
            z=features([state[:,i] for i in range(state.shape[1])],lambda a:tf.stack(a,axis=1))
            with tf.variable_scope('mpc_value_fn'):
                init=np.concatenate([(.1*np.eye(n)).ravel(),np.zeros(n)]).reshape(-1,1).astype(np.float32)
                w=tf.get_variable('kernel',initializer=init)
                b=tf.get_variable('bias',initializer=np.zeros(1,dtype=np.float32))
            L=tf.reshape(w[:n*n],[n,n])
            value=tf.reduce_sum(tf.square(tf.matmul(z,L)),axis=1,keepdims=True)+tf.matmul(z,w[n*n:])+b
            if not reuse:self.mpc_value_fn=value
        self.mpc_vf_w_b=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='model/'+scope)
        return value

    def ca_value(self,state,parameters):
        z=features([v for v in ca.vertsplit(ca.vertcat(state,parameters))],lambda a:ca.vertcat(*a))
        w=ca.SX.sym('ol_weights',n*n+n,1);b=ca.SX.sym('ol_bias',1)
        self.weights=ca.tools.struct_symSX([ca.tools.entry('ol_weights',sym=w)])
        self.biases=ca.tools.struct_symSX([ca.tools.entry('ol_bias',sym=b)])
        value=b[0]+sum(z[i]*w[n*n+i] for i in range(n))
        value+=sum(sum(z[i]*w[i*n+j] for i in range(n))**2 for j in range(n))
        self.weights_num=np.zeros(self.weights.shape);self.biases_num=np.zeros(self.biases.shape)
        self.eval_VF=ca.Function('relative_psd_terminal',[state,parameters,self.weights,self.biases],[value])
    AHMPCPolicy.make_mpc_value_fn=tf_value
    casadiNNVF.create_function=ca_value


def install_solver(env,task):
    """Three deterministic initial guesses for vehicle; one for pendulum."""
    import casadi as ca
    ctrl=env.control_system.controller;mpc=ctrl.mpc
    solve_original=mpc.solve
    npoints=len(mpc.opt_x_num['_x',0,0])
    x_indices=np.array([[mpc.opt_x_num.f['_x',k,0,j] for j in range(npoints)] for k in range(51)])
    u_indices=np.array([mpc.opt_x_num.f['_u',k,0] for k in range(50)])
    def initialize(mode):
        state=np.array(mpc.opt_p_num['_x0']).ravel()
        data=np.zeros(mpc.opt_x_num.shape[0])
        if task=='pendulum':
            data[x_indices]=state
            mpc.opt_x_num.master=ca.DM(data)
            return
        theta,x,y=state
        h=int(ctrl.history['mpc_horizon'][-1])
        for k in range(51):
            data[x_indices[k]]=np.array([theta,x,y])
            if k<50:
                speed=3. if k<h else 0.
                turn=mode*2. if k<min(h,8) else 0.
                data[u_indices[k]]=np.array([turn,speed])
                x+=.1*speed*np.cos(theta+.05*turn);y+=.1*speed*np.sin(theta+.05*turn);theta+=.1*turn
        mpc.opt_x_num.master=ca.DM(data)
    def solve(self):
        started=time.perf_counter();candidates=[]
        for mode in ([0,-1,1] if task=='vehicle' else [0]):
            initialize(mode);solve_original()
            g=np.array(mpc.opt_g_num).ravel();x=np.array(mpc.opt_x_num.cat).ravel()
            residual=max(0.,np.max(np.array(mpc.cons_lb).ravel()-g),np.max(g-np.array(mpc.cons_ub).ravel()),
                np.max(np.array(mpc.lb_opt_x.cat).ravel()-x),np.max(x-np.array(mpc.ub_opt_x.cat).ravel()))
            good=bool(mpc.solver_stats.get('success',False)) and residual<=1e-5
            candidates.append({'mode':mode,'good':good,'objective':float(mpc.opt_f_num),'residual':float(residual),
                'x':ca.DM(mpc.opt_x_num.cat),'g':ca.DM(mpc.opt_g_num),'lam_g':ca.DM(mpc.lam_g_num),
                'lam_x':ca.DM(mpc.lam_x_num),'stats':copy.deepcopy(mpc.solver_stats)})
        feasible=[r for r in candidates if r['good']]
        chosen=min(feasible,key=lambda r:r['objective']) if feasible else min(candidates,key=lambda r:(r['residual'],r['objective']))
        mpc.opt_x_num.master=chosen['x'];mpc.opt_x_num_unscaled.master=chosen['x']*mpc.opt_x_scaling
        mpc.opt_g_num=chosen['g'];mpc.opt_f_num=ca.DM(chosen['objective'])
        mpc.lam_g_num=chosen['lam_g'];mpc.lam_x_num=chosen['lam_x'];mpc.solver_stats=chosen['stats']
        mpc.calculate_aux_num()
        env.optimized_solver_info={'solver_calls':len(candidates),'solver_wall_s':time.perf_counter()-started,
            'selected_initial_guess':chosen['mode'],'selected_objective':chosen['objective'],
            'max_constraint_residual':chosen['residual'],
            'candidate_objectives':[r['objective'] for r in candidates],
            'feasible_candidates':len(feasible)}
    mpc.solve=types.MethodType(solve,mpc)


def make_env(task,seed,fixed_horizon=None):
    from gym import spaces
    env=base_env(task,seed,fixed_horizon,aligned=True,scaled_obs=True)
    ctrl=env.control_system.controller
    obs_original=env.get_observation;step_original=env.step;reset_original=env.reset
    p_original=ctrl._get_p_values
    def parameters(t):
        p=p_original(t)
        h=int(ctrl.history['mpc_horizon'][-1]) if ctrl.history['mpc_horizon'] else 50
        if task=='vehicle':
            p['_p',0,'goal_x']=ctrl._tvp_data['trajectory_x'][h]
            p['_p',0,'goal_y']=ctrl._tvp_data['trajectory_y'][h]
        else:p['_p',0,'pos_r']=ctrl._tvp_data['pos_r'][h]
        return p
    ctrl.mpc.p_fun=parameters
    def ref(name,offset=0):
        clock=env.control_system._step_count+offset
        tvp=env.control_system.tvps[name]
        if len(tvp.values)<=clock:tvp.generate_values(clock+1-len(tvp.values))
        return float(np.asarray(tvp.get_values(clock)).ravel()[0])
    def observation():
        obs=obs_original();state=env.control_system.current_state
        if task=='vehicle':
            forecasts=[]
            for j in range(ctrl.n_objects):
                dist=min(ctrl.get_obj_distance(state,j)/ctrl._max_obj_dist,1.)
                forecasts.extend((np.asarray(ctrl.object_noise_seed[j])*dist/np.array([5.,5.,1.])).tolist())
            extra=forecasts+[(env.trajectory_goal_x-state['x'])/30.,(env.trajectory_goal_y-state['y'])/30.,
                (env.max_steps-env.steps_count)/env.max_steps]
            # MPC-known reference previews; no access to future physical outcomes.
            for k in [10,25,50]:extra.extend([(ref('trajectory_x',k)-state['x'])/15.,(ref('trajectory_y',k)-state['y'])/15.])
        else:
            extra=[ref('pos_r',k) for k in [10,25,50]]+[(env.max_steps-env.steps_count)/env.max_steps]
        return np.concatenate([obs,extra]).astype(np.float32)
    env.get_observation=observation
    n=32 if task=='vehicle' else 9
    env.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(n,),dtype=np.float32)
    install_solver(env,task)
    def reset(**kwargs):
        vf=ctrl.mpc.vf;w,b=copy.deepcopy(vf.weights_num),copy.deepcopy(vf.biases_num)
        vf.weights_num=np.zeros_like(w);vf.biases_num=np.zeros_like(b)
        try:obs=reset_original(**kwargs)
        finally:vf.weights_num=w;vf.biases_num=b
        assert obs.shape==(n,)
        return obs
    def step(action):
        before=env.control_system.get_state_vector(env.control_system.current_state).ravel().copy()
        refs=[ref(v) for v in (['trajectory_x','trajectory_y'] if task=='vehicle' else ['pos_r'])]
        obs,r,done,info=step_original(action)
        following=env.control_system.get_state_vector(env.control_system.current_state).ravel()
        nextrefs=[ref(v) for v in (['trajectory_x','trajectory_y'] if task=='vehicle' else ['pos_r'])]
        info['data']['mpc_state']=np.concatenate([before,refs])
        info['data']['mpc_next_state']=np.concatenate([following,nextrefs])
        info.update(env.optimized_solver_info)
        return obs,r,done,info
    env.reset=reset;env.step=step
    return env
