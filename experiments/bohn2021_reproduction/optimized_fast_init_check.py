"""Exact equivalence and runtime comparison of initial-guess construction."""
import copy
import json
import time
import numpy as np
from runtime import ART
from optimized_runtime import install_terminal,make_env
from run import write

install_terminal('vehicle')
import casadi as ca
env=make_env('vehicle',111)
env.reset(**copy.deepcopy(json.loads((ART/'configs/vehicle_validation_bank.json').read_text())['cases'][0]))
mpc=env.control_system.controller.mpc
npnt=len(mpc.opt_x_num['_x',0,0])
xi=np.array([[mpc.opt_x_num.f['_x',k,0,j] for j in range(npnt)] for k in range(51)])
ui=np.array([mpc.opt_x_num.f['_u',k,0] for k in range(50)])
timings=[]
for h in [1,5,10,25,50]:
    for mode in [0,-1,1]:
        initial=np.array([.123,2.345,-6.789]);theta,x,y=initial
        start=time.perf_counter();mpc.opt_x_num.master=ca.DM.zeros(*mpc.opt_x_num.shape)
        for k in range(51):
            for j in range(npnt):mpc.opt_x_num['_x',k,0,j]=np.array([theta,x,y])
            if k<50:
                speed=3. if k<h else 0.;turn=mode*2. if k<min(h,8) else 0.
                mpc.opt_x_num['_u',k,0]=np.array([turn,speed])
                x+=.1*speed*np.cos(theta+.05*turn);y+=.1*speed*np.sin(theta+.05*turn);theta+=.1*turn
        expected=np.array(mpc.opt_x_num.cat).ravel();slow=time.perf_counter()-start
        start=time.perf_counter();arr=np.zeros(mpc.opt_x_num.shape[0]);theta,x,y=initial
        for k in range(51):
            arr[xi[k]]=np.array([theta,x,y])
            if k<50:
                speed=3. if k<h else 0.;turn=mode*2. if k<min(h,8) else 0.
                arr[ui[k]]=np.array([turn,speed])
                x+=.1*speed*np.cos(theta+.05*turn);y+=.1*speed*np.sin(theta+.05*turn);theta+=.1*turn
        mpc.opt_x_num.master=ca.DM(arr)
        assert np.array_equal(expected,np.array(mpc.opt_x_num.cat).ravel())
        timings.append({'h':h,'mode':mode,'slow_s':slow,'fast_s':time.perf_counter()-start})
write(ART/'results/optimized/initialization_equivalence.json',{'exact':True,'conditions':15,'timings':timings,
    'mean_slow_s':float(np.mean([r['slow_s'] for r in timings])),'mean_fast_s':float(np.mean([r['fast_s'] for r in timings]))})
print(json.dumps({'exact':True,'mean_slow_s':np.mean([r['slow_s'] for r in timings]),'mean_fast_s':np.mean([r['fast_s'] for r in timings])}),flush=True)
