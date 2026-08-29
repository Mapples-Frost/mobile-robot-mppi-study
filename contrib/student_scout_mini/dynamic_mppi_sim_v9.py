"""Student SLAM/MPPI v9 prototype, isolated from production deployment.

The synchronization fixes path and tuple-shape defects so the prototype can
be reproduced, but does not promote it into the paper or real-robot runtime.
"""

import math,json,time,random,argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

TRACK_WIDTH=0.32; COLLISION_RADIUS=0.25; V_MAX=0.50; OMEGA_MAX=0.80

# SLAM Map Integration (v9)
SLAM_MAP_PATH = Path(__file__).resolve().parent / 'slam_map_u_shape.npy'
SLAM_RES = 0.05
SLAM_ORIGIN = (0.0, 0.0)
_slam_grid = None

def load_slam_map(path=SLAM_MAP_PATH):
    global _slam_grid
    if _slam_grid is not None:
        return _slam_grid
    try:
        m = np.load(path)
        print('Loaded SLAM map: shape=%s obs=%d free=%d unk=%d' % (str(m.shape), (m==1).sum(), (m==0).sum(), (m==2).sum()))
        _slam_grid = m
        return m
    except (FileNotFoundError, OSError, ValueError) as exc:
        print('SLAM map unavailable: %s' % exc)
        _slam_grid = None
        return None

def map_to_scene_cylinders(map_data, bounds, min_cl=30, max_cl=800, margin=0.10):
    if map_data is None:
        return []
    obs = (map_data == 1)
    grid_cx = map_data.shape[1] / 2.0
    grid_cy = map_data.shape[0] / 2.0
    bx_min = int(max(0, (bounds['x_min'] - SLAM_ORIGIN[0]) / SLAM_RES + grid_cx))
    bx_max = int(min(map_data.shape[1], (bounds['x_max'] - SLAM_ORIGIN[0]) / SLAM_RES + grid_cx))
    by_min = int(max(0, (bounds['y_min'] - SLAM_ORIGIN[1]) / SLAM_RES + grid_cy))
    by_max = int(min(map_data.shape[0], (bounds['y_max'] - SLAM_ORIGIN[1]) / SLAM_RES + grid_cy))
    if bx_max <= bx_min or by_max <= by_min:
        return []
    crop = obs[by_min:by_max, bx_min:bx_max]
    labeled = np.zeros(crop.shape, dtype=np.int32)
    lid = 0
    for y in range(crop.shape[0]):
        for x in range(crop.shape[1]):
            if crop[y,x] and labeled[y,x] == 0:
                lid += 1
                stack = [(y,x)]
                labeled[y,x] = lid
                while stack:
                    cy,cx = stack.pop()
                    for ny,nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                        if 0<=ny<crop.shape[0] and 0<=nx<crop.shape[1]:
                            if crop[ny,nx] and labeled[ny,nx]==0:
                                labeled[ny,nx]=lid
                                stack.append((ny,nx))
    cylinders = []
    for idx2 in range(1, lid+1):
        mask = (labeled == idx2)
        size = int(mask.sum())
        if size < min_cl or size > max_cl:
            continue
        ys, xs = np.where(mask)
        wx_min = (xs.min()+bx_min-grid_cx)*SLAM_RES + SLAM_ORIGIN[0]
        wx_max = (xs.max()+bx_min-grid_cx)*SLAM_RES + SLAM_ORIGIN[0]
        wy_min = (ys.min()+by_min-grid_cy)*SLAM_RES + SLAM_ORIGIN[1]
        wy_max = (ys.max()+by_min-grid_cy)*SLAM_RES + SLAM_ORIGIN[1]
        cx = (wx_min+wx_max)/2
        cy = (wy_min+wy_max)/2
        r = max(wx_max-wx_min, wy_max-wy_min)/2 + margin
        cylinders.append((round(cx,3), round(cy,3), round(r,3), True))
    print('  Map cylinders: %d' % len(cylinders))
    return cylinders

def get_map_cylinders(bounds):
    return map_to_scene_cylinders(load_slam_map(), bounds)
V_MIN=0.0; GOAL_THRESHOLD=0.35

@dataclass
class DynamicObstacle:
    x:float; y:float; radius:float; vx:float=0.0; vy:float=0.0; label:str=''
    known_static:bool=False
    def position_at(self,t): return (self.x+self.vx*t,self.y+self.vy*t)
    def clone(self): return DynamicObstacle(self.x,self.y,self.radius,self.vx,self.vy,self.label,self.known_static)

def wrap_angle(t):
    while t>math.pi: t-=2*math.pi
    while t<-math.pi: t+=2*math.pi
    return t

def kinematic_step(state,control,dt):
    x,y,theta=state; v,omega=control
    return x+v*math.cos(theta)*dt, y+v*math.sin(theta)*dt, wrap_angle(theta+omega*dt)

def rollout_sequence(start_state,ctrl_seq,dt):
    traj=[start_state]; s=list(start_state)
    for c in ctrl_seq: s=list(kinematic_step(tuple(s),c,dt)); traj.append(tuple(s))
    return traj

def obstacle_clearance(x,y,ox,oy,orr,rr): return math.hypot(x-ox,y-oy)-(rr+orr)

def dynamic_collision_cost(x,y,obs,robot_r,dt_pred,safety_margin=0.15):
    ox,oy=obs.position_at(0); dx,dy=obs.vx,obs.vy
    rx,ry=x-ox,y-oy; rvx=-dx; rvy=-dy
    threshold=robot_r+obs.radius+safety_margin
    a=rvx*rvx+rvy*rvy; b=2*(rx*rvx+ry*rvy); c=rx*rx+ry*ry-threshold*threshold
    min_dist=float("inf")
    if a>1e-12:
        disc=b*b-4*a*c
        if disc>=0:
            t1=(-b-math.sqrt(disc))/(2*a); t2=(-b+math.sqrt(disc))/(2*a)
            for tt in [t1,t2]:
                if 0<=tt<=dt_pred: return 5000.0
    else:
        min_dist=math.hypot(rx,ry)
    if min_dist<threshold: return 5000.0*(1-min_dist/threshold)
    gap=min_dist-threshold
    return 50.0*(1.0-gap)**2 if gap<1.0 else 0.0

def trajectory_cost(trajectory,ctrl_seq,goal,static_obs,dyn_obs,robot_r,bounds,dyn_dt_pred=1.0,cost_fn_type='dynamic'):
    gx,gy=goal; goal_cost=smooth=0.0; static_safety=0.0; dyn_safety=0.0
    collided=False; H=max(len(ctrl_seq),1)
    total_fwd=sum(max(0,ctrl_seq[i][0]) for i in range(len(ctrl_seq)))
    # Time pressure: penalize slow progress
    time_penalty=30.0*max(0,H*0.1-total_fwd/0.30)/H if total_fwd>0 else 0.0
    for i,(x,y,theta) in enumerate(trajectory[1:]):
        gd=math.hypot(gx-x,gy-y)
        dh=math.atan2(gy-y,gx-x)
        heading_err=abs(wrap_angle(dh-theta))
        # Goal attraction (always active, strong)
        goal_cost -= 60.0*max(0,10.0-gd)
        goal_cost += 0.5*heading_err
        v=ctrl_seq[i][0]
        # Strong velocity incentive
        goal_cost -= 15.0*v
        if cost_fn_type=='dynamic':
            for obs in dyn_obs:
                ocx,ocy=obs.position_at(dyn_dt_pred)
                oc_dist=math.hypot(ocx-x,ocy-y)
                # Directional repulsion: only when obstacle is approaching
                to_obs_dx=ocx-x; to_obs_dy=ocy-y
                to_obs_norm=math.hypot(to_obs_dx,to_obs_dy)
                if to_obs_norm>0.01 and v>0.01:
                    cos_angle=(v*math.cos(theta)*to_obs_dx+v*math.sin(theta)*to_obs_dy)/(v*to_obs_norm)
                else:
                    cos_angle=0.0
                # Skip aggressive dynamic cost for known-static map obstacles
                if getattr(obs, 'known_static', False):
                    ocx2,ocy2=obs.position_at(dyn_dt_pred)
                    oc_dist2=math.hypot(ocx2-x,ocy2-y)
                    if oc_dist2<3.0 and v>0.01:
                        to_dx=ocx2-x; to_dy=ocy2-y
                        to_norm=math.hypot(to_dx,to_dy)
                        cos_a=(v*math.cos(theta)*to_dx+v*math.sin(theta)*to_dy)/(v*to_norm) if to_norm>0.01 else 0.0
                        if oc_dist2<3.0 and cos_a>0.3:
                            dyn_safety += 10.0*max(0,1.0-oc_dist2/3.0)**2
                    continue
                # Only penalize when obstacle is close AND approaching
                if oc_dist<2.5 and cos_angle>0.5:
                    dyn_safety += 50.0*max(0,1.0-oc_dist)**2
                ttc=dynamic_collision_cost(x,y,obs,robot_r,dyn_dt_pred)
                dyn_safety += ttc*0.0003
        for ox,oy,orr,*_ in static_obs:
            cl=obstacle_clearance(x,y,ox,oy,orr,robot_r)
            if cl<=0: collided=True; static_safety+=100000.0; break
            if cl<1.0: static_safety+=10.0*(1.0-cl)**2
        ex_min=bounds['x_min']+robot_r; ex_max=bounds['x_max']-robot_r
        ey_min=bounds['y_min']+robot_r; ey_max=bounds['y_max']-robot_r
        if not (ex_min<=x<=ex_max and ey_min<=y<=ey_max):
            collided=True; margin=min(x-ex_min,ex_max-x,y-ey_min,ey_max-y)
            if margin<0: static_safety+=50000.0*margin**2
    fx,fy,ft=trajectory[-1]; fgd=math.hypot(gx-fx,gy-fy)
    terminal=8.0*fgd+0.5*abs(wrap_angle(math.atan2(gy-fy,gx-fx)-ft))+time_penalty
    for i in range(1,len(ctrl_seq)):
        v0,w0=ctrl_seq[i-1]; v1,w1=ctrl_seq[i]
        smooth+=0.02*((v1-v0)**2+0.1*(w1-w0)**2)
    total=(goal_cost+static_safety+dyn_safety+smooth)/H+terminal
    return total,collided

def build_goal_warm_start(state,goal,horizon,dt,v_max,omega_max,prefix_steps=8):
    seq=[]; s=list(state)
    for _ in range(min(horizon,prefix_steps)):
        gx,gy=goal; dh=math.atan2(gy-s[1],gx-s[0]); err=wrap_angle(dh-s[2])
        dist=math.hypot(gx-s[0],gy-s[1])
        if err>0.3: v=0.3*v_max
        elif err>0.1: v=0.6*v_max
        else: v=1.0*v_max
        w=err/max(dt,1e-8)
        w=max(-omega_max,min(omega_max,w))
        seq.append((v,w)); s=list(kinematic_step(tuple(s),(v,w),dt))
    while len(seq)<horizon: seq.append((v_max*0.5,0.0))
    return seq

def compute_weights(costs,temp):
    min_c=min(costs); unorm=[math.exp(-(c-min_c)/max(temp,1e-8)) for c in costs]
    s=sum(unorm)
    if s<1e-12: return [1.0/len(costs)]*len(costs)
    return [w/s for w in unorm]

def weighted_update(seqs,weights):
    H=len(seqs[0]); updated=[]
    for t in range(H):
        wv=sum(w*seq[t][0] for seq,w in zip(seqs,weights))
        wo=sum(w*seq[t][1] for seq,w in zip(seqs,weights))
        updated.append((wv,wo))
    return updated

def shift_seq(seq,tail=(0.0,0.0)):
    if not seq: return []
    return seq[1:]+[tail]

def make_safe_control(state,proposed,dt,bounds,robot_r,dyn_obs=None):
    vx,om=proposed
    for scale in [1.0,0.75,0.5,0.25,0.1,0.0]:
        c=(scale*vx,om); nx,ny,ntheta=kinematic_step(state,c,dt)
        ex_min=bounds['x_min']+robot_r; ex_max=bounds['x_max']-robot_r
        ey_min=bounds['y_min']+robot_r; ey_max=bounds['y_max']-robot_r
        if ex_min<=nx<=ex_max and ey_min<=ny<=ey_max: return c,(scale<1.0)
    return (0.0,0.0),True

def sample_control_sequences(nominal,state,dt,static_obs,dyn_obs,bounds,robot_r,N,v_std,omega_std,v_min,v_max,omega_max,anisotropic=False,sdf_dist=1.2,sigma_par=0.08,sigma_perp=0.03,cost_fn=None):
    nom_traj=rollout_sequence(state,nominal,dt); samples=[]
    for _ in range(N):
        seq=[]
        for t,(nv,now) in enumerate(nominal):
            xt,yt,tht=nom_traj[t]
            if anisotropic and cost_fn is not None:
                base_cost=cost_fn(nom_traj,nominal,(0,0),static_obs,dyn_obs,robot_r,bounds,0)[0]
                alt_seq=nominal[:t]+[(nv+v_std,now)]+nominal[t+1:]
                alt_traj=rollout_sequence(state,alt_seq,dt)
                alt_cost=cost_fn(alt_traj,alt_seq,(0,0),static_obs,dyn_obs,robot_r,bounds,0)[0]
                alt_seq2=nominal[:t]+[(nv,now+omega_std)]+nominal[t+1:]
                alt_traj2=rollout_sequence(state,alt_seq2,dt)
                alt_cost2=cost_fn(alt_traj2,alt_seq2,(0,0),static_obs,dyn_obs,robot_r,bounds,0)[0]
                dv_grad=(alt_cost-base_cost)/v_std if v_std>0 else 0
                dw_grad=(alt_cost2-base_cost)/omega_std if omega_std>0 else 0
                sv=nv-0.3*dv_grad+np.random.randn()*v_std*0.5
                sw=now-0.3*dw_grad+np.random.randn()*omega_std*0.5
            elif anisotropic:
                min_cl=float('inf'); best_grad=np.array([1.0,0.0])
                for ox,oy,orr,*_ in static_obs:
                    cl=obstacle_clearance(xt,yt,ox,oy,orr,robot_r)
                    if cl<min_cl: min_cl=cl; d=math.hypot(xt-ox,yt-oy)
                    best_grad=np.array([(xt-ox)/max(d,1e-8),(yt-oy)/max(d,1e-8)])
                ex_min=bounds['x_min']+robot_r; ex_max=bounds['x_max']-robot_r
                ey_min=bounds['y_min']+robot_r; ey_max=bounds['y_max']-robot_r
                for cl_val,grad in [(xt-ex_min,np.array([1.,0.])),(ex_max-xt,np.array([-1.,0.])),(yt-ey_min,np.array([0.,1.])),(ey_max-yt,np.array([0.,-1.]))]:
                    if cl_val<min_cl: min_cl=cl_val; best_grad=grad
                for obs in dyn_obs:
                    ocx,ocy=obs.position_at(t*dt)
                    cl=obstacle_clearance(xt,yt,ocx,ocy,obs.radius,robot_r)
                    if cl<min_cl: min_cl=cl; d=math.hypot(xt-ocx,yt-ocy)
                    best_grad=np.array([(xt-ocx)/max(d,1e-8),(yt-ocy)/max(d,1e-8)])
                if min_cl>sdf_dist:
                    sv=np.random.randn()*v_std+nv; sw=np.random.randn()*omega_std+now
                else:
                    n=best_grad/max(np.linalg.norm(best_grad),1e-8)
                    t_dir=np.array([-n[1],n[0]])
                    Sigma=sigma_par*np.outer(t_dir,t_dir)+sigma_perp*np.outer(n,n)
                    eps=np.random.multivariate_normal(np.zeros(2),Sigma)
                    u_nom=np.array([nv*math.cos(tht),nv*math.sin(tht)])
                    u_samp=u_nom+eps; sv=float(np.clip(np.linalg.norm(u_samp),v_min,v_max))
                    if sv<1e-8: sw=0.0
                    else:
                        dh2=math.atan2(u_samp[1],u_samp[0]); sw=wrap_angle(dh2-tht)/max(dt,1e-8)
            else:
                sv=np.random.randn()*v_std+nv; sw=np.random.randn()*omega_std+now
            sv=max(v_min,min(v_max,sv)); sw=max(-omega_max,min(omega_max,sw))
            seq.append((sv,sw))
        samples.append(seq)
    if samples: samples[0]=nominal.copy()
    return samples

@dataclass
class MPPIController:
    horizon:int; num_samples:int; temperature:float; dt:float
    v_std:float; omega_std:float; v_min:float=0.0; v_max:float=0.5; omega_max:float=0.8
    use_warm_start:bool=True; anisotropic:bool=False; sdf_dist:float=1.2
    sigma_par:float=0.08; sigma_perp:float=0.03
    dyn_dt_pred:float=1.5; use_dynamic_cost:bool=True
    def __init__(self,horizon=12,num_samples=100,temperature=2.0,dt=0.1,
                 v_std=0.35,omega_std=0.80,v_min=0.0,v_max=0.50,omega_max=0.80,
                 use_warm_start=True,anisotropic=False,sdf_dist=1.2,
                 sigma_par=0.08,sigma_perp=0.03,dyn_dt_pred=1.5,
                 use_dynamic_cost=True):
        self.horizon=horizon; self.num_samples=num_samples
        self.temperature=temperature; self.dt=dt
        self.v_std=v_std; self.omega_std=omega_std
        self.v_min=v_min; self.v_max=v_max; self.omega_max=omega_max
        self.use_warm_start=use_warm_start; self.anisotropic=anisotropic
        self.sdf_dist=sdf_dist; self.sigma_par=sigma_par; self.sigma_perp=sigma_perp
        self.dyn_dt_pred=dyn_dt_pred; self.use_dynamic_cost=use_dynamic_cost
        self.last_nominal=None
    def reset(self): self.last_nominal=None
    def step(self,state,goal,static_obs,dyn_obs,bounds,robot_r):
        ctrl=self.plan(state,goal,static_obs,dyn_obs,bounds,robot_r)
        pv,pw=ctrl[0]
        fv,fw=make_safe_control(state,(pv,pw),self.dt,bounds,robot_r)[0]
        return (fv,fw),None
    def plan(self,state,goal,static_obs,dyn_obs,bounds,robot_r):
        if self.use_warm_start and getattr(self,'last_nominal',None) is not None:
            nominal=shift_seq(self.last_nominal,tail=(self.v_max*0.3,0.0))
        else:
            nominal=build_goal_warm_start(state,goal,self.horizon,self.dt,self.v_max,self.omega_max)
        self.last_nominal=nominal
        def cf(traj,ctrl,g,so,do,rr,b,dp): return trajectory_cost(traj,ctrl,g,so,do,rr,b,self.dyn_dt_pred,'dynamic' if self.use_dynamic_cost else 'static')
        sampled=sample_control_sequences(nominal,state,self.dt,static_obs,dyn_obs,bounds,robot_r,self.num_samples,self.v_std,self.omega_std,self.v_min,self.v_max,self.omega_max,self.anisotropic,self.sdf_dist,self.sigma_par,self.sigma_perp,cf)
        trajectories=[rollout_sequence(state,seq,self.dt) for seq in sampled]
        costs=[]
        for traj,seq in zip(trajectories,sampled):
            c,_=trajectory_cost(traj,seq,goal,static_obs,dyn_obs,robot_r,bounds,self.dyn_dt_pred,'dynamic' if self.use_dynamic_cost else 'static')
            costs.append(c)
        weights=compute_weights(costs,self.temperature)
        updated=weighted_update(sampled,weights); self.last_nominal=updated
        best_idx=min(range(len(costs)),key=lambda i:costs[i])
        self.last_trajectory=trajectories[best_idx]
        if len(updated)==0:
            return (0.0,0.0),costs,best_idx
        control,_=make_safe_control(state,updated[0],self.dt,bounds,robot_r)
        return control,costs,best_idx
    def reset(self): self.last_nominal=None; self.last_trajectory=None

DYNAMIC_SCENES={
    'crossing_obstacle':{
        'start':(0.0,0.0,0.0),'goal':(6.0,0.0),
        'static_obs':[(2.2,0.9,0.55,False),(2.2,-0.9,0.55,False),(4.2,0.9,0.55,False),(4.2,-0.9,0.55,False)],
        'dyn_obs':[DynamicObstacle(3.0,1.8,0.3,vx=0.0,vy=-0.3,label='C1'),DynamicObstacle(3.0,-1.8,0.3,vx=0.0,vy=0.3,label='C2')],
        'bounds':{'x_min':-0.5,'x_max':6.5,'y_min':-2.0,'y_max':2.0},'use_map':True},
    'chasing_obstacle':{
        'start':(0.0,0.0,0.0),'goal':(6.0,0.0),
        'static_obs':[(1.5,0.8,0.4,False),(1.5,-0.8,0.4,False),(4.5,0.8,0.4,False),(4.5,-0.8,0.4,False)],
        'dyn_obs':[DynamicObstacle(3.5,0.0,0.4,vx=0.25,vy=0.0,label='P1'),DynamicObstacle(5.0,0.5,0.3,vx=-0.15,vy=0.0,label='P2')],
        'bounds':{'x_min':-0.5,'x_max':6.5,'y_min':-1.5,'y_max':1.5},'use_map':True},
    'perpendicular_crossing':{
        'start':(0.0,0.0,0.0),'goal':(2.0,5.0),
        'static_obs':[],
        'dyn_obs':[DynamicObstacle(0.0,1.5,0.35,vx=0.4,vy=0.0,label='X1'),DynamicObstacle(0.0,3.5,0.35,vx=-0.4,vy=0.0,label='X2'),DynamicObstacle(2.0,2.5,0.3,vx=0.0,vy=0.35,label='X3')],
        'bounds':{'x_min':-1.5,'x_max':2.0,'y_min':-0.5,'y_max':5.5},'use_map':False},
    'dense_dynamic':{
        'start':(0.0,0.0,0.0),'goal':(6.0,0.0),
        'static_obs':[(1.0,0.5,0.3,False),(2.1,-0.3,0.3,False),(3.2,0.5,0.3,False),(4.3,-0.4,0.3,False),(5.1,0.4,0.3,False)],
        'dyn_obs':[DynamicObstacle(2.5,0.0,0.35,vx=0.2,vy=0.15,label='D1'),DynamicObstacle(4.0,1.0,0.3,vx=-0.15,vy=-0.2,label='D2'),DynamicObstacle(3.5,-1.0,0.3,vx=0.1,vy=-0.25,label='D3'),DynamicObstacle(5.0,0.3,0.25,vx=-0.2,vy=0.1,label='D4')],
        'bounds':{'x_min':-0.5,'x_max':6.5,'y_min':-1.5,'y_max':1.5},'use_map':True},
    'zigzag_obstacles':{
        'start':(0.0,0.0,0.0),'goal':(6.0,0.0),
        'static_obs':[],
        'dyn_obs':[DynamicObstacle(1.5,1.0,0.3,vx=0.0,vy=-0.4,label='Z1'),DynamicObstacle(3.0,-1.0,0.3,vx=0.0,vy=0.4,label='Z2'),DynamicObstacle(4.5,1.0,0.3,vx=0.0,vy=-0.4,label='Z3'),DynamicObstacle(2.2,0.0,0.35,vx=0.3,vy=0.0,label='Z4'),DynamicObstacle(5.0,0.0,0.35,vx=-0.3,vy=0.0,label='Z5')],
        'bounds':{'x_min':-0.5,'x_max':6.5,'y_min':-2.0,'y_max':2.0},'use_map':True}
}

def run_episode(controller,scene,seed=0,max_steps=400):
    random.seed(seed); np.random.seed(seed); controller.reset()
    state=list(scene['start'])
    if scene.get('dyn_obs',[]):
        state[2]=wrap_angle(state[2]-0.15)
    else:
        state[2]=wrap_angle(state[2]+random.gauss(0,0.01))
    state=tuple(state)
    trajectory=[state]; metrics={'min_clearance':float('inf'),'min_dyn_clearance':float('inf'),'planning_times':[],'costs':[],'controls':[],'obstacle_proximity':[],'react_interventions':0,'react_reasons':[]}
    goal=scene['goal']; static_obs=scene['static_obs']; dyn_obs=[o.clone() for o in scene['dyn_obs']]; bounds=scene['bounds']; robot_r=COLLISION_RADIUS
    use_map = scene.get('use_map', False)
    if use_map:
        map_obs = get_map_cylinders(bounds)
        static_obs = list(static_obs) + list(map_obs)
    for step in range(max_steps):
        t0=time.perf_counter()
        control,reason=controller.step(state,goal,static_obs,dyn_obs,bounds,robot_r)
        if reason is not None:
            metrics['react_interventions']+=1
            metrics['react_reasons'].append(reason)
        t1=time.perf_counter(); metrics['planning_times'].append((t1-t0)*1000); metrics['costs'].append(0.0)
        for obs in dyn_obs:
            obs.x+=obs.vx*controller.dt; obs.y+=obs.vy*controller.dt
            by_min=bounds['y_min']+obs.radius+0.1; by_max=bounds['y_max']-obs.radius-0.1
            if obs.y<by_min or obs.y>by_max: obs.vy=-obs.vy; obs.y=max(by_min,min(by_max,obs.y))
            bx_min=bounds['x_min']+obs.radius+0.1; bx_max=bounds['x_max']-obs.radius-0.1
            if obs.x<bx_min or obs.x>bx_max: obs.vx=-obs.vx; obs.x=max(bx_min,min(bx_max,obs.x))
        nx,ny,ntheta=kinematic_step(state,control,controller.dt)
        coll=False
        for ox,oy,orr,*_ in static_obs:
            if math.hypot(nx-ox,ny-oy)<robot_r+orr: coll=True; break
        if not coll:
            for obs in dyn_obs:
                if math.hypot(nx-obs.x,ny-obs.y)<robot_r+obs.radius: coll=True; break
        if coll:
            return {'success':False,'collision':True,'timeout':False,'steps':step,'trajectory':trajectory,'metrics':metrics}
        ex_min=bounds['x_min']+robot_r; ex_max=bounds['x_max']-robot_r
        ey_min=bounds['y_min']+robot_r; ey_max=bounds['y_max']-robot_r
        if not (ex_min<=nx<=ex_max and ey_min<=ny<=ey_max):
            return {'success':False,'collision':True,'timeout':False,'steps':step,'trajectory':trajectory,'metrics':metrics}
        state=(nx,ny,ntheta); trajectory.append(state)
        for ox,oy,orr,*_ in static_obs:
            cl=math.hypot(nx-ox,ny-oy)-robot_r-orr; metrics['min_clearance']=min(metrics['min_clearance'],cl)
        for obs in dyn_obs:
            cl=math.hypot(nx-obs.x,ny-obs.y)-robot_r-obs.radius
            metrics['min_dyn_clearance']=min(metrics['min_dyn_clearance'],cl)
            metrics['obstacle_proximity'].append(math.hypot(nx-obs.x,ny-obs.y))
        if math.hypot(nx-goal[0],ny-goal[1])<GOAL_THRESHOLD:
            return {'success':True,'collision':False,'timeout':False,'steps':step,'trajectory':trajectory,'metrics':metrics}
    return {'success':False,'collision':False,'timeout':True,'steps':max_steps,'trajectory':trajectory,'metrics':metrics}

def plot_result(traj,scene,result,params,save_path=None):
    fig,ax=plt.subplots(figsize=(12,6))
    gx,gy=scene['goal']; b=scene['bounds']
    ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],facecolor='#f0f0f0',edgecolor='gray',lw=1.5,zorder=0))
    for ox,oy,orr,*metadata in scene['static_obs']:
        known = bool(metadata[0]) if metadata else False
        color = '#7f8c8d' if known else '#e74c3c'
        alpha = 0.4 if known else 0.6
        ax.add_patch(Circle((ox,oy),orr,facecolor=color,alpha=alpha,zorder=2))
        ax.add_patch(Circle((ox,oy),orr,fill=False,edgecolor=color,lw=1.5,zorder=3))
    for obs in scene['dyn_obs']:
        ax.add_patch(Circle((obs.x,obs.y),obs.radius,facecolor='#3498db',alpha=0.7,zorder=2))
        ax.add_patch(Circle((obs.x,obs.y),obs.radius,fill=False,edgecolor='#2980b9',lw=1.5,zorder=3))
        ax.arrow(obs.x,obs.y,obs.vx*1.5,obs.vy*1.5,head_width=0.1,head_length=0.05,facecolor='#3498db',edgecolor='#2980b9',lw=1,alpha=0.7)
        ax.text(obs.x+0.15,obs.y+0.15,obs.label,fontsize=8,color='#2980b9')
    tx=[p[0] for p in traj]; ty=[p[1] for p in traj]
    ax.plot(tx,ty,'b-',lw=2,label='MPPI Trajectory',zorder=5)
    ax.plot(tx[0],ty[0],'go',ms=10,label='Start',zorder=6)
    ax.plot(gx,gy,'bs',ms=10,label='Goal',zorder=6)
    fx,fy,ftheta=traj[-1]
    ax.plot(fx,fy,'ro',ms=8,zorder=6)
    ax.arrow(fx,fy,0.2*math.cos(ftheta),0.2*math.sin(ftheta),head_width=0.08,head_length=0.06,facecolor='red',edgecolor='red',lw=1.5,zorder=7)
    ax.add_patch(Circle((fx,fy),COLLISION_RADIUS,fill=False,linestyle='--',edgecolor='red',lw=1.5,zorder=5))
    ax.set_xlim(b['x_min'],b['x_max']); ax.set_ylim(b['y_min'],b['y_max'])
    ax.set_aspect('equal'); ax.grid(True,alpha=0.3); ax.legend(loc='upper left',fontsize=9)
    s='SUCCESS' if result.get('success') else ('COLLISION' if result.get('collision') else 'TIMEOUT')
    ax.set_title('MPPI Dynamic v9 | %s | h=%d N=%d T=%.1f | steps=%d' % (s,params['horizon'],params['num_samples'],params['temperature'],result.get('steps',None)))
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight'); plt.close()
        print('  Saved: '+str(save_path))
    else: plt.show()

def main():
    global SLAM_MAP_PATH
    parser=argparse.ArgumentParser(description='Student MPPI Dynamic Obstacle Simulation v9')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--scene',default='crossing_obstacle',choices=list(DYNAMIC_SCENES.keys()))
    parser.add_argument('--episodes',type=int,default=15)
    parser.add_argument('--slam-map',type=Path,default=SLAM_MAP_PATH)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    args=parser.parse_args()
    SLAM_MAP_PATH = args.slam_map
    results_dir=args.output_dir
    results_dir.mkdir(parents=True,exist_ok=True)
    if args.demo:
        scene=DYNAMIC_SCENES[args.scene]
        params={'horizon':15,'num_samples':120,'temperature':2.0,'dt':0.1,'v_std':0.30,'omega_std':0.50,'use_warm_start':True,'anisotropic':False,'sdf_dist':1.2,'sigma_par':0.08,'sigma_perp':0.03,'dyn_dt_pred':2.0,'use_dynamic_cost':True}
        ctrl=MPPIController(**params)
        result=run_episode(ctrl,scene,seed=0)
        sp=results_dir/('demo_'+args.scene+'_v9.png')
        plot_result(result['trajectory'],scene,result,params,save_path=sp)
        s='SUCCESS' if result['success'] else ('COLLISION' if result['collision'] else 'TIMEOUT')
        print('Demo v9: '+s+' steps='+str(result['steps']))

if __name__=='__main__':
    main()
