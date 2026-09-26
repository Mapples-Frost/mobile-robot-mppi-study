"""Low-dimensional horizon policy fitted by complete-episode training search."""
import math
import numpy as np

BASE={'vehicle':25,'pendulum':30}
PROFILES={
    'vehicle':[
        dict(tracking=.05,heading=.03,yaw_rate=.1,min_speed=2.5,clearance=1.5),
        dict(tracking=.15,heading=.1,yaw_rate=.5,min_speed=2.,clearance=1.),
        dict(tracking=.4,heading=.25,yaw_rate=1.,min_speed=1.,clearance=.5)],
    'pendulum':[
        dict(position=.025,velocity=.05,angle=.015,angular_velocity=.05,input=.5),
        dict(position=.1,velocity=.25,angle=.06,angular_velocity=.25,input=2.),
        dict(position=.25,velocity=.75,angle=.15,angular_velocity=.75,input=4.)]}


def candidates(task):
    guards=[5,15,30] if task=='vehicle' else [5,10,20]
    return [dict(id='h%d_p%d_g%d'%(h,i,g),task=task,short_h=h,profile=i,guard=g)
            for h in (5,10,15,20) for i in range(3) for g in guards]


def decide(policy,ctx):
    task=policy['task'];base=BASE[task]
    if policy['id']=='fixed':return base,dict(use_short=False,reason='fixed')
    state=ctx['state'];preview=ctx['previews'];p=PROFILES[task][policy['profile']];guard=policy['guard']
    if ctx['elapsed']<5:return base,dict(use_short=False,reason='initial_five_steps')
    if task=='pendulum':
        ref=np.asarray(preview['pos_r'],float)
        values=dict(position=abs(state['pos']-ref[0]),velocity=abs(state['v']),angle=abs(state['theta']),
            angular_velocity=abs(state['omega']),input=abs(float(np.asarray(ctx['previous_input']['u1']).ravel()[0])),
            reference_change=float(np.max(abs(ref[:guard+1]-ref[0]))))
        good=all(values[k]<=p[k] for k in p) and values['reference_change']<=1e-8
    else:
        x=np.asarray(preview['trajectory_x'],float);y=np.asarray(preview['trajectory_y'],float)
        dx=x[guard]-x[0];dy=y[guard]-y[0];length=math.hypot(dx,dy)
        angle=math.atan2(dy,dx)-state['theta'];heading=abs(math.atan2(math.sin(angle),math.cos(angle)))
        # Only present obstacle geometry is used; no hidden future plant states.
        positions=np.column_stack([np.r_[state['x'],x[:guard+1]],np.r_[state['y'],y[:guard+1]]])
        clearance=float('inf')
        for j in range(len(ctx['noise'])):
            center=np.array([preview['obj_%d_x'%j][0],preview['obj_%d_y'%j][0]])
            radius=preview['obj_%d_r'%j][0]
            clearance=min(clearance,float(np.linalg.norm(positions-center,axis=1).min()-1.5*radius))
        values=dict(tracking=math.hypot(state['x']-x[0],state['y']-y[0]),heading=heading,
            yaw_rate=abs(float(np.asarray(ctx['previous_input']['u_omega']).ravel()[0])),
            speed=float(np.asarray(ctx['previous_input']['u_s']).ravel()[0]),clearance=clearance,preview_distance=length)
        good=values['tracking']<=p['tracking'] and heading<=p['heading'] and values['yaw_rate']<=p['yaw_rate'] and values['speed']>=p['min_speed'] and clearance>=p['clearance'] and length>=.2
    return (policy['short_h'] if good else base),dict(use_short=bool(good),values=values)
