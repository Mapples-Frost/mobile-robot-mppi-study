"""Reconstruct missing experiment configs; all local assumptions documented in README."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'


def var(name, kind):
    return {'name': name, 'type': kind}


def uniform(low, high):
    return {'type': 'uniform', 'kw': {'low': low, 'high': high}}


def bounds(name, kind, low, high):
    return [dict(var(name, kind), constraint_type=side, value=value, cost=1000)
            for side, value in [('lower', low), ('upper', high)]]


def config(task):
    pend = task == 'pendulum'
    dt = .04 if pend else .1
    if pend:
        model = {'type': 'continuous', 'class': 'nonlinear',
                 'states': {
                     'pos': {'rhs': 'v'},
                     'v': {'rhs': '(m*g*np.sin(theta)*np.cos(theta)-(4/3)*(u1+m*l*omega**2*np.sin(theta)))/(m*np.cos(theta)**2-(4/3)*M)'},
                     'theta': {'rhs': 'omega'},
                     'omega': {'rhs': '(M*g*np.sin(theta)-np.cos(theta)*(u1+m*l*omega**2*np.sin(theta)))/((4/3)*M*l-m*l*np.cos(theta)**2)'}},
                 'inputs': {'u1': {}}, 'parameters': {'m': .2, 'M': .8, 'l': .25, 'g': 9.81}}
        cost = '.5*M*v**2 + m*l*v*omega*np.cos(theta) + (2/3)*m*l*l*omega**2 - m*g*l*np.cos(theta) + 10*(pos-pos_r)**2 + .1*u1**2'
        cvars = [var(n, '_x') for n in model['states']] + [var('u1', '_u'), var('pos_r', '_tvp')] + [var(n, 'parameter') for n in model['parameters']]
        constraints = bounds('pos', '_x', -1.5, 1.5) + bounds('theta', '_x', -1.5707963267948966, 1.5707963267948966) + bounds('u1', '_u', -5, 5)
        initial = {n: uniform(*lims) for n, lims in {'pos': [-.5,.5], 'v': [-1,1], 'theta': [-.78,.78], 'omega': [-1,1]}.items()}
        obs = [var(n, 'state') for n in ['pos','v','theta','omega']] + [var('pos_r','tvp')]
        plant = copy.deepcopy(model)
        model['tvps'] = {'pos_r': {'true': [dict(uniform(-1,1), redraw_probability=.04, forecast_aware=True)]}}
        model['ps'] = {'pos_r': {}}  # Parameter at the terminal point is separate from TVP.
        # The author's model parser gives TVPs priority over same-named parameters;
        # p is passed separately to the terminal estimator by do-mpc.
        reward_vars = [var(n,'state') for n in ['pos','v','theta','omega']] + [var('u1','input'), var('pos_r','tvp')]
        reward_cost = cost
        for k, v in {'m': .2, 'M': .8, 'l': .25, 'g': 9.81}.items():
            import re
            reward_cost = re.sub(r'\b'+k+r'\b', str(v), reward_cost)
    else:
        model = {'type':'continuous','class':'nonlinear',
                 'states': {'x': {'rhs':'u_s*np.cos(theta)'}, 'y': {'rhs':'u_s*np.sin(theta)'}, 'theta': {'rhs':'u_omega'}},
                 'inputs': {'u_s': {}, 'u_omega': {}}, 'parameters': {}, 'ps': {'goal_x': {}, 'goal_y': {}}}
        plant = copy.deepcopy(model)
        plant.pop('ps')
        cost = '(x-trajectory_x)**2+(y-trajectory_y)**2'
        cvars = [var('x','_x'),var('y','_x'),var('trajectory_x','_tvp'),var('trajectory_y','_tvp')]
        constraints = bounds('u_s','_u',0,5) + bounds('u_omega','_u',-4,4)
        initial = {n: uniform(0,0) for n in ['x','y','theta']}
        obs = [var(n,'state') for n in ['x','y','theta']] + [var(n,'tvp') for n in ['trajectory_x','trajectory_y']]
        obs += [var('obj_%d_%s' % (i,n),'tvp') for i in range(3) for n in ['x','y','r']]
        reward_vars = [var('x','state'),var('y','state'),var('trajectory_x','tvp'),var('trajectory_y','tvp')]
        reward_cost = cost
    lam = .003 if pend else .001
    return {
        'environment': {'max_steps':100 if pend else 150,
            'end_on_constraint_violation':['pos','theta'] if pend else [],
            'observation': {'variables': obs},
            'action': {'variables': [{'name':'mpc_horizon'}]},
            'reward': {'variables': reward_vars + [var('mpc_horizon','action')],
                'expression': '-('+reward_cost+')-'+str(lam)+'*mpc_horizon',
                'termination_weight':-10 if pend else -2},
            'randomize': {'state':initial,'reference':{},'model':{}},
            'render': {'plot_action':True,'plot_reward':True},
            'info': {'reward':{'performance':reward_cost, 'computation':str(lam)+'*mpc_horizon'}}},
        'plant': {'render':{},'params': {'t_step':dt},'model':plant},
        'mpc': {'type':'AHMPC' if pend else 'TTAHMPC',
            'model':model,'params':dict({'n_horizon':50,'t_step':dt,'n_robust':0,'store_full_solution':True,'use_nn_vf':True}, **({} if pend else {'n_objects':3})),
            'constraints':constraints, 'reference':{},
            'objective': {'lterm':{'variables':cvars,'expression':cost}, 'mterm':{'expression':'0'},
                'discount_factor':.97, 'vf':{'type':'poly','layers':[]}}},
        'lqr': {'model': 'plant'}}


if __name__ == '__main__':
    out = ART/'configs'
    out.mkdir(parents=True, exist_ok=True)
    for task in ['pendulum','vehicle']:
        (out/(task+'.json')).write_text(json.dumps(config(task),indent=2)+'\n')
    print(out)
