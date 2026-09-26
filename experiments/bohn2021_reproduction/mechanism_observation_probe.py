"""Check whether persistent forecast uncertainty is visible in the RL input."""
import copy
import json
import numpy as np
from mechanism_probe import OUT, fresh, checked_step
from run import write


def main():
    out=OUT/'observation_alias';out.mkdir(exist_ok=True)
    write(out/'protocol.json',{'post_hoc':True,'cases':[0,3,6],'anchor':20,'horizons':[10,25,40],
        'forecast_seed_scales':[1,0,-1],'prefix':'Nominal zero-terminal H10',
        'intervention':'Only change the persistent obstacle forecast seed after identical prefix; preserve physical state, true obstacles, solver warm start and RL observation.',
        'scope':'Demonstrates hidden controller context, not a proof that all stochastic observations violate the Markov property.'})
    bank=json.loads((OUT/'vehicle_bank.json').read_text())['cases']
    rows=[]
    for case_id in [0,3,6]:
        base=json.loads((OUT/('vehicle_case%02d'%case_id)/'h10.json').read_text())
        for h in [10,25,40]:
            observations=[];states=[];outcomes=[]
            for scale in [1,0,-1]:
                env,obs=fresh('vehicle',bank[case_id])
                for t in range(20):
                    obs,done,r=checked_step(env,'vehicle',10)
                    assert not done and r['state']==base['trace'][t]['state']
                ctrl=env.control_system.controller
                ctrl.object_noise_seed=(np.asarray(ctrl.object_noise_seed)*scale).tolist()
                observations.append(env.get_observation().tolist())
                states.append(copy.deepcopy(env.control_system.current_state))
                obs,done,r=checked_step(env,'vehicle',h)
                outcomes.append({'noise_scale':scale,'next_state':r['state'],'input':r['input'],
                    'one_step_cost':r['cost'],'solver_success':r['solver_success']})
            assert observations[0]==observations[1]==observations[2]
            assert states[0]==states[1]==states[2]
            row={'case':case_id,'horizon':h,'state':states[0],'rl_observation':observations[0],
                'outcomes':outcomes,'same_observation_exact':True,
                'max_input_delta':float(max(np.max(np.abs(np.array(list(a['input'].values()))-np.array(list(b['input'].values())))) for a in outcomes for b in outcomes)),
                'max_position_delta':float(max(np.hypot(a['next_state']['x']-b['next_state']['x'],a['next_state']['y']-b['next_state']['y']) for a in outcomes for b in outcomes))}
            rows.append(row)
            write(out/'progress.json',{'finished':len(rows),'target':9})
    write(out/'completed.json',{'rows':rows,'same_observations_and_states_exact':True,
        'persistent_hidden_variable':'object_noise_seed drawn at reset and reused throughout episode',
        'n_conditions':27,'max_input_delta':max(r['max_input_delta'] for r in rows),
        'max_position_delta':max(r['max_position_delta'] for r in rows)})
    print(json.dumps([{'case':r['case'],'h':r['horizon'],'input_delta':r['max_input_delta'],'position_delta':r['max_position_delta']} for r in rows]),flush=True)


if __name__=='__main__':main()
