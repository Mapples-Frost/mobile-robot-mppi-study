"""Audit recorded physical states and inputs separately from termination labels."""
import json
import math
from runtime import ART
from run import write

rows=[]
for p in sorted((ART/'results/full').glob('*/completed.json')):
    folder=p.parent;spec=json.loads((folder/'manifest.json').read_text());task=spec['task']
    bank=json.loads((ART/'configs'/(task+'_test_bank.json')).read_text())['cases']
    for mode in ['eval_value','eval_no_value']:
        summary=json.loads((folder/mode/'summary.json').read_text())
        for ep in summary['episodes']:
            trace=json.loads((folder/mode/('trace_%02d.json'%ep['episode'])).read_text())
            state_bad=[];input_bad=[];max_excess=0.
            for step,t in enumerate(trace,1):
                s=t['state'];u=t['input']
                if task=='pendulum':
                    bad=abs(s['pos'])>1.5+1e-6 or abs(s['theta'])>math.pi/2+1e-6
                    excess=max(0.,abs(u['u1'][0])-5.)
                else:
                    tvp=bank[ep['episode']]['tvp']
                    bad=any(math.hypot(s['x']-tvp['obj_%d_x'%j][0]['true'][0],s['y']-tvp['obj_%d_y'%j][0]['true'][0]) < tvp['obj_%d_r'%j][0]['true'][0]-1e-6 for j in range(3))
                    excess=max(0.,-u['u_s'][0],u['u_s'][0]-5.,abs(u['u_omega'][0])-4.)
                if bad:state_bad.append(step)
                if excess>1e-6:input_bad.append(step)
                max_excess=max(max_excess,excess)
            rows.append({'run':folder.name,'mode':mode,'case':ep['episode'],'termination':ep['termination'],
                         'physical_state_violation_steps':state_bad,'input_bound_violation_steps':input_bad,
                         'max_input_bound_excess':max_excess})
result={'tolerance':1e-6,'rows':rows,'episodes_with_state_violation':sum(bool(r['physical_state_violation_steps']) for r in rows),
        'episodes_with_input_violation':sum(bool(r['input_bound_violation_steps']) for r in rows),
        'state_violations_without_constraint_termination':[r for r in rows if r['physical_state_violation_steps'] and r['termination']!='constraint']}
write(ART/'results/physical_constraint_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
