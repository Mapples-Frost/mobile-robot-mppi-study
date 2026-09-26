"""Check improved solver on pre-existing diagnosed scene; never select settings."""
import copy
import json
import numpy as np
from runtime import ART
from optimized_runtime import install_terminal,make_env
from mechanism_probe import checked_step
from run import write

install_terminal('vehicle')
env=make_env('vehicle',99,10)
case=json.loads((ART/'results/mechanism_probe/vehicle_bank.json').read_text())['cases'][6]
obs=env.reset(**copy.deepcopy(case));trace=[]
while True:
    obs,done,row=checked_step(env,'vehicle',10)
    row.update(env.optimized_solver_info);trace.append(row)
    if done:break
out=ART/'results/optimized'
write(out/'solver_regression.json',{'case':'mechanism_probe vehicle6','h':10,'terminal':'zero',
    'cost':sum(r['cost'] for r in trace),'termination':trace[-1]['termination'],
    'solver_failures':sum(not r['solver_success'] for r in trace),'steps':len(trace),'trace':trace,
    'scope':'Known diagnostic scene; not validation/holdout performance evidence; no settings chosen from this result.'})
print('Known scene H10 zero terminal cost',sum(r['cost'] for r in trace),flush=True)
