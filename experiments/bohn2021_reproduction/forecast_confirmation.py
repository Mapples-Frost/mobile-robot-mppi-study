"""Confirm all full-preview seeds on forty additional predeclared scenes."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import forecast_suite as suite
from runtime import ART,make_env
from run import write,snapshot

def main():
    suite.OUT=ART/'results/forecast_confirmation';suite.OUT.mkdir(exist_ok=True)
    if not (suite.OUT/'protocol.json').exists():write(suite.OUT/'protocol.json',{
        'purpose':'Confirm the initial1.05% mean gain without training, seed selection, or method changes.',
        'seed':26091707,'independent_scenes':40,'models':'All3forecast + all3sparsealpha.01 + all3fixedH30',
        'modes':['value','no_value'],'episodes':720,'selection':'Every model retained; report first20 and confirmation40 separately.'})
    if not (suite.OUT/'holdout_bank.json').exists():
        env=make_env('pendulum',26091707,aligned=True,scaled_obs=True);np.random.seed(26091707);cases=[]
        for _ in range(40):env.reset();cases.append(snapshot(env))
        write(suite.OUT/'holdout_bank.json',{'seed':26091707,'cases':cases})
    folders=[ART/'results/forecast_refinement'/('pendulum_forecast_s%d'%s) for s in range(3)]
    folders += [ART/'results/policy_refinement'/('pendulum_alpha0p01_s%d'%s) for s in range(3)]
    folders += [ART/'results/optimized'/('pendulum_fixed_h30_s%d'%s) for s in range(3)]
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(suite.evaluate,[(f,m) for f in folders for m in ['value','no_value']]))
    write(suite.OUT/'evaluation_completed.json',{'complete':True,'jobs':rows})

if __name__=='__main__':main()
