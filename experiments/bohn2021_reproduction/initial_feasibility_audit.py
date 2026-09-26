"""A sufficient angular non-recoverability certificate, not a full viability test."""
import json
import numpy as np
from runtime import ART
from run import write


def main():
    threshold=float(np.arctan(5/(.8*9.81)))
    sources={
        'validation':ART/'results/optimized/pendulum_fixed_h30_s0/eval_value/summary.json',
        'optimized_holdout':ART/'results/optimized/pendulum_fixed_h30_s0/holdout_value/summary.json',
        'entropy_holdout':ART/'results/policy_refinement/evaluations/pendulum_fixed_h30_s0/value/summary.json',
        'forecast_holdout':ART/'results/forecast_refinement/evaluations/pendulum_fixed_h30_s0/value/summary.json'}
    result={'threshold_rad':threshold,'threshold_deg':float(np.degrees(threshold)),
        'certificate':'For theta>atan(umax/(M*g)) and omega>=0, acceleration at omega=0 is strictly positive for every allowed input. Thus omega cannot cross from nonnegative to negative while above the threshold; theta cannot return to upright. Symmetric for negative theta.',
        'limits':'Sufficient condition only. Inward initial velocity may permit recovery; no certificate is asserted for those states. Cart constraints can further reduce recoverability.',
        'banks':{}}
    for name,p in sources.items():
        eps=json.loads(p.read_text())['episodes'];rows=[]
        for ep in eps:
            state=ep['initial_state'];theta,omega=state['theta'],state['omega']
            certified=abs(theta)>threshold and theta*omega>=0
            rows.append({'case':ep['episode'],'theta':theta,'omega':omega,'certificate':bool(certified),
                'observed_constraint_termination':ep['termination']=='constraint','steps':ep['steps']})
        result['banks'][name]={'n':len(rows),'certified_nonrecoverable':sum(r['certificate'] for r in rows),
            'observed_constraint_episodes':sum(r['observed_constraint_termination'] for r in rows),'rows':rows}
    write(ART/'results/optimized/initial_feasibility_audit.json',result)
    print(json.dumps({k:{n:v[n] for n in ['n','certified_nonrecoverable','observed_constraint_episodes']} for k,v in result['banks'].items()},indent=2))


if __name__=='__main__':main()
