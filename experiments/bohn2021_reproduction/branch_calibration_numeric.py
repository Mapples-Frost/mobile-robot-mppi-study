"""Correct the independent float32 Jacobian audit, without changing labels.

The pinned TF1 graph folds 1 + EPS before subtracting the squared action.
The previous auditor rounded after subtraction. This matters near saturation.
All other branch checks and tolerances are retained from paper_h_soft_report.
"""
import math
import numpy as np
from paper_h_soft_probe import GAMMA, SCALES
from paper_h_soft_report import close
from paper_grid_audit_report import physics


def audit_branch(task, case, original, anchor, h, repeat, branch):
    trace = branch['trace']
    limit = 150 if task == 'vehicle' else 100
    assert trace and len(trace) == branch['steps'] <= limit-anchor
    assert branch['start_state'] == original[anchor-1]['state']
    assert trace[0]['horizon'] == branch['first_h'] == h
    assert branch['first_log_probability'] == trace[0]['log_probability'] == 0.
    assert trace[0]['sampling'] is None
    z = np.random.RandomState(branch['noise_seed']).normal(size=200).astype(np.float32)
    costs, entropy, samples = [], [], []
    for k, row in enumerate(trace):
        physical, violated, excess = physics(task, row, case, anchor+k)
        compute = row['horizon'] * (.001 if task == 'vehicle' else .003)
        penalty = (2 if task == 'vehicle' else 10) * (limit-anchor-k-1) if violated else 0.
        close(physical, row['performance']); close(compute, row['compute'])
        close(penalty, row['constraint']); close(physical+compute+penalty, -row['reward'])
        assert excess <= 1e-5 and 1 <= row['horizon'] <= 50
        if k:
            s = row['sampling']
            assert s['z'] == float(z[k])
            close(s['std'], math.exp(s['log_std']))
            close(s['latent'], float(np.float32(s['mu']) + np.float32(s['std']) * z[k]))
            close(s['raw_action'], math.tanh(s['latent']))
            a = np.float32(s['raw_action'])
            mapped = int(np.clip(np.rint(np.float32(1)+(a+np.float32(1))*np.float32(24.5)), 1, 50))
            assert row['horizon'] == mapped
            lp = -.5 * (((s['latent']-s['mu'])/(s['std']+1e-6))**2
                        + 2*s['log_std'] + math.log(2*math.pi))
            # Match TF1 arithmetic folding, not a looser tolerance.
            jac = float(np.float32(np.float32(1)+np.float32(1e-6)) - np.float32(a*a))
            lp -= math.log(jac)
            close(lp, row['log_probability'], atol=4e-5)
            samples.append(s)
        if k < len(trace)-1:
            assert not violated and row['termination'] is None
        costs.append(physical+compute+penalty)
        entropy.append(-row['log_probability'])
    term = branch['termination']
    assert term == trace[-1]['termination']
    if term == 'steps': assert anchor+len(trace) == limit
    elif term == 'constraint': assert violated
    else:
        assert term == 'goal' and task == 'vehicle' and not violated
        end = case['reference']['traj_steps']-1
        s = trace[-1]['state']
        assert math.hypot(s['x']-case['tvp']['trajectory_x'][end]['true'][0],
                          s['y']-case['tvp']['trajectory_y'][end]['true'][0]) <= .5+1e-8
    dc = sum(GAMMA**k*c for k,c in enumerate(costs))
    ent = sum(GAMMA**k*e for k,e in enumerate(entropy))
    soft = -dc/SCALES[task]+ent
    close(sum(costs), branch['total_cost']); close(dc, branch['discounted_cost'])
    close(ent, branch['entropy_contribution']); close(soft, branch['finite_soft_return'])
    close(branch['discounted_target_v_tail'], GAMMA**len(trace)*branch['target_v'])
    close(branch['soft_return_with_target_tail'], soft+branch['discounted_target_v_tail'])
    if term != 'steps': assert branch['target_v'] == 0.
    assert branch['solver_failure_steps'] == sum(not r['solver_success'] for r in trace)
    return samples
