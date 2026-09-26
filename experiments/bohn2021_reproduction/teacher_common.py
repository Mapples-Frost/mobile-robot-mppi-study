"""Registered teacher-value pilot; no modifications to previous experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import ART, ROOT
from run import write

OUT = ART / 'results/teacher_value'
HS = [5, 10, 20, 25, 30, 40, 50]
STEPS = 600
GAMMA = .97
OFFSET = .4905
SCALE = .6
BRANCH = 60
SEEDS = [0, 1, 2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bank(seed, count):
    rng = np.random.RandomState(seed)
    scenes = []
    for i in range(count):
        magnitude = rng.uniform(.35, .85)
        direction = rng.choice([-1, 1])
        levels = [direction*magnitude, -direction*rng.uniform(.35, .85), direction*rng.uniform(.35, .85)]
        switches = [int(rng.randint(165, 236)), int(rng.randint(365, 436))]
        refs = np.full(STEPS+120, levels[0])
        refs[switches[0]:] = levels[1]
        refs[switches[1]:] = levels[2]
        state = {'pos': float(levels[0]+rng.uniform(-.08, .08)),
                 'v': float(rng.uniform(-.15, .15)),
                 'theta': float(rng.uniform(-.07, .07)),
                 'omega': float(rng.uniform(-.15, .15))}
        anchors = sorted(set([0, 40, switches[0]-46, switches[0]-31, switches[0]-16,
                    switches[0]-1, switches[0]+14, switches[0]+44,
                    switches[1]-31, switches[1]-11, switches[1]+9, switches[1]+39]))
        scenes.append({'id': i, 'switches': switches, 'levels': levels, 'anchors': anchors,
            'case': {'state': state, 'reference': {},
                'tvp': {'pos_r': [{'true': [float(v)], 'forecast': []} for v in refs]}}})
    return {'seed': seed, 'scenes': scenes}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'hashes.json').exists():
        verify()
        return
    protocol = {
        'method': 'Supervised counterfactual relative-cost critic, then one student-state dataset aggregation round. No SAC updates.',
        'scenes': {'train': [26091840, 8], 'validation': [26091841, 4],
                   'aggregation': [26091842, 4], 'holdout': [26091843, 12]},
        'horizons': HS, 'steps': STEPS, 'gamma': GAMMA, 'reward_divisor': SCALE,
        'reward': '-(original physical cost + .4905 + .003*H + original constraint penalty + .4905*remaining_steps on physical failure)/.6',
        'failure': 'Original 10*remaining_steps penalty retained, plus offset padding only. Full finite episode has no continuation at step600. All failures retained.',
        'teacher': 'First choose candidate H once, then frozen causal H5/H30 rule. Forked process preserves exact state, MPC solver state, clocks and RNG.',
        'information': 'In branches, reference beyond initial 50-step preview is held at its last visible value. Labels thus use no privileged future. This is an explicit forecast extrapolation assumption, not exact Q*.',
        'label': '60-step discounted reward plus discounted analytic Riccati tail (physical value plus .4905/(1-.97)) and settled H5 compute tail, clipped to remaining finite steps. Tail is approximate; never bootstrap after termination.',
        'target': 'Discounted cost difference relative to H30, divided by .6; subtract known first-step H penalty difference for regression and add it back at action selection.',
        'critic': 'MLP 64x64 ReLU, 18 causal features, 7 outputs, H30 difference constrained to zero; physical residual prediction plus exact immediate H penalty.',
        'training': 'Seeds0,1,2; Adam3e-4, batch64, 3000 supervised updates per round; SmoothL1 plus .1 weighted pairwise ranking; final checkpoint only. Round1 retrains from identical seed initialization on original and aggregation data.',
        'aggregation': 'Predetermined initial seed0 policy rolls out four new training scenes; 12 scheduled states per surviving episode labelled by same frozen teacher. All three seeds train on common augmented data.',
        'features': 'Position error, velocity, angle, angular velocity, absolute position, current reference, remaining fraction, next reference-change distance/amplitude, mean relative preview in ten 5-step bins (19 inputs). No event schedules.',
        'selection': 'No seed or checkpoint selection. Fixed-H coarse grid 1,5,...50 on validation then integer +/-4 refinement. Validation compares initial/final students and fixed rule. Holdout reports all six students, rule, selected fixed and H30.',
        'metrics': 'Relative-cost MAE, action regret on unseen labelled validation scenes, complete raw/offset cost, shaped reward, discounted reward, tracking RMSE, constraint and solver failures, H cost; all seeds.',
        'budget': '144 initial anchors plus up to48 student anchors, each7x60 continuation steps; prefixes, evaluation and repeated fork smoke counted separately. Two active MPC workers maximum.',
        'limits': ['New scenario distribution and supervised algorithm, not pure data-only causal ablation against old SAC.',
                   'Teacher labels are truncated, model-based, policy-specific and tail-approximated; no global-optimality guarantee.',
                   'H cost remains proxy; shifted percentages depend on specified reward zero.',
                   'No arbitrary reward clipping, no reference-action imitation bonus, no holdout training.']}
    protocol['critic'] = protocol['critic'].replace('18 causal', '19 causal')
    write(OUT/'protocol.json', protocol)
    for split, (seed, count) in protocol['scenes'].items():
        write(OUT/(split+'_bank.json'), make_bank(seed, count))
    source_names = ['teacher_common.py', 'teacher_worker.py', 'teacher_train.py', 'teacher_suite.py',
                    'runtime.py', 'optimized_runtime.py', 'forecast_runtime.py',
                    'riccati_terminal_probe.py', 'mechanism_probe.py', 'plateau_screen.py', 'run.py']
    paths = [ROOT/'experiments/bohn2021_reproduction'/n for n in source_names]
    paths += [ART/'configs/pendulum.json', OUT/'protocol.json']
    paths += [OUT/(s+'_bank.json') for s in protocol['scenes']]
    write(OUT/'hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})


def verify():
    for name, value in read(OUT/'hashes.json').items():
        assert sha(ROOT/name) == value, name


def features(state, refs, remaining):
    refs = np.asarray(refs, dtype=float)
    assert len(refs) == 51
    delta = refs[1:] - refs[0]
    changes = np.flatnonzero(abs(delta) > 1e-8)
    first = int(changes[0]) if len(changes) else None
    features = [(state['pos']-refs[0])/1.5, state['v']/2., state['theta']/.5,
                state['omega']/3., state['pos']/1.5, refs[0]/1.5, remaining/STEPS,
                (first+1)/50. if first is not None else 1.1,
                delta[first]/1.5 if first is not None else 0.]
    features += (delta.reshape(10, 5).mean(axis=1)/1.5).tolist()
    return np.asarray(features, dtype=np.float32)


def infer(model, x):
    value = np.asarray(x, dtype=float)
    for i, layer in enumerate(model['layers']):
        value = value @ np.asarray(layer['weight']).T + np.asarray(layer['bias'])
        if i < len(model['layers'])-1:
            value = np.maximum(value, 0.)
    value = value - value[HS.index(30)]
    return value + .003*(np.asarray(HS)-30)/SCALE


def reward_cost(row, elapsed):
    padding = OFFSET*(STEPS-elapsed) if row['termination'] == 'constraint' else 0.
    return (row['cost']+OFFSET+padding)/SCALE
