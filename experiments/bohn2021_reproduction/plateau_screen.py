"""Registered pilot: long plateaus and previewed cart-position reversals."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from runtime import ART, ROOT
from run import write

OUT = ART / 'results/plateau_screen'
STEPS = 600
OFFSET = .2 * 9.81 * .25
COARSE = [1] + list(range(5, 51, 5))
RULES = ['switch_%d_%d' % (short, long)
         for short in [5, 10] for long in [30, 40]]
SOURCES = [Path(__file__).resolve()] + [ROOT / 'experiments/bohn2021_reproduction' / name
    for name in ['runtime.py', 'run.py', 'optimized_runtime.py',
                 'forecast_runtime.py', 'mechanism_probe.py',
                 'riccati_terminal_probe.py']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(seed, repeats):
    rng = np.random.RandomState(seed)
    scenes = []
    for repeat in range(repeats):
        for amplitude in [.4, .6, .8]:
            direction = int(rng.choice([-1, 1]))
            switches = [int(rng.randint(180, 221)), int(rng.randint(380, 421))]
            values = np.full(STEPS + 55, direction * amplitude)
            values[switches[0]:switches[1]] *= -1
            state = {'pos': float(values[0] + rng.uniform(-.02, .02)),
                     'v': float(rng.uniform(-.04, .04)),
                     'theta': float(rng.uniform(-.02, .02)),
                     'omega': float(rng.uniform(-.04, .04))}
            scenes.append({'id': len(scenes), 'amplitude': amplitude,
                'repeat': repeat, 'switch_clocks': switches,
                'case': {'state': state, 'reference': {},
                    'tvp': {'pos_r': [{'true': [float(v)], 'forecast': []}
                                     for v in values]}}})
    return {'seed': seed, 'scenes': scenes}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'source_hashes.json').exists():
        verify_registration()
        return
    protocol = {
        'purpose': 'Exploratory scenario screening, no RL training or paper reproduction claim.',
        'steps': STEPS, 't_step': .04, 'amplitudes_m': [.4, .6, .8],
        'switch_clocks': [[180, 220], [380, 420]],
        'validation_seed': 26091830, 'validation_scenes': 3,
        'holdout_seed': 26091831, 'holdout_scenes': 6,
        'fixed_coarse': COARSE, 'switch_rules': RULES,
        'switch_logic': 'Long H if reference changes within the next long-H steps, or abs(position error)>0.03, abs(v)>0.08, abs(theta)>0.04, abs(omega)>0.12. Otherwise short H. No oracle access.',
        'fixed_refinement': 'Evaluate every integer within +/-4 of validation-best coarse H, clipped to 1..50; retain all coarse candidates. This is not an exhaustive global integer search.',
        'selection': 'Lowest mean undiscounted raw total cost on all 3 validation scenes, with deterministic arm-name ties. Select one fixed arm and one switch arm globally, not per amplitude.',
        'holdout_arms': 'Selected fixed, selected switch, and fixed H30 reference; deduplicate.',
        'frozen_terminal': 'Identical float32 analytic Riccati prior; zero-terminal H50 reset warmup for every arm; fresh MPC instance per episode.',
        'unchanged': ['plant', 'input and state bounds', 'performance objective',
                      '0.003 H computation proxy', 'full 50-step reference availability'],
        'outcomes': ['raw total cost', 'raw total + 0.4905*600 for readable nonnegative comparison',
                     'performance + 0.4905*actual steps', 'H proxy cost', 'tracking RMSE',
                     'constraint terminations', 'solver failures', 'mean H', 'phase costs'],
        'phase': 'Exogenous transition windows: 50 steps before through 100 steps after each reference switch; remaining steps are plateau. Same labels for all arms.',
        'failure': 'Keep physical-constraint early termination and the existing 10*remaining_steps penalty. Never discard failed scenes.',
        'limits': 'Small pilot; no seed replication of learning, no significance claims, no actual speedup claim. All validation arms and all holdout outcomes reported.',
        'resources': 'At most 2 MPC worker processes. No training. Each job timeout 1800 seconds.'}
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'validation_bank.json', bank(26091830, 1))
    write(OUT / 'holdout_bank.json', bank(26091831, 2))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    hashes[str((ART / 'configs/pendulum.json').relative_to(ROOT))] = digest(ART / 'configs/pendulum.json')
    for name in ['protocol.json', 'validation_bank.json', 'holdout_bank.json']:
        p = OUT / name
        hashes[str(p.relative_to(ROOT))] = digest(p)
    write(OUT / 'source_hashes.json', hashes)


def verify_registration():
    for path, expected in json.loads((OUT / 'source_hashes.json').read_text()).items():
        assert digest(ROOT / path) == expected, path


def action(env, arm):
    if arm.startswith('fixed_'):
        return int(arm.split('_')[1]), 'fixed'
    _, short, long = arm.split('_')
    short, long = int(short), int(long)
    c = env.control_system
    clock = c._step_count
    ref = c.tvps['pos_r'].get_values(clock)
    preview = c.tvps['pos_r'].get_values(clock + 1, clock + long + 1)
    upcoming = any(abs(v - ref) > 1e-9 for v in preview)
    s = c.current_state
    unsettled = (abs(s['pos'] - ref) > .03 or abs(s['v']) > .08 or
                 abs(s['theta']) > .04 or abs(s['omega']) > .12)
    return (long, 'preview' if upcoming else 'recovery') if upcoming or unsettled else (short, 'settled')


def episode(arm, scene):
    from optimized_runtime import install_terminal
    from forecast_runtime import make_env
    from riccati_terminal_probe import prior
    from mechanism_probe import checked_step
    install_terminal('pendulum')
    env = make_env('pendulum', 26091832)
    env.max_steps = STEPS
    env.config['environment']['max_steps'] = STEPS
    w, b, _ = prior()
    w, b = w.astype(np.float32), b.astype(np.float32)
    env.set_value_function_weights_and_biases([w], [b])
    env.reset(**copy.deepcopy(scene['case']))
    initial = copy.deepcopy(env.control_system.current_state)
    trace = []
    start = time.monotonic()
    while True:
        clock = env.control_system._step_count
        h, reason = action(env, arm)
        _, done, row = checked_step(env, 'pendulum', h)
        row.update(env.optimized_solver_info)
        next_clock = env.control_system._step_count
        reference = float(env.control_system.tvps['pos_r'].get_values(next_clock))
        assert reference == scene['case']['tvp']['pos_r'][next_clock]['true'][0]
        phase = 'transition' if any(t - 50 <= clock < t + 100
                                   for t in scene['switch_clocks']) else 'plateau'
        row.update(clock=clock, next_clock=next_clock, reference=reference,
                   reason=reason, phase=phase)
        trace.append(row)
        vf = env.control_system.controller.mpc.vf
        np.testing.assert_array_equal(np.asarray(vf.weights_num).ravel(), w.ravel())
        np.testing.assert_array_equal(np.asarray(vf.biases_num).ravel(), b.ravel())
        if done:
            break
    assert len(trace) == STEPS or trace[-1]['termination'] == 'constraint'
    perf = sum(r['performance'] for r in trace)
    total = sum(r['cost'] for r in trace)
    phases = {}
    for phase in ['plateau', 'transition']:
        part = [r for r in trace if r['phase'] == phase]
        phases[phase] = {'steps': len(part),
            'shifted_performance_cost': sum(r['performance'] + OFFSET for r in part),
            'computation_cost': sum(r['compute'] for r in part),
            'constraint_cost': sum(r['constraint'] for r in part)}
    summary = {'scene': scene['id'], 'amplitude': scene['amplitude'], 'arm': arm,
        'initial_state': initial, 'steps': len(trace), 'termination': trace[-1]['termination'],
        'total_cost': total, 'shifted_total_cost': total + OFFSET * STEPS,
        'shifted_performance_cost': perf + OFFSET * len(trace),
        'performance_cost': perf, 'computation_cost': sum(r['compute'] for r in trace),
        'constraint_cost': sum(r['constraint'] for r in trace),
        'tracking_rmse': float(np.sqrt(np.mean([(r['state']['pos'] - r['reference'])**2 for r in trace]))),
        'max_abs_theta': max(abs(r['state']['theta']) for r in trace),
        'mean_horizon': float(np.mean([r['horizon'] for r in trace])),
        'solver_failure_steps': sum(not r['solver_success'] for r in trace),
        'max_constraint_residual': max(r['max_constraint_residual'] for r in trace),
        'phase_costs': phases, 'elapsed_s': time.monotonic() - start}
    return summary, trace


def worker(split, arm):
    verify_registration()
    dest = OUT / split / arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'summary.json').exists():
        return
    scenes = json.loads((OUT / (split + '_bank.json')).read_text())['scenes']
    summaries = []
    for scene in scenes:
        path = dest / ('episode_%02d.json' % scene['id'])
        if path.exists():
            result = json.loads(path.read_text())
            summary = result['summary']
        else:
            summary, trace = episode(arm, scene)
            write(path, {'summary': summary, 'trace': trace})
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write(dest / 'summary.json', {'arm': arm, 'episodes': summaries,
        'mean_total_cost': float(np.mean([r['total_cost'] for r in summaries])),
        'mean_shifted_total_cost': float(np.mean([r['shifted_total_cost'] for r in summaries])),
        'constraint_episodes': sum(r['termination'] == 'constraint' for r in summaries)})


def launch(split, arms):
    def job(arm):
        log = OUT / ('%s_%s.log' % (split, arm))
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                '--worker', '--split', split, '--arm', arm], stdout=handle,
                stderr=subprocess.STDOUT, timeout=1800)
        result = {'split': split, 'arm': arm, 'exit_code': p.returncode}
        print(json.dumps(result), flush=True)
        if p.returncode:
            raise RuntimeError(result)
        return result
    # Seeded execution order keeps timing drift from following horizon order.
    order = list(arms)
    np.random.RandomState(26091833).shuffle(order)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(job, order))
    return results


def summaries(split):
    return [json.loads(p.read_text()) for p in sorted((OUT / split).glob('*/summary.json'))]


def select(rows, prefix):
    return min((r for r in rows if r['arm'].startswith(prefix)),
               key=lambda r: (r['mean_total_cost'], r['arm']))


def suite():
    prepare()
    launch('validation', ['fixed_%d' % h for h in COARSE] + RULES)
    best = select(summaries('validation'), 'fixed_')
    center = int(best['arm'].split('_')[1])
    refined = [h for h in range(max(1, center - 4), min(50, center + 4) + 1)
               if h not in COARSE]
    write(OUT / 'refinement.json', {'coarse_best': best['arm'], 'additional_fixed_h': refined})
    launch('validation', ['fixed_%d' % h for h in refined])
    rows = summaries('validation')
    selected = {'fixed': select(rows, 'fixed_')['arm'],
                'switch': select(rows, 'switch_')['arm'],
                'validation_summaries_sha256': {r['arm']: digest(OUT / 'validation' / r['arm'] / 'summary.json') for r in rows}}
    write(OUT / 'selection.json', selected)
    launch('holdout', sorted(set([selected['fixed'], selected['switch'], 'fixed_30'])))
    verify_registration()
    write(OUT / 'completed.json', {'selected': selected, 'no_training': True,
        'validation_episodes': sum(len(r['episodes']) for r in rows),
        'holdout_episodes': sum(len(r['episodes']) for r in summaries('holdout'))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--split', choices=['validation', 'holdout'], default='validation')
    ap.add_argument('--arm', default='fixed_30')
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.split, args.arm)
    else:
        suite()


if __name__ == '__main__':
    main()
