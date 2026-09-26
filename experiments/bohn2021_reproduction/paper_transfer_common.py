"""Frozen transfer gate before restoring joint terminal learning (stdlib only)."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17'
OUT = ART / 'results/paper_transfer_2026-09-23'
PRESERVE = ART / 'results/sac_preserve'
TEACHER = ART / 'results/teacher_value'
LEGACY = '/home/mapples/.local/share/bohn2021-python37/bin/python'
DOMAINS = ['plateau_near_600', 'plateau_broad_600', 'redraw_near_600',
           'redraw_broad_600', 'redraw_broad_100']
LEARNED = ['%s_s%d' % (arm, seed) for arm in ['free', 'rule', 'value'] for seed in range(3)]
CONTROLS = ['rule_baseline', 'value_teacher']
FIXED = ['fixed_%02d' % h for h in [1] + list(range(5, 51, 5))]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(seed, count):
    rng = random.Random(seed)
    domains = {name: [] for name in DOMAINS}
    for i in range(count):
        sign = rng.choice([-1, 1])
        levels = [sign*rng.uniform(.35, .85), -sign*rng.uniform(.35, .85), sign*rng.uniform(.35, .85)]
        switches = [rng.randint(165, 235), rng.randint(365, 435)]
        plateau = [levels[0] if t < switches[0] else levels[1] if t < switches[1] else levels[2]
                   for t in range(720)]
        redraw = [rng.uniform(-1, 1)]
        for t in range(1, 720):
            redraw.append(rng.uniform(-1, 1) if rng.random() < .04 else redraw[-1])
        broad = dict(pos=rng.uniform(-.5, .5), v=rng.uniform(-1, 1),
                     theta=rng.uniform(-.78, .78), omega=rng.uniform(-1, 1))
        jitter = dict(pos=rng.uniform(-.08, .08), v=rng.uniform(-.15, .15),
                      theta=rng.uniform(-.07, .07), omega=rng.uniform(-.15, .15))
        for name in DOMAINS:
            reference, initial, duration = name.split('_')
            refs = plateau if reference == 'plateau' else redraw
            state = broad.copy() if initial == 'broad' else dict(jitter, pos=refs[0]+jitter['pos'])
            domains[name].append({'id': i, 'steps': int(duration), 'domain': name,
                'case': {'state': state, 'reference': {},
                         'tvp': {'pos_r': [{'true': [v], 'forecast': []} for v in refs]}}})
    return {'seed': seed, 'domains': domains}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'hashes.json').exists():
        verify()
        return
    protocol = {
        'question': 'Does the successful teacher-preserving horizon policy transfer toward the reconstructed paper task before joint terminal learning is restored?',
        'scope': 'Frozen-policy diagnostic gate, zero training updates; not an optimization success or original-paper reproduction claim.',
        'domains': DOMAINS,
        'design': '2x2 initial-state/reference factorial at 600 steps, plus paired redraw/broad 100-step arm. Scene index pairs the factors. The 100-step arm uses the identical state and reference prefix as redraw_broad_600.',
        'paper_assumptions': '100 steps; pos[-.5,.5],v[-1,1],theta[-.78,.78],omega[-1,1]; reference uniform[-1,1] with independent redraw probability .04. These are existing reconstructed config assumptions, not recovered original experimental files.',
        'splits': {'validation': [260923301, 4], 'holdout': [260923302, 12], 'smoke': [260923399, 1]},
        'conditions': {'learned': LEARNED, 'controls': CONTROLS, 'fixed_validation': FIXED},
        'selection': 'Per domain, fixed H selected on validation by lexicographic (physical failure count, total solver failure fraction, mean raw cost, H). All fixed-grid validation outcomes retained. All 9 learned finals and both frozen controls evaluated; no checkpoint or seed selection.',
        'holdout': '12 conditions per domain: all 9 learned policies, rule, frozen value ensemble, validation-selected fixed H. 60 condition/domain jobs, 720 episodes maximum. Never tune on these scenes.',
        'validation': '22 conditions x 5 domains x 4 scenes = 440 episodes. Total formal evaluation target = 1160 episodes.',
        'primary': 'Value arm vs validation-selected fixed H in redraw_broad_100; cost difference per training seed and paired scene, physical terminations and solver failures separately. Seed is replication unit; no large-sample significance claim.',
        'secondary': 'Value arm vs its frozen value teacher; rule/free contrasts; interactions of reference regime and initial-state distribution; duration contrast. All cells reported, not pooled into a single favorable result.',
        'gate': 'Proceed to jointly learned terminal study only if all three value seeds have lower mean cost than selected fixed H in redraw_broad_100 and neither physical failure count nor solver failure fraction exceeds that comparator. Otherwise first diagnose the failed transfer factor and retrain on a new training distribution with separately frozen new validation/test seeds.',
        'terminal': 'Same frozen float32 Riccati terminal and zero-terminal H50 reset warmup; joint terminal learning deliberately not yet restored.',
        'observation': 'Original56D full causal50-step preview; last feature is actual remaining fraction (T-t)/T for both durations. Frozen19D encoder unchanged. Short duration changes feature semantics in physical steps; report this limitation.',
        'reward': 'Original performance + .003H + 10*(T-t) on physical failure; report raw cost and raw+.4905*T. No reward or penalty tuning. Tail offset .4905*(T-executed) reported separately after early failure.',
        'resources': 'At most two legacy MPC workers, one numerical thread each. Wait for measured-delay drivers, workers AND existing backfill watcher. Smoke/registration checks before formal evaluation. No real-time speedup claims.',
        'recovery': 'Each episode saved atomically. Complete scene files are reused after source checks. Incomplete worker records retained with observed physical-step lower bounds; no exact mid-episode resume claim.',
        'next_priority': 'Continue Bohn reproduction/optimization first. Only then freeze new mobile-robot joint K/H experiments that expose resource-demand heterogeneity fairly. Existing ladder results are retained as historical evidence, not used to cherry-pick a test.'}
    write(OUT/'protocol.json', protocol)
    for split, (seed, n) in protocol['splits'].items():
        write(OUT/(split+'_bank.json'), bank(seed, n))
    folder = Path(__file__).parent
    names = ['paper_transfer_common.py', 'paper_transfer_worker.py', 'paper_transfer_suite.py',
             'paper_transfer_report.py', 'runtime.py', 'run.py', 'optimized_runtime.py',
             'forecast_runtime.py', 'riccati_terminal_probe.py', 'mechanism_probe.py',
             'plateau_screen.py', 'sac_preserve_inference.py', 'teacher_common.py']
    paths = [folder/n for n in names] + [ART/'configs/pendulum.json', OUT/'protocol.json']
    paths += list(OUT.glob('*_bank.json'))
    paths += [PRESERVE/'export'/(n+'.json') for n in LEARNED]
    paths += [TEACHER/'models'/('round1_s%d.json' % s) for s in range(3)]
    write(OUT/'hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})


def verify():
    for name, digest in read(OUT/'hashes.json').items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Frozen source or model changed: '+name)


def metrics(trace, duration):
    raw = sum(r['cost'] for r in trace)
    return {'steps': len(trace), 'raw_cost': raw, 'adjusted_cost': raw+.4905*duration,
            'physical_cost': sum(r['performance']+.4905 for r in trace),
            'H_cost': sum(r['compute'] for r in trace),
            'failure_penalty': sum(r['constraint'] for r in trace),
            'unexecuted_offset': .4905*(duration-len(trace)),
            'physical_failure': trace[-1]['termination']=='constraint',
            'solver_failures': sum(not r['solver_success'] for r in trace),
            'mean_H': sum(r['horizon'] for r in trace)/len(trace)}
