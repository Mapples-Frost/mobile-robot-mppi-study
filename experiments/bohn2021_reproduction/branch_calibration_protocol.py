"""Register frozen-policy branch calibration and fresh confirmation banks."""
import fcntl
import json
from pathlib import Path
import numpy as np
from runtime import ART, ROOT, make_env
from run import snapshot, write
from min_q_eval_suite import model_dir
from paper_h_soft_probe import digest, read

OUT = ART / 'results/branch_calibration_2026-09-24'
TASKS = ('vehicle', 'pendulum')
HORIZONS = (1, 10, 20, 25, 30, 40, 50)
ANCHORS = (20, 60)
REPEATS = 2
COUNTS = {'train': 8, 'validation': 10, 'test': 20}
BASE_H = {'vehicle': 25, 'pendulum': 30}
ARMS = ('actor', 'raw_greedy', 'calibrated_greedy', 'fixed')


def bank_path(task, split, seed=None):
    return OUT / ('%s_%s%s_bank.json' % (task, split, '_s%d' % seed if split == 'train' else ''))


def protocol():
    return {
        'label': 'Method extension: finite branch-return calibration of frozen min-Q critics; not strict Bohn reproduction.',
        'hypothesis': 'Supervised paired soft-return advantages improve the learned ranking of H and the resulting greedy controller relative to an identical uncalibrated greedy controller.',
        'models': 'Existing min_q final 15000-step models for both tasks, seeds0/1/2; all retained. Actor, terminal polynomial, V, targetV frozen.',
        'smoke': {'task': 'vehicle', 'model_seed': 0, 'scene_seed': 2609241000,
                  'scenes': 1, 'anchors': [20], 'candidates': [10, 25], 'extra_actor_candidate': True,
                  'repeats': 2, 'fit_steps': 10, 'minimum_anchors': 1,
                  'rule': 'Technical verification only; retain failure, no replacement by outcome; exclude from training and validation.'},
        'execution': {'workers': 2, 'evaluation_order': 'Seeded permutation of four arms per task/seed (2609260000 + task_index*100 + seed).',
                      'replication': 'Three independently trained seeds; paired scenes and common-noise branches are dependent repeated measures, not independent training replicates.',
                      'accounting': 'Persist reset/step call counts per attempt before invocation; includes failed calls, smoke and replay separately. Unfinished calls counted as attempted, not completed. Constructor and bank-generation overhead separately disclosed.'},
        'banks': {'train': {'base_seed': 2609241100, 'seed_rule': 'base + training seed', 'scenes': 8},
                  'validation': {'seed': 2609241200, 'scenes': 10}, 'test': {'seed': 2609241300, 'scenes': 20}},
        'data': {'anchors': list(ANCHORS), 'candidates': list(HORIZONS),
                 'extra_candidate': 'Current original deterministic actor integer H, deduplicated.',
                 'continuation': 'Frozen stochastic original actor, 2 common Gaussian sequences for every candidate at each anchor. Full suffix to termination.',
                 'prefix': 'Generate original deterministic actor episode; replay exactly. If it terminates before an anchor, record skip and failure; no replacement scene.',
                 'reward_scale': {'vehicle': .3, 'pendulum': .6}, 'gamma': .97, 'alpha': 1.,
                 'target': 'Mean sampled finite soft return plus frozen targetV only on time limit; zero forced-first entropy. Center labels within each state, then fit centered Q1 and Q2 separately.',
                 'noise': '2609250000 + task_index*100000 + seed*10000 + case*100 + anchor + repeat; float32 Gaussian reparameterization and author tanh Jacobian.'},
        'fit': {'steps': 1000, 'learning_rate': .0001, 'loss': 'Mean per-state Huber(delta=1) of centered Q minus centered branch target, equal state weight; full batch; no checkpoint selection.',
                'optimizer': 'Fresh Adam for model/values_fn/qf1 and qf2 only; all other parameters byte-identical.',
                'minimum': 'At least four available anchor states per model. Otherwise stop entire study as insufficient data; no outcome-based seed exclusion.'},
        'evaluation': {'arms': list(ARMS), 'terminal': 'Three adaptive arms share original min-Q terminal and reset warmup. Fixed H has its own separately trained terminal for each seed.',
                       'greedy': 'Argmax min(Q1,Q2) over HORIZONS plus current original actor H, smaller-H ties. Same support and Q extraction in raw/calibrated arms; no online rollout lookahead.',
                       'endpoint': 'Mean undiscounted physical + original H-proxy + constraint episode cost; paired scenes and seeds; report goals, constraints, solver failed steps.',
                       'validation_gate': 'On each task, calibrated mean cost below raw_greedy and actor in >=2/3 matched seeds, no more constraint episodes or solver-failed steps than either by seed, and across-seed mean below fixed H. All conditions required.',
                       'test': 'Only open after all six fits and all 24 validation arms audited and gate passes. Once-only 20-scene test; no tuning. Robust claim requires calibrated lower cost than raw/actor/fixed in all matched seeds on both tasks with no extra constraint or solver failures.'},
        'budget': {'inherited_fixed_search_steps': 300000, 'inherited_fixed_extra_seed_steps': 60000,
                   'inherited_author_rl_steps': 90000, 'inherited_min_q_steps': 90000,
                   'new': 'Report every prefix, source episode, branch, reset warmup, critic gradient update, validation/test episode, failure and any interrupted attempt. Supervised calibration has extra simulation; no equal-compute or measured-speed claim.',
                   'fixed_fairness': 'Retain complete historical ten-H search and all independent selected-H terminal seeds. No new fixed H selection using the confirmation test. Fixed methods need no critic calibration. Their full selection/terminal budget and adaptive extra budget are both disclosed.'},
        'limits': 'Small training bank and two MC repeats can overfit/noisily rank actions. Frozen-continuation Q need not predict repeatedly greedy deployment. Mean-centering, support restriction, terminal freezing and greedy extraction are explicit deviations. Fresh banks do not recover missing original paper configs.',
        'excluded': 'No prior validation labels used for fitting, no exposed test used for selection, no mobile-robot joint K/H work.'}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'prepare.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = OUT / 'protocol.json'
    if p.exists(): assert read(p) == protocol()
    else: write(p, protocol())
    for task in TASKS:
        for split in COUNTS:
            for seed in (range(3) if split == 'train' else (None,)):
                path = bank_path(task, split, seed)
                bank_seed = 2609241100 + seed if split == 'train' else {'validation': 2609241200, 'test': 2609241300}[split]
                if path.exists():
                    saved = read(path)
                    assert saved['seed'] == bank_seed and len(saved['cases']) == COUNTS[split]
                    continue
                env = make_env(task, bank_seed, aligned=True, scaled_obs=True)
                np.random.seed(bank_seed)
                cases = []
                for _ in range(COUNTS[split]):
                    env.reset(); cases.append(snapshot(env))
                write(path, {'task': task, 'split': split, 'seed': bank_seed, 'cases': cases})
    paths = [p] + list(OUT.glob('*_bank.json'))
    for task in TASKS:
        paths.append(ART / 'configs' / (task + '.json'))
        for method in ('min_q', 'fixed'):
            for seed in range(3):
                folder = model_dir(task, method, seed)
                paths.extend(folder / name for name in ('model.zip', 'completed.json', 'manifest.json'))
    scripts = ROOT / 'experiments/bohn2021_reproduction'
    paths.extend(scripts / name for name in ('branch_calibration_protocol.py', 'branch_calibration_run.py',
                                             'branch_calibration_audit.py', 'min_q_eval_suite.py', 'min_q_protocol.py',
                                             'runtime.py', 'run.py', 'paper_h_soft_report.py', 'paper_h_soft_probe.py', 'paper_grid_audit_report.py'))
    paths.extend((ART / 'sources').glob('*/**/*.py'))
    hashes = {str(path): digest(path) for path in paths}
    if (OUT / 'inputs_sha256.json').exists(): assert read(OUT / 'inputs_sha256.json') == hashes
    else: write(OUT / 'inputs_sha256.json', hashes)
    print(json.dumps({'prepared': True, 'banks': len(list(OUT.glob('*_bank.json'))), 'inputs': len(hashes)}), flush=True)


def verify():
    assert read(OUT / 'protocol.json') == protocol()
    for path, expected in read(OUT / 'inputs_sha256.json').items():
        assert digest(Path(path)) == expected, path


if __name__ == '__main__': prepare()
