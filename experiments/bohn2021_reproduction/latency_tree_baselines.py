"""Strong fixed-grid nomination, then only missing independent-terminal training."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from latency_tree_evaluation_spec import (
    ROOT, OUT, SCRIPTS, TASKS, BASE, HS, SETTINGS, EVAL_REG,
    read, sha, frozen_write, verify_hashes, verify_evaluation, freeze_policies,
    normalize, source_for, check_model, fixed_model, model_key, load, safe,
)
from latency_tree_protocol import LEGACY, MODERN, bank_name


def nominate():
    freeze_policies()
    target = OUT / 'baseline_selection.json'
    if target.exists():
        saved = read(target)
        verify_hashes(saved['hashes'])
        assert saved['source_sha256'] == sha(Path(__file__).resolve())
        return
    audit_path = OUT / 'audit_evaluation_validation_initial.json'
    audit = read(audit_path)
    assert audit['passed']
    verify_hashes(audit['hashes'])
    nominations, pending, supplement, hashes = {}, [], [], {str(audit_path): sha(audit_path)}
    for task in TASKS:
        primary = [load(task, 'primary', s, BASE[task]) for s in range(3)]
        grid = [load(task, 'grid', 0, h) for h in HS]
        chosen = min((a for a in grid if safe(a, primary[0])),
                     key=lambda a: (a['total_cost'], a['h']))
        matched = {h: [load(task, 'matched', s, h) for s in range(3)] for h in HS}
        eligible = [h for h, arms in matched.items()
                    if all(safe(a, b) for a, b in zip(arms, primary))]
        import math
        selected_h = min(eligible, key=lambda h: (
            math.fsum(a['total_cost'] for a in matched[h]) / 3, h))
        light = lambda a: {k: v for k, v in a.items() if k != 'episodes'}
        nominations[task] = dict(
            independent_h=chosen['h'], matched_h=selected_h,
            independent_grid=[light(a) for a in grid],
            matched_grid={str(h): [light(a) for a in arms] for h, arms in matched.items()},
        )
        for a in grid + sum(list(matched.values()), []):
            p = Path(a['path']) / 'completed.json'
            hashes[str(p)] = sha(p)
        if chosen['h'] != BASE[task]:
            for seed in (1, 2):
                job = normalize(task, 'grid', seed, chosen['h'])
                supplement.append(job)
                model = fixed_model(task, chosen['h'], seed)
                inventory = read(EVAL_REG)['model_inventory'][model_key(task, chosen['h'], seed)]
                if inventory is None:
                    assert not model.exists(), 'Inspect any unregistered/partial new baseline'
                    pending.append(dict(task=task, seed=seed, h=chosen['h'], training_steps=15000))
                else:
                    check_model(model, task, chosen['h'], seed)
    value = dict(nominations=nominations, pending_independent_baselines=pending,
                 supplement_jobs=supplement, hashes=hashes,
                 source_sha256=sha(Path(__file__).resolve()), test_accessed=False)
    frozen_write(target, value)
    print(json.dumps(dict(nominations={t: {k: v for k, v in n.items() if k.endswith('_h')}
                                      for t, n in nominations.items()},
                          new_training=pending, supplement_jobs=supplement), indent=2))


def complete():
    verify_evaluation()
    selection = read(OUT / 'baseline_selection.json')
    verify_hashes(selection['hashes'])
    receipt = OUT / 'baseline_completion.json'
    if receipt.exists():
        old = read(receipt)
        assert old['passed']
        verify_hashes(old['hashes'])
        return
    from latency_tree_run import acquire
    from conservative_iteration_timing import idle
    idle()
    lock = acquire(OUT / 'execution_locks/baseline_training')
    pending = selection['pending_independent_baselines']
    records = []
    hashes = {str(OUT / 'baseline_selection.json'): sha(OUT / 'baseline_selection.json')}
    for j in pending:
        task, h, seed = j['task'], j['h'], j['seed']
        assert seed in (1, 2) and h != BASE[task] and j['training_steps'] == 15000
        folder = fixed_model(task, h, seed)
        assert folder == OUT / 'extra_fixed' / model_key(task, h, seed)
        if not (folder / 'completed.json').exists():
            assert not folder.exists(), 'Partial training requires a preserved recovery audit'
            log = OUT / ('fixed_%s_%d.log' % (model_key(task, h, seed), time.time_ns()))
            cmd = [LEGACY, '-u', str(SCRIPTS / 'latency_tree_fixed_train.py'),
                   '--task', task, '--seed', str(seed), '--fixed-horizon', str(h),
                   '--steps', '15000', '--out', str(folder),
                   '--test-bank', str(bank_name(task, 'validation')), '--eval-episodes', '64',
                   '--aligned', '--scaled-obs', '--ent-coef', '1.0',
                   '--batch-size', '256', '--buffer-size', '1000000']
            with log.open('x') as stream:
                result = subprocess.run(cmd, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, (str(log), result.returncode)
        done = check_model(folder, task, h, seed)
        assert done['weights_changed'] and done['evaluation_frozen']
        instrumentation = read(folder / 'instrumentation_completed.json')
        assert instrumentation['passed'] and not instrumentation['smoke']
        verify_hashes(instrumentation['hashes'])
        audit_path = folder / 'independent_log_audit.json'
        if not audit_path.exists():
            log = OUT / ('audit_fixed_%s_%d.log' % (model_key(task, h, seed), time.time_ns()))
            with log.open('x') as stream:
                result = subprocess.run(
                    [MODERN, str(SCRIPTS / 'conservative_fixed_log_audit.py'), '--folder', str(folder)],
                    cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, (str(log), result.returncode)
        audit = read(audit_path)
        assert audit['passed'] and audit['training_steps'] == 15000
        verify_hashes(audit['hashes'])
        hashes[str(audit_path)] = sha(audit_path)
        records.append(dict(job=j, folder=str(folder), new_training_steps=15000,
                            logged_steps=audit['logged_steps'], logged_resets=audit['logged_resets']))
    for task, family, seed, h in selection['supplement_jobs']:
        folder = source_for(task, family, seed, h)
        check_model(folder, task, h, seed)
        for name in ('model.zip', 'manifest.json', 'completed.json'):
            hashes[str(folder / name)] = sha(folder / name)
    frozen_write(receipt, dict(passed=True, training=records,
                              new_training_steps=15000 * len(records),
                              hashes=hashes, test_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('nominate', 'complete'), required=True)
    args = parser.parse_args()
    nominate() if args.mode == 'nominate' else complete()
