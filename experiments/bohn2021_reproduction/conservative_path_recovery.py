"""One-time, pre-validation provenance amendment for path identity checks only."""
import ast
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/conservative_iteration_2026-09-24'
BACKUP = OUT/'recovery_2026-09-25'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    receipt = BACKUP/'amendment.json'
    assert not receipt.exists(), 'One-time amendment; do not repeat'
    assert not (OUT/'evaluations').exists(), 'Must precede any new validation/test outcomes'
    changed = {}
    for name in ('conservative_policy_model.py', 'conservative_iteration_evaluate.py', 'conservative_iteration_timing.py'):
        old = BACKUP/name
        new = ROOT/'experiments/bohn2021_reproduction'/name
        before, after = ast.parse(old.read_text()), ast.parse(new.read_text())
        # All inference, learning, rollout, timing and selection code is identical.
        before.body = [n for n in before.body if not isinstance(n, ast.FunctionDef) or n.name != 'register']
        after.body = [n for n in after.body if not isinstance(n, ast.FunctionDef) or n.name != 'register']
        assert ast.dump(before) == ast.dump(after), name
        changed[str(new)] = dict(before=digest(old), after=digest(new), only_register_changed=True)
    registrations = ['learner_registration.json', 'evaluation_registration.json',
                     'timing_registration.json', 'confirmation_analysis_registration.json']
    revisions = []
    for name in registrations:
        path = OUT/name
        assert path.read_bytes() == (BACKUP/name).read_bytes()
        data = json.loads(path.read_text())
        old_hash = digest(path)
        for key, value in list(data['hashes'].items()):
            full = str(Path(key).resolve())
            if full in changed:
                assert value == changed[full]['before']
                data['hashes'][key] = changed[full]['after']
            else:
                assert digest(Path(key)) == value, key
        path.write_text(json.dumps(data, indent=2)+'\n')
        changed[str(path)] = dict(before=old_hash, after=digest(path))
        revisions.append(name)
    fits = []
    for path in sorted(OUT.glob('*_s*_r*/fit_completed.json')):
        data = json.loads(path.read_text())
        assert digest(path.parent/'policy.json') == data['policy_hash']
        assert digest(path.parent/'collection_completed.json') == data['dataset_hash']
        fits.append(dict(path=str(path), sha256=digest(path), updates=data['training_updates']))
    receipt.write_text(json.dumps(dict(recorded_unix=time.time(), reason='Relative vs absolute __file__ keys falsely rejected unchanged sources.',
        changes=changed, registrations=revisions, preserved_fits=fits,
        numerical_method_unchanged=True, validation_outcomes_exist=False, test_outcomes_exist=False,
        historical_fit_registrations='Original fit receipts retain original learner registration hash; archived original and register-only AST equivalence bridge provenance.',
        failed_attempt='pipeline_fit_vehicle_s0_r0_1790245392966675856.log failed before optimizer creation; zero additional updates.',
        source_hash=digest(Path(__file__))), indent=2)+'\n')
    print(json.dumps(dict(preserved_models=len(fits), formal_updates_preserved=sum(f['updates'] for f in fits), amendment=str(receipt))))


if __name__ == '__main__':
    main()
