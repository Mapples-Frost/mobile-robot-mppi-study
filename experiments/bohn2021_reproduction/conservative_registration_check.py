"""Regression check: path spelling is accepted; content tampering is rejected."""
import json
from pathlib import Path
from unittest.mock import patch
import conservative_policy_model as learner
import conservative_iteration_evaluate as evaluation
import conservative_iteration_timing as timing
from conservative_iteration import ROOT, OUT
from paper_h_soft_probe import digest
from run import write


def main():
    assert Path.cwd() == ROOT
    rows = []
    for module in (learner, evaluation, timing):
        absolute = Path(module.__file__).resolve()
        for spelling in (str(absolute), str(absolute.relative_to(ROOT))):
            with patch.object(module, '__file__', spelling):
                module.register()
                real_digest = module.digest
                def altered(path):
                    return '0'*64 if Path(path).resolve() == absolute else real_digest(path)
                with patch.object(module, 'digest', altered):
                    try:
                        module.register()
                    except AssertionError:
                        rejected = True
                    else:
                        rejected = False
                assert rejected, 'Changed source hash must still be rejected'
            rows.append(dict(module=module.__name__, spelling=spelling,
                             accepted_unchanged=True, rejected_changed_hash=True))
    dest = OUT/'recovery_2026-09-25/registration_regression.json'
    write(dest, dict(passed=True, checks=rows, source_hash=digest(Path(__file__)),
                    no_training_or_evaluation=True))
    print(json.dumps(dict(passed=True, path_cases=len(rows), rejection_cases=len(rows))))


if __name__ == '__main__':
    main()
