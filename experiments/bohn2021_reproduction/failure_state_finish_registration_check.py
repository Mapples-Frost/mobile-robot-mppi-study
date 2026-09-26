"""Exercise the actual repaired register AST with in-memory corruptions."""
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
REG = OUT / 'registration.json'
TARGET = SCRIPTS / 'failure_state_finish.py'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    finish = read(OUT / 'finish_status.json')
    assert finish['complete'] and not finish['active']
    assert not (Path('/proc') / str(finish['pid']) / 'cmdline').exists()
    amendment_path = OUT / 'timing_controller_repair/finish_amendment.json'
    amendment = read(amendment_path)
    assert sha(TARGET) == amendment['changed_source']['new_sha256']
    original = ast.parse((ROOT / amendment['changed_source']['archive']).read_text(encoding='utf-8-sig'))
    tree = ast.parse(TARGET.read_text(encoding='utf-8-sig'))
    rest = lambda t: [ast.dump(n) for n in t.body if not isinstance(n, ast.FunctionDef) or n.name != 'register']
    assert rest(original) == rest(tree)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'register')
    module = ast.Module(body=[node], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(TARGET), 'exec')
    timing_path = OUT / 'timing_controller_repair/amendment.json'
    finish_reg = OUT / 'finish_registration.json'
    checks = []

    def evaluate(name, change=None, bad_sha=None):
        def reading(path):
            data = read(path)
            if change is not None:
                change(path, data)
            return data

        def hashing(path):
            return '0' * 64 if bad_sha is not None and path == bad_sha else sha(path)

        def no_write(*args):
            raise RuntimeError('Registration test must never write existing research records')

        namespace = dict(Path=Path, __file__=str(TARGET), ROOT=ROOT, SCRIPTS=SCRIPTS, OUT=OUT,
                         REG=REG, read=reading, sha=hashing, write=no_write, verify=lambda: None)
        exec(code, namespace)
        rejected = False
        try:
            namespace['register']()
        except (AssertionError, KeyError):
            rejected = True
        expected_rejection = change is not None or bad_sha is not None
        assert rejected == expected_rejection, (name, rejected, expected_rejection)
        checks.append(dict(name=name, rejected=rejected, expected_rejection=expected_rejection))

    def change_at(target, mutator):
        def change(path, data):
            if path == target:
                mutator(data)
        return change

    evaluate('valid_registered_amendments')
    evaluate('unchanged_protocol_text_enforced', change_at(finish_reg, lambda d: d.update(conditional='changed')))
    evaluate('timing_original_hash_enforced', change_at(timing_path, lambda d: d.update(original_timing_registration_sha256='0'*64)))
    evaluate('timing_source_map_rejects_unregistered_source', change_at(timing_path, lambda d: d['changed_sources'].update({'unexpected.py': {'old_sha256':'0'*64,'new_sha256':'1'*64}})))
    evaluate('finish_original_registration_hash_enforced', change_at(amendment_path, lambda d: d.update(original_finish_registration_sha256='0'*64)))
    evaluate('finish_link_to_timing_amendment_enforced', change_at(amendment_path, lambda d: d.update(timing_amendment_sha256='0'*64)))
    evaluate('finish_source_identity_enforced', change_at(amendment_path, lambda d: d['changed_source'].update(path='unexpected.py')))
    evaluate('execution_change_disallowed', change_at(amendment_path, lambda d: d.update(execution_logic_changed=True)))
    evaluate('finish_archive_bytes_enforced', bad_sha=ROOT / amendment['changed_source']['archive'])
    evaluate('live_finish_source_bytes_enforced', bad_sha=TARGET)
    result = dict(passed=True, checks=checks, unchanged_execution_ast=True,
                  actual_register_function_executed=True, test_accessed=False, simulations=0,
                  checker_sha256=sha(Path(__file__)), source_sha256=sha(TARGET),
                  amendment_sha256=sha(amendment_path), original_registration_sha256=sha(finish_reg))
    output = OUT / 'timing_controller_repair/finish_registration_checks.json'
    if output.exists():
        assert read(output) == result
    else:
        with output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
