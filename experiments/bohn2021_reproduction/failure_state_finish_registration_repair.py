"""Archive and amend only finish registration verification after timing exits.

Never changes an original registration or an experiment/measurement function.
Partial applications fail closed for inspection rather than guessing a resume.
"""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/failure_state_probe_2026-09-26'
REPAIR = OUT / 'timing_controller_repair'
TARGET = SCRIPTS / 'failure_state_finish.py'
ORIGINAL = "    if path.exists():assert read(path)==data\n    else:write(path,data)\n"
REPLACEMENT = """    if path.exists():
        original=read(path);expected=dict(original['source_hashes'])
        timing_amendment=OUT/'timing_controller_repair/amendment.json'
        if timing_amendment.exists():
            amended=read(timing_amendment)
            assert amended['original_timing_registration_sha256']==sha(OUT/'timing_registration.json')
            for name,h in amended['archived_files'].items():assert sha(ROOT/name)==h
            allowed={str(SCRIPTS/'failure_state_timing.py'),str(SCRIPTS/'failure_state_timing_audit.py')}
            assert set(amended['changed_sources'])==allowed
            for name,item in amended['changed_sources'].items():
                assert expected[name]==item['old_sha256']
                expected[name]=item['new_sha256']
        finish_amendment=OUT/'timing_controller_repair/finish_amendment.json'
        if finish_amendment.exists():
            amended=read(finish_amendment)
            assert amended['original_finish_registration_sha256']==sha(path)
            assert amended['timing_amendment_sha256']==sha(timing_amendment)
            assert amended['registration_only'] and not amended['execution_logic_changed']
            for name,h in amended['archived_files'].items():assert sha(ROOT/name)==h
            name=str(Path(__file__).resolve());item=amended['changed_source']
            assert item['path']==name and expected[name]==item['old_sha256']
            assert sha(ROOT/item['archive'])==item['old_sha256']
            expected[name]=item['new_sha256']
        assert data['source_hashes']==expected
        assert {k:v for k,v in data.items() if k!='source_hashes'}=={k:v for k,v in original.items() if k!='source_hashes'}
    else:write(path,data)
"""


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    for name in ('status.json', 'finish_status.json', 'timing_status.json'):
        status = read(OUT / name)
        assert status['complete'] and not status['active'], ('Wait for successful completion', name)
        proc = Path('/proc') / str(status['pid']) / 'cmdline'
        assert not proc.exists(), ('Controller still present; no source mutation', name, status['pid'])
        for p, h in status['output_hashes'].items():
            assert sha(Path(p)) == h
    timing = read(OUT / 'timing_delivery/report.json')
    assert timing['passed'] and timing['core_effect_not_evaluated']
    amendment_path = REPAIR / 'finish_amendment.json'
    if amendment_path.exists():
        existing = read(amendment_path)
        assert sha(TARGET) == existing['changed_source']['new_sha256']
        assert existing['original_finish_registration_sha256'] == sha(OUT / 'finish_registration.json')
        assert existing['timing_amendment_sha256'] == sha(REPAIR / 'amendment.json')
        for p, h in existing['archived_files'].items():
            assert sha(ROOT / p) == h
        print('Existing finish registration repair verified; no mutation')
        return
    source_before = TARGET.read_bytes()
    text = source_before.decode('utf-8-sig').replace('\r\n', '\n')
    original_reg = read(OUT / 'finish_registration.json')
    original_hash = original_reg['source_hashes'][str(TARGET)]
    assert sha(TARGET) == original_hash, 'Unexpected finish source; inspect before repair'
    assert text.count(ORIGINAL) == 1
    changed = text.replace(ORIGINAL, REPLACEMENT)
    before_ast = ast.parse(text)
    after_ast = ast.parse(changed)
    before_other = [ast.dump(n) for n in before_ast.body if not isinstance(n, ast.FunctionDef) or n.name != 'register']
    after_other = [ast.dump(n) for n in after_ast.body if not isinstance(n, ast.FunctionDef) or n.name != 'register']
    assert before_other == after_other, 'Execution outside register changed'
    archived = REPAIR / 'failure_state_finish.py'
    assert not archived.exists(), 'Partial repair archive present; inspect before continuing'
    saved_reg = REPAIR / 'finish_registration.json'
    assert saved_reg.exists() and sha(saved_reg) == sha(OUT / 'finish_registration.json')
    with archived.open('xb') as f:
        f.write(source_before)
    new_bytes = changed.encode('utf-8')
    new_hash = hashlib.sha256(new_bytes).hexdigest()
    amendment = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        original_finish_registration_sha256=sha(OUT / 'finish_registration.json'),
        timing_amendment_sha256=sha(REPAIR / 'amendment.json'),
        archived_files={relative(archived): sha(archived), relative(saved_reg): sha(saved_reg)},
        changed_source=dict(path=str(TARGET), old_sha256=original_hash, new_sha256=new_hash, archive=relative(archived)),
        repair_script_sha256=sha(Path(__file__)), registration_only=True, execution_logic_changed=False,
        performed_after_measurements=True, performed_after_finish_exit=True,
        pre_repair_status_hashes={str(OUT / n): sha(OUT / n) for n in ('status.json', 'timing_status.json', 'finish_status.json')},
        reason='Accept only the explicitly mapped prior timing amendment and this self-source registration-only amendment. Preserve original registrations and exact execution AST outside register. Completed timing used the archived original finish execution logic.',
        scientific_rules_changed=False, test_accessed=False)
    temporary = TARGET.with_name(TARGET.name + '.finish_repair.tmp')
    assert not temporary.exists()
    with temporary.open('xb') as f:
        f.write(new_bytes)
    temporary.replace(TARGET)
    with amendment_path.open('x', encoding='utf-8') as f:
        json.dump(amendment, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    assert sha(TARGET) == new_hash and sha(archived) == original_hash
    assert sha(OUT / 'finish_registration.json') == amendment['original_finish_registration_sha256']
    print(json.dumps(amendment, indent=2))


if __name__ == '__main__':
    main()
