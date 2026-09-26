"""Record a pre-validation deterministic-reset correction with full provenance."""
import json
import time
from pathlib import Path
from runtime import ROOT,ART
from paper_h_soft_probe import read,digest
from run import write

OUT=ART/'results/conservative_iteration_2026-09-24'


def main():
    dest=OUT/'reset_diagnosis';archive=dest/'pre_amendment'
    receipt=dest/'reset_amendment.json'
    assert not receipt.exists() and not (OUT/'evaluations').exists()
    probe_path=dest/'canonical_probe/completed.json';probe=read(probe_path)
    assert probe['complete'] and probe['all_exact']
    preserved={}
    for p in list(OUT.glob('*_s*_r*/collection_completed.json'))+list(OUT.glob('smoke_*/collection_completed.json')):
        d=read(p)
        for f,h in d['hashes'].items():assert digest(Path(f))==h
        preserved[str(p)]=digest(p)
    for r in probe['results']:
        assert digest(OUT/r['folder']/('source_%02d.json'%r['case']))==r['source_hash']
    changed={}
    for name in ('conservative_iteration.py','conservative_iteration_evaluate.py','conservative_iteration_timing.py','conservative_iteration_audit.py'):
        p=ROOT/'experiments/bohn2021_reproduction'/name
        changed[str(p)]=dict(before=digest(archive/name),after=digest(p))
    helper=ROOT/'experiments/bohn2021_reproduction/conservative_canonical_reset.py'
    names=['collection_inputs_sha256.json','learner_registration.json','evaluation_registration.json',
           'timing_registration.json','confirmation_analysis_registration.json','baseline_completion_registration.json']
    for name in names:
        p=OUT/name;assert p.read_bytes()==(archive/name).read_bytes()
        data=read(p);mapping=data if name=='collection_inputs_sha256.json' else data['hashes'];before=digest(p)
        for key,value in list(mapping.items()):
            resolved=str(Path(key).resolve())
            if resolved in changed:
                assert value==changed[resolved]['before'],(name,key)
                mapping[key]=changed[resolved]['after']
            else:assert digest(Path(key))==value,(name,key)
        if name=='collection_inputs_sha256.json':mapping[str(helper)]=digest(helper)
        write(p,data);changed[str(p)]=dict(before=before,after=digest(p))
    write(receipt,dict(recorded_unix=time.time(),reason='Author reset omitted NLP soft-slack initial guess; after-source reset changed initial context and failed exact replay.',
        changes=changed,new_adapter=dict(path=str(helper),sha256=digest(helper)),
        old_collection_inputs_hash=digest(archive/'collection_inputs_sha256.json'),
        preserved_collection_completions=preserved,canonical_source_probe_all_exact=True,
        canonical_source_probe_hash=digest(probe_path),checked_initial_contexts=len(probe['results']),
        method_limits='Only inter-episode eps initial guess set to zero; within-episode warm starts and learned terminal weights unchanged. All new efficacy and timing arms use the same reset adapter. Historical terminal training (including any newly nominated fixed-H training) retains original reset behavior; supervised branch labels use canonical reset after this amendment.',
        reuse_limits='All saved source initial contexts match canonical reset exactly; completed branch data passed source-choice and physical/observation audits. This initial-context probe is not a replay of all stored alternative branches. Historical receipts retain their original input hashes, accepted only by archived exact completion hashes.',
        protocol_and_selection_unchanged=True,validation_outcomes_exist=False,test_outcomes_exist=False,
        interrupted_work='All persisted source/branch files retained. Unfinished calls charged by durable attempt counters; only missing files resumed.',
        source_hash=digest(Path(__file__))))
    print(json.dumps(dict(checked_initial_contexts=len(probe['results']),preserved_datasets=len(preserved),receipt=str(receipt))))


if __name__=='__main__':main()
