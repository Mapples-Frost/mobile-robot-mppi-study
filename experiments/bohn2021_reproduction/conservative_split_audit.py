"""Scenario-identity audit across saved history; never reads current outcomes.

Historical bank membership is conservatively treated as previously available,
including old banks whose outcomes may still have been sealed. Fingerprints use
pre-warmup state, reset reference parameters and true/forecast exogenous inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from runtime import ART
from conservative_iteration import OUT, TASKS
from paper_h_soft_probe import read, digest
from run import write


def normalize(value):
    if isinstance(value,bool) or value is None:return value
    if isinstance(value,(int,float)):return float(value)
    if isinstance(value,list):return [normalize(v) for v in value]
    if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
    return value


def fingerprint(case):
    if not isinstance(case,dict) or not all(k in case for k in ('state','reference','tvp')):return None
    state=case['state']
    if not isinstance(state,dict):return None
    if {'pos','v','theta','omega'}<=set(state):task='pendulum';length=152
    elif {'x','y','theta'}<=set(state):task='vehicle';length=202
    else:return None
    tvp={k:v[:length] for k,v in case['tvp'].items() if k!='hend'}
    if not tvp or not all(isinstance(v,list) and len(v)>=length for v in tvp.values()):return None
    payload=normalize(dict(task=task,state=state,reference=case['reference'],tvp=tvp))
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return task,hashlib.sha256(raw).hexdigest()


def main(require_eval):
    paths=set()
    for root in (ART/'configs',ART/'results'):
        for pattern in ('*bank*.json','*validation.json','*test.json','*holdout.json'):
            paths.update(root.rglob(pattern))
    history={};files=[];unparsed=[];non_banks=[]
    for path in sorted(paths):
        if OUT in path.parents:continue
        data=read(path)
        if not isinstance(data,dict) or not isinstance(data.get('cases'),list):
            non_banks.append(str(path));continue
        files.append(dict(path=str(path),sha256=digest(path),cases=len(data['cases'])))
        for cid,case in enumerate(data['cases']):
            identity=fingerprint(case)
            if identity is None:
                unparsed.append(dict(path=str(path),case=cid));continue
            history.setdefault(identity,[]).append(dict(path=str(path),case=cid))
    current_paths=sorted(OUT.glob('*_s*_r*/train_bank.json'))+sorted(OUT.glob('smoke_*/train_bank.json'))
    current_paths+=sorted((OUT/'banks').glob('*_validation.json'))+sorted((OUT/'banks').glob('*_test.json'))
    rows=[];seen={};overlaps=[];bank_rows=[]
    for path in current_paths:
        bank=read(path);split=bank.get('split','smoke' if path.parent.name.startswith('smoke_') else 'train')
        bank_rows.append(dict(path=str(path),split=split,cases=len(bank['cases']),sha256=digest(path)))
        for cid,case in enumerate(bank['cases']):
            identity=fingerprint(case);assert identity is not None,(path,cid)
            row=dict(path=str(path),split=split,case=cid,task=identity[0],fingerprint=identity[1])
            if identity in seen:overlaps.append(dict(current=row,other=seen[identity],kind='current_bank_overlap'))
            if identity in history:overlaps.append(dict(current=row,other=history[identity],kind='historical_bank_overlap'))
            rows.append(row);seen[identity]=row
    expected=[OUT/'banks'/('%s_%s.json'%(t,s)) for t in TASKS for s in ('validation','test')]
    eval_complete=all(p.exists() for p in expected)
    if eval_complete:
        for p in expected:assert len(read(p)['cases'])==(24 if 'validation' in p.name else 48)
    dest=OUT/'split_audit';dest.mkdir(exist_ok=True)
    result=dict(available_banks_disjoint=not overlaps,evaluation_banks_complete=eval_complete,
        final_split_audit_passed=eval_complete and not overlaps and not unparsed,
        historical_banks=files,historical_unique_cases=len(history),historical_unparsed_cases=unparsed,
        excluded_non_bank_files=non_banks,current_banks=bank_rows,current_cases=rows,overlaps=overlaps,
        code_hash=digest(Path(__file__)),
        scope='Full reset-argument fingerprints with task-length forecast prefix; hend excluded as endogenous horizon mask. All historical banks treated as previously available. Current test outcomes never read.',
        limitations='Exact scenario identity only, not proof of stochastic independence. Unrecorded historical training transitions and unsupported legacy bank formats are not covered. Shared scenarios across training seeds are intentional at evaluation and appear once per task bank.')
    write(dest/'audit.json',result)
    print(json.dumps(dict(available_banks_disjoint=not overlaps,evaluation_banks_complete=eval_complete,
        historical_banks=len(files),historical_unique_cases=len(history),unparsed=len(unparsed),
        current_banks=len(bank_rows),current_cases=len(rows),overlaps=len(overlaps)),indent=2))
    assert not overlaps,'Scenario reuse must be resolved before new independent claims'
    if require_eval:assert eval_complete and not unparsed,'Complete coverage is required or unsupported formats need review'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--require-eval',action='store_true');a=ap.parse_args();main(a.require_eval)
