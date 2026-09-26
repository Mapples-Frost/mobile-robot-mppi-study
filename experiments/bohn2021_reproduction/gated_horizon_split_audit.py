"""Audit fresh gate-search scenario identities, not any held-out outcomes."""
from pathlib import Path
import json
from runtime import ART
from gated_horizon_search import OUT,freeze
from conservative_split_audit import fingerprint
from paper_h_soft_probe import read,digest
from run import write


def main():
    freeze();history={};historical_banks=[];unparsed=[]
    paths=set()
    for root in (ART/'configs',ART/'results'):
        for pattern in ('*bank*.json','*validation.json','*test.json','*holdout.json'):paths.update(root.rglob(pattern))
    for p in sorted(paths):
        if OUT in p.parents:continue
        d=read(p)
        if not isinstance(d,dict) or not isinstance(d.get('cases'),list):continue
        historical_banks.append(dict(path=str(p),sha256=digest(p)))
        for cid,c in enumerate(d['cases']):
            identity=fingerprint(c)
            if identity is None:unparsed.append(dict(path=str(p),case=cid))
            else:history.setdefault(identity,[]).append(dict(path=str(p),case=cid))
    current=list(sorted((OUT/'banks').glob('*_bank.json')));assert len(current)==12
    seen={};rows=[];overlaps=[]
    for p in current:
        d=read(p);split=d['split'];expected=24 if split.startswith('train') else {'smoke':2,'validation':32,'test':64}[split]
        assert len(d['cases'])==expected
        for cid,c in enumerate(d['cases']):
            identity=fingerprint(c);assert identity
            row=dict(path=str(p),case=cid,split=split,task=identity[0],fingerprint=identity[1])
            if identity in history:overlaps.append(dict(current=row,historical=history[identity]))
            if identity in seen:overlaps.append(dict(current=row,other=seen[identity]))
            seen[identity]=row;rows.append(row)
    result=dict(passed=not overlaps and not unparsed,current_banks=len(current),current_cases=len(rows),
        historical_bank_count=len(historical_banks),historical_unique_cases=len(history),historical_banks=historical_banks,
        current=rows,overlaps=overlaps,unparsed=unparsed,code_hash=digest(Path(__file__)),
        scope='Exact reset-input fingerprints; no test outcomes. Does not prove statistical independence or cover unsaved historical training transitions.')
    write(OUT/'split_audit.json',result);assert result['passed']
    print(json.dumps({k:v for k,v in result.items() if k not in ('historical_banks','current')},indent=2))


if __name__=='__main__':main()
