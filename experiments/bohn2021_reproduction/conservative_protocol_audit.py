"""Check all pre-outcome registrations without loading any held-out outcomes."""
from pathlib import Path
import json
import time
from conservative_iteration import OUT,verify
from paper_h_soft_probe import read,digest
from run import write


def main():
    verify();checked={}
    for name in ('learner_registration.json','evaluation_registration.json','timing_registration.json'):
        path=OUT/name;data=read(path)
        for p,value in data['hashes'].items():assert digest(Path(p))==value,(name,p)
        checked[str(path)]=digest(path)
    for name,key in [('selection_registration.json','script'),('analysis_registration.json','source')]:
        path=OUT/name;data=read(path)
        assert data['registered_before_validation']
        assert digest(Path(data[key]))==data['source_sha256'],name
        checked[str(path)]=digest(path)
    for name in ('protocol.json','collection_inputs_sha256.json'):
        checked[str(OUT/name)]=digest(OUT/name)
    info=dict(passed=True,checked_unix=time.time(),hashes=checked,source_hash=digest(Path(__file__)),
        scope='Input registrations only. No evaluation outcomes or test bank content read. Does not establish efficacy.')
    write(OUT/'protocol_integrity_audit.json',info)
    print(json.dumps(dict(passed=True,registration_files=len(checked))))


if __name__=='__main__':main()
