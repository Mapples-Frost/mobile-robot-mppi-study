"""Archive mutable training diagnostics and recover the hashed round0 figure input.

Recovery is accepted only if reconstructed bytes match the figure's previously
recorded SHA256; it never changes model weights, raw trajectories or figure data.
"""
import hashlib
import json
from pathlib import Path
from conservative_iteration import OUT
from paper_h_soft_probe import read,digest
from run import write


def main():
    root=OUT/'training_diagnosis';source=root/'diagnosis.json';data=read(source)
    snapshots=root/'snapshots';snapshots.mkdir(exist_ok=True)
    current=snapshots/(digest(source)+'.json')
    if current.exists():assert current.read_bytes()==source.read_bytes()
    else:current.write_bytes(source.read_bytes())
    records=[]
    for path in root.glob('round*_figures/input_audit.json'):
        spec=read(path);expected=spec['source_hash'];target=snapshots/(expected+'.json')
        recovered=False
        if not target.exists():
            assert spec['round']==0,'Only known initial round0 summary can be reconstructed'
            old=dict(data)
            for key in ('conditions','members','extraction'):
                old[key]=[r for r in data[key] if r['round']==0]
            old['hashes']={p:h for p,h in data['hashes'].items() if Path(p).parent.name.endswith('_r0')}
            raw=(json.dumps(old,indent=2,allow_nan=False)+'\n').encode()
            assert hashlib.sha256(raw).hexdigest()==expected,'Historical snapshot bytes differ; never invent a match'
            target.write_bytes(raw);recovered=True
        assert digest(target)==expected
        records.append(dict(figure_input=str(path),snapshot=str(target),sha256=expected,
                            reconstructed_byte_identical=recovered))
    write(root/'snapshot_manifest.json',dict(current_snapshot=str(current),figures=records,
        scope='Immutable training-diagnostic inputs. Recovery requires identity with a previously saved figure input hash; current mutable diagnosis may include newer models.',
        source_hash=digest(Path(__file__))))
    print(json.dumps(dict(current_snapshot=str(current),figures=records),indent=2))


if __name__=='__main__':main()
