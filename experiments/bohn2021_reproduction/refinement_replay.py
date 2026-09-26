"""Replay one fixed case of every refinement seed using its exact adapter."""
import argparse
import json
import tempfile
from pathlib import Path
from runtime import ART,imports
from optimized_runtime import install_terminal,make_env as sparse_env
from forecast_runtime import make_env as forecast_env
from optimized_evaluate import evaluate
from run import write,weights_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['policy_refinement','forecast_refinement'],required=True);args=ap.parse_args()
    root=ART/'results'/args.group;bank=json.loads((root/'holdout_bank.json').read_text())['cases']
    pattern='pendulum_alpha*' if args.group=='policy_refinement' else 'pendulum_forecast*'
    install_terminal('pendulum');_,SAC,_=imports();rows=[]
    for folder in sorted(root.glob(pattern)):
        if not folder.is_dir():continue
        model=SAC.load(str(folder/'model.zip'));before=weights_hash(model)
        make_env=forecast_env if args.group=='forecast_refinement' else sparse_env
        env=make_env('pendulum',662)
        with tempfile.TemporaryDirectory() as tmp:
            evaluate(model,env,bank[:1],Path(tmp),None,True)
            actual=json.loads((Path(tmp)/'trace_00.json').read_text())
        original=json.loads((root/'evaluations'/folder.name/'value/trace_00.json').read_text())
        keys=['state','input','horizon','cost','selected_objective','candidate_objectives']
        assert [{k:t[k] for k in keys} for t in actual]==[{k:t[k] for k in keys} for t in original]
        assert weights_hash(model)==before;model.sess.close()
        rows.append({'model':folder.name,'exact':True,'steps':len(actual),'weights_sha256':before})
    assert len(rows)==(6 if args.group=='policy_refinement' else 3)
    write(root/'replay.json',{'passed':True,'rows':rows});print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
