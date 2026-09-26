"""Deterministic inter-episode slack initialization for this extension only.

Preserves all within-episode warm starts, dynamics, costs, learned terminals and
solver settings. The author reset already resets x/u/z; only omitted eps is added.
Historical terminal training is unchanged. All evaluation arms use this adapter.
"""
from pathlib import Path
from runtime import make_env as original_make_env
from paper_h_soft_probe import read,digest


def make_env(*args,**kwargs):
    env=original_make_env(*args,**kwargs)
    original_reset=env.reset
    def reset(**kw):
        env.control_system.controller.mpc.opt_x_num['_eps']=0
        return original_reset(**kw)
    env.reset=reset
    return env


def verify_collection_provenance(dest,done,current_hash):
    if done['inputs_hash']==current_hash:return
    receipt=read(dest.parent/'reset_diagnosis/reset_amendment.json')
    key=str((dest/'collection_completed.json').resolve())
    assert receipt['canonical_source_probe_all_exact']
    assert done['inputs_hash']==receipt['old_collection_inputs_hash']
    assert receipt['preserved_collection_completions'][key]==digest(dest/'collection_completed.json')
