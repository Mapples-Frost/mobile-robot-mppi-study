"""Freeze the new min-Q training comparison and independent evaluation banks."""
import hashlib
import json

import numpy as np

from min_q_sac import SOURCE, source_sha256
from runtime import ART, ROOT, make_env
from run import snapshot, write


OUT = ART / "results/min_q_training_2026-09-24"
SEEDS = {"validation": 26092461, "test": 26092462}
COUNTS = {"validation": 10, "test": 20}
BASELINE_H = {"vehicle": 25, "pendulum": 30}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    return {
        "label": "Method extension: actor target min(Q1,Q2); not strict paper reproduction.",
        "single_change": "Replace qf1_pi with existing min_qf_pi in author SAC policy_kl_loss only.",
        "author_source": str(SOURCE.relative_to(ROOT)),
        "author_source_sha256": source_sha256(),
        "training": {"tasks": ["vehicle", "pendulum"], "seeds": [0, 1, 2],
                     "steps": 15000, "batch_size": 256, "buffer_size": 1000000,
                     "ent_coef": 1.0, "aligned": True, "scaled_obs": True,
                     "terminal_value": "learned online, enabled during evaluation",
                     "checkpoints": "Final 15000-step models only; no checkpoint selection."},
        "comparators": {
            "author_rl": "All three existing paper_defaults RL seeds per task, frozen and re-evaluated on the new banks.",
            "fixed_H": BASELINE_H,
            "fixed_selection": "H25 vehicle and H30 pendulum were selected on the old paper-grid validation set before this intervention. The complete 10-H seed0 search and its compute are charged to fixed-H selection; no re-selection on either new bank.",
            "fixed_training": "Reuse existing seed0 at selected H; train seeds1/2 with the same 15000-step author run.py settings and individual terminal values. All seeds retained.",
        },
        "splits": {name: {"seed": SEEDS[name], "episodes_per_task": COUNTS[name]}
                   for name in SEEDS},
        "split_policy": "New validation diagnoses behavior. Independent test is evaluated once after all final models and analysis rules are frozen. No parameter or seed selection on test; report every model and failure.",
        "primary_metric": "Paired episode mean of physical + horizon-proxy + constraint cost, lower is better.",
        "secondary": ["goal/constraint terminations", "solver failure steps", "H distribution",
                      "physical, H and constraint components"],
        "claim_rule": "Report seedwise min-Q minus author RL and min-Q minus matched fixed-H on each task. A robust advantage requires all three min-Q seeds below all three matched fixed-H seeds in test mean total cost, no more constraint terminations, and no more solver failures. Otherwise report mixed/negative evidence, not a reproduced paper claim.",
        "compute_accounting": "Retain training steps, gradient updates, reset/evaluation episodes and the complete fixed-H search budget separately. H cost is a paper proxy; concurrent wall time is not a latency comparison.",
        "stopping": "Train all registered seeds barring reproducible technical failure; disclose any early stop and never treat it as a positive test result.",
        "excluded": "Do not reuse the exposed historical holdout or the unused H-range-probe test bank for selection or new confirmation. No mobile-robot joint K/H experiment.",
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "protocol.json"
    frozen = protocol()
    if p.exists():
        assert json.loads(p.read_text()) == frozen, "Frozen protocol changed"
    else:
        write(p, frozen)
    for task in ("vehicle", "pendulum"):
        for split in SEEDS:
            path = OUT / (task + "_" + split + "_bank.json")
            if path.exists():
                bank = json.loads(path.read_text())
                assert bank["seed"] == SEEDS[split] and len(bank["cases"]) == COUNTS[split]
                continue
            env = make_env(task, SEEDS[split], aligned=True, scaled_obs=True)
            np.random.seed(SEEDS[split])
            cases = []
            for _ in range(COUNTS[split]):
                env.reset()
                cases.append(snapshot(env))
            write(path, {"seed": SEEDS[split], "task": task, "split": split,
                         "cases": cases, "generator": "make_env.reset + run.snapshot"})
    paths = [p] + sorted(OUT.glob("*_bank.json"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    hp = OUT / "bank_hashes.json"
    if hp.exists():
        assert json.loads(hp.read_text()) == hashes, "Frozen bank hash changed"
    else:
        write(hp, hashes)
    print(json.dumps({"prepared": True, "hashes": hashes}, indent=2))


if __name__ == "__main__":
    prepare()
