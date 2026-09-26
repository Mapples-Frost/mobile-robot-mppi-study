"""Frozen-policy H-range intervention, explicitly separate from paper reproduction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from runtime import ART, ROOT, imports, make_env
from run import evaluate, snapshot, weights_hash, write


OLD = ART / "results/paper_defaults"
GRID = ART / "results/paper_exact_grid_2026-09-23"
OUT = ART / "results/horizon_range_probe_2026-09-24"
LIMITS = (10, 35)
SEEDS = {"validation": 26092441, "test": 26092442}
COUNTS = {"validation": 8, "test": 20}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_dir(task, method):
    if method.startswith("fixed_h"):
        h = int(method[7:])
        folder = OLD if (task, h) in {("vehicle", 5), ("vehicle", 10),
                                       ("vehicle", 15), ("pendulum", 20),
                                       ("pendulum", 30), ("pendulum", 40)} else GRID
        return folder / ("%s_fixed_h%d" % (task, h))
    return OLD / ("%s_rl_s%d" % (task, int(method[2:])))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    protocol = {
        "label": "Method extension: frozen actor H-range intervention; not original paper SAC training.",
        "question": "Do extreme executed horizons materially explain validation cost excess?",
        "intervention": "Clip the author's deterministic actor output to integer H in [10,35] before env.step; weights and terminal value stay frozen.",
        "limit_choice": "Registered from old validation H distribution and fixed-grid optimum region, before new banks or outcomes; one range for both tasks, no tuning on new test.",
        "splits": {key: {"seed": SEEDS[key], "count": COUNTS[key]} for key in SEEDS},
        "fixed_candidates": list(range(5, 51, 5)),
        "model_source": "All 26 completed paper_exact_grid models. Fixed H has only seed0; this probe cannot prove seed-matched superiority.",
        "selection": "For each task choose one fixed H by minimum mean total cost on new validation, tie toward shorter H. Evaluate all 3 RL seeds, original and clipped, with no RL seed/checkpoint selection.",
        "test_gate": "Clipped RL must improve paired mean cost versus its own unmodified actor in all three seeds, with no extra physical or solver failure, before claiming a robust mechanism improvement. Compare against validation-selected fixed H separately; do not tune the range after test.",
        "evaluation": "Terminal value on; all failures retained; H cost is a proxy, not measured compute time. Reset includes author H50 warmup with each model's terminal value.",
        "source_validation": "results/paper_exact_grid_2026-09-23/validation_diagnosis.json",
    }
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write(protocol_path, protocol)
    for task in ("vehicle", "pendulum"):
        for split in SEEDS:
            path = OUT / ("%s_%s_bank.json" % (task, split))
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
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in [protocol_path] + sorted(OUT.glob("*_bank.json"))}
    path = OUT / "bank_hashes.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write(path, manifest)


def verify_banks():
    hashes = json.loads((OUT / "bank_hashes.json").read_text())
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name


class RestrictedActor:
    def __init__(self, model):
        self.model = model
        self.policy_tf = model.policy_tf

    def predict(self, obs, deterministic=True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        return np.clip(action, *LIMITS), state


def run_one(task, split, method, variant):
    verify_banks()
    if split == "test":
        assert not (OUT / "stopped_after_validation.json").exists(), "Probe stopped after validation"
        selection = json.loads((OUT / "selection.json").read_text())
        assert method in ("fixed_h%d" % selection[task]["fixed_h"], "rl0", "rl1", "rl2")
        if method.startswith("fixed_h"):
            assert variant == "original"
    folder = model_dir(task, method)
    spec = json.loads((folder / "manifest.json").read_text())
    assert spec["task"] == task and (folder / "completed.json").exists()
    bank_path = OUT / ("%s_%s_bank.json" % (task, split))
    bank = json.loads(bank_path.read_text())
    dest = OUT / "evaluations" / split / task / (method + "_" + variant)
    dest.mkdir(parents=True, exist_ok=True)
    completion = dest / "completed.json"
    if completion.exists():
        data = json.loads(completion.read_text())
        assert data["bank_sha256"] == digest(bank_path)
        assert data["model_sha256"] == digest(folder / "model.zip")
        return
    if any(dest.glob("trace_*.json")):
        raise RuntimeError("Partial evaluation requires inspection: " + str(dest))
    env = make_env(task, 26092450, fixed_horizon=spec["fixed_horizon"],
                   aligned=spec["adaptations"]["aligned"],
                   scaled_obs=spec["adaptations"]["scaled_obs"])
    _, SAC, _ = imports()
    model = SAC.load(str(folder / "model.zip"), env=env)
    before = weights_hash(model)
    actor = RestrictedActor(model) if variant == "clipped" else model
    evaluate(actor, env, bank["cases"], dest, spec["fixed_horizon"], True)
    assert weights_hash(model) == before
    write(completion, {"task": task, "split": split, "method": method, "variant": variant,
                       "bank_sha256": digest(bank_path), "model_sha256": digest(folder / "model.zip"),
                       "weights_hash": before, "episodes": len(bank["cases"]), "frozen": True})
    model.sess.close()


def lock_selection():
    verify_banks()
    assert not (OUT / "stopped_after_validation.json").exists(), "Probe stopped after validation"
    selected = {}
    for task in ("vehicle", "pendulum"):
        candidate_scores = []
        for h in range(5, 51, 5):
            dest = OUT / "evaluations/validation" / task / ("fixed_h%d_original" % h)
            assert (dest / "completed.json").exists(), str(dest)
            summary = json.loads((dest / "summary.json").read_text())
            candidate_scores.append({"H": h, "mean_cost": summary["mean_total_cost"],
                                     "constraint_episodes": summary["constraint_episodes"],
                                     "solver_failures": sum(e["solver_failure_steps"] for e in summary["episodes"])})
        for seed in range(3):
            for variant in ("original", "clipped"):
                dest = OUT / "evaluations/validation" / task / ("rl%d_%s" % (seed, variant))
                assert (dest / "completed.json").exists(), str(dest)
        winner = min(candidate_scores, key=lambda row: (row["mean_cost"], row["H"]))
        selected[task] = {"fixed_h": winner["H"], "fixed_grid": candidate_scores,
                          "selection_rule": "min validation mean total cost, tie shorter H"}
    path = OUT / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == selected
    else:
        write(path, selected)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("lock-selection")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--task", choices=["vehicle", "pendulum"], required=True)
    ev.add_argument("--split", choices=list(SEEDS), required=True)
    ev.add_argument("--method", required=True)
    ev.add_argument("--variant", choices=["original", "clipped"], default="original")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "lock-selection":
        lock_selection()
    else:
        if args.method.startswith("fixed_h"):
            assert int(args.method[7:]) in range(5, 51, 5) and args.variant == "original"
        else:
            assert args.method in ("rl0", "rl1", "rl2")
        run_one(args.task, args.split, args.method, args.variant)


if __name__ == "__main__":
    main()
