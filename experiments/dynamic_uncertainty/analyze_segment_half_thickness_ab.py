"""Apply the frozen segment-half-thickness decision rule to paired episodes.

Thresholds are read from the protocol, which was frozen before execution. This
script applies them; it does not choose them.

Safety is evaluated first and can veto. Because this intervention REDUCES
planner conservatism, the safety rule includes a clearance-regression trigger in
addition to the treatment-only-collision count.

The continuous co-primary is restricted to pairs whose outcome TYPE is identical
in both arms. In the noise-basis screen the unstratified version was produced
entirely by collision-to-timeout conversion and measured survival time rather
than navigation quality. That stratification is pre-specified here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

DEFAULT_PROTOCOL = (
    ROOT / "configs/research/segment_half_thickness_ab_development_v1.yaml"
)


def _num(row, key, default=float("nan")):
    v = row.get(key)
    return float(default) if v in (None, "", "nan", "None") else float(v)


def _episode(path: Path, expected_hash: str):
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((path / "config_resolved.yaml").read_text(encoding="utf-8"))
    recorded = (path / "protocol_sha256.txt").read_text(encoding="ascii").strip()
    if recorded != expected_hash:
        raise ValueError(f"protocol hash mismatch at {path}")
    with (path / "trajectory.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))

    collision = bool(metrics["collision"])
    success = bool(metrics["success"]) and not collision
    reason = str(metrics["termination_reason"])
    timeout = (not success) and (not collision) and (
        int(metrics["steps"]) >= int(config["experiment"]["max_steps"])
        or reason.lower() in {"timeout", "max_steps"}
    )
    static_feas = [
        _num(r, "known_static_map_candidate_feasible_fraction") for r in rows
    ]
    static_feas = [v for v in static_feas if math.isfinite(v)]
    tail = static_feas[-300:] if len(static_feas) >= 300 else static_feas
    override = [_num(r, "safety_override", 0.0) > 0.0 for r in rows]

    return {
        "outcome": "success" if success else "collision" if collision else
                   "timeout" if timeout else reason,
        "safe_success": success,
        "collision": collision,
        "steps": int(metrics["steps"]),
        "final_goal_distance": float(metrics["final_goal_distance"]),
        "minimum_clearance": float(metrics.get("minimum_clearance", float("nan"))),
        "static_feasible_median": float(np.median(static_feas)) if static_feas else float("nan"),
        "static_feasible_tail_median": float(np.median(tail)) if tail else float("nan"),
        "safety_override_fraction": float(np.mean(override)) if override else 0.0,
        "half_thickness": bool(
            config["planner"]["known_static_map_segment_half_thickness"]
        ),
        "noise_basis": str(config["planner"]["noise_basis"]),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args(argv)

    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    expected = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
    th = protocol["decision_rule"]["thresholds"]
    root = args.root.resolve()

    pairs, missing = [], []
    for seed in [int(s) for s in protocol["design"]["seeds"]]:
        c, t = root / f"seed{seed}" / "control", root / f"seed{seed}" / "treatment"
        if not (c / "metrics.json").exists() or not (t / "metrics.json").exists():
            missing.append(seed)
            continue
        ce, te = _episode(c, expected), _episode(t, expected)
        if ce["half_thickness"] is not False or te["half_thickness"] is not True:
            raise ValueError(f"seed {seed}: arm flag assignment is wrong")
        if ce["noise_basis"] != te["noise_basis"]:
            raise ValueError(f"seed {seed}: noise_basis differs between arms")
        pairs.append({"seed": seed, "control": ce, "treatment": te})

    if missing:
        print(f"INCOMPLETE: {len(missing)} pairs missing: {missing}")
        return 1

    print("=" * 104)
    print("SEGMENT HALF-THICKNESS A/B  (chapter1, noise_basis held at ar1:2.0)")
    print("control = legacy full thickness   treatment = corrected half thickness")
    print("=" * 104)
    print(f"{'seed':>10s} {'ctrl':>10s} {'trt':>10s} {'ctrl dist':>10s} {'trt dist':>9s} "
          f"{'d_dist':>8s} {'ctrl statF':>11s} {'trt statF':>10s} {'ctrl clr':>9s} {'trt clr':>8s}")
    print("-" * 104)
    for pr in pairs:
        c, t = pr["control"], pr["treatment"]
        print(f"{pr['seed']:>10d} {c['outcome']:>10s} {t['outcome']:>10s} "
              f"{c['final_goal_distance']:>10.3f} {t['final_goal_distance']:>9.3f} "
              f"{c['final_goal_distance'] - t['final_goal_distance']:>+8.3f} "
              f"{c['static_feasible_tail_median']:>11.4f} {t['static_feasible_tail_median']:>10.4f} "
              f"{c['minimum_clearance']:>9.3f} {t['minimum_clearance']:>8.3f}")
    print("-" * 104)

    # ---- safety, first -----------------------------------------------------
    treat_only_coll = [p["seed"] for p in pairs
                       if p["treatment"]["collision"] and not p["control"]["collision"]]
    ctrl_only_coll = [p["seed"] for p in pairs
                      if p["control"]["collision"] and not p["treatment"]["collision"]]
    clearance_delta = np.array([
        p["control"]["minimum_clearance"] - p["treatment"]["minimum_clearance"]
        for p in pairs
    ])
    clearance_regression = float(np.median(clearance_delta))

    # ---- efficacy ----------------------------------------------------------
    gained = [p["seed"] for p in pairs
              if p["treatment"]["safe_success"] and not p["control"]["safe_success"]]
    lost = [p["seed"] for p in pairs
            if p["control"]["safe_success"] and not p["treatment"]["safe_success"]]
    net = len(gained) - len(lost)

    same = [p for p in pairs if p["control"]["outcome"] == p["treatment"]["outcome"]]
    changed = [p for p in pairs if p["control"]["outcome"] != p["treatment"]["outcome"]]
    d_same = np.array([p["control"]["final_goal_distance"]
                       - p["treatment"]["final_goal_distance"] for p in same])
    d_changed = np.array([p["control"]["final_goal_distance"]
                          - p["treatment"]["final_goal_distance"] for p in changed])
    median_same = float(np.median(d_same)) if len(d_same) else 0.0

    sf_c = float(np.median([p["control"]["static_feasible_tail_median"] for p in pairs]))
    sf_t = float(np.median([p["treatment"]["static_feasible_tail_median"] for p in pairs]))

    print(f"\n  outcomes  control  : success {sum(p['control']['safe_success'] for p in pairs)}"
          f"  collision {sum(p['control']['collision'] for p in pairs)}")
    print(f"            treatment: success {sum(p['treatment']['safe_success'] for p in pairs)}"
          f"  collision {sum(p['treatment']['collision'] for p in pairs)}")
    print(f"  treatment-only collisions : {len(treat_only_coll)} {treat_only_coll}")
    print(f"  control-only collisions   : {len(ctrl_only_coll)} {ctrl_only_coll}")
    print(f"  median clearance change   : {-clearance_regression:+.4f} m "
          f"(negative = treatment has less clearance)")
    print(f"  net safe_success pairs    : {net:+d}  (+{len(gained)} {gained} / -{len(lost)} {lost})")
    print(f"  goal-distance reduction   : same-outcome stratum n={len(same)} "
          f"median {median_same:+.4f} m  [GATING]")
    print(f"                            : changed-outcome stratum n={len(changed)} "
          f"median {float(np.median(d_changed)) if len(d_changed) else 0.0:+.4f} m  [reported only]")
    print(f"  MECHANISM static feasible fraction (tail median): "
          f"control {sf_c:.4f} -> treatment {sf_t:.4f}")

    # ---- frozen decision rule ---------------------------------------------
    if (len(treat_only_coll) >= int(th["treatment_only_collision_pairs"])
            or clearance_regression > float(th["minimum_clearance_regression_m"])):
        verdict = "FAIL_SAFETY"
        action = ("Stop. Seal the evidence, keep the flag default false, do not run "
                  "a confirmatory. Reduced conservatism cost clearance or caused "
                  "collisions.")
    elif (net >= int(th["net_success_pairs"])
          or median_same >= float(th["median_goal_distance_reduction_m"])):
        verdict = "PASS_TO_CONFIRMATORY"
        action = "Pre-register and run a confirmatory at n >= 20 fresh paired seeds."
    else:
        verdict = "FAIL_NO_EFFECT"
        action = ("The correction is geometrically right but does not relieve the "
                  "chapter-1 deadlock. Keep the flag, default false, and look "
                  "upstream.")

    print("\n" + "=" * 104)
    print(f"VERDICT: {verdict}")
    print("=" * 104)
    print(f"  {action}")
    print("\n  12 pairs is a screening sample. No statistical claim follows, either way.")
    print("  The flag default is NOT changed by this protocol regardless of outcome.")

    out = root / "segment_half_thickness_result.json"
    out.write_text(json.dumps({
        "protocol": protocol["protocol"],
        "verdict": verdict,
        "action": action,
        "n_pairs": len(pairs),
        "treatment_only_collisions": treat_only_coll,
        "control_only_collisions": ctrl_only_coll,
        "median_clearance_regression_m": clearance_regression,
        "net_success_pairs": net,
        "success_gained": gained,
        "success_lost": lost,
        "median_goal_distance_reduction_same_outcome_m": median_same,
        "n_same_outcome": len(same),
        "n_changed_outcome": len(changed),
        "static_feasible_control": sf_c,
        "static_feasible_treatment": sf_t,
        "thresholds": dict(th),
        "pairs": pairs,
    }, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
