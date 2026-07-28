"""Analyse the chapter-1 budget-extension continuation screen.

Verifies that each extended episode reproduces its 1200-step reference exactly,
then reports whether the extra budget completed the route.

Route progress uses the task contract's own windowed projection
(projection_backtrack_distance / projection_forward_distance). A naive
nearest-point projection aliases across the arms of a spiral and reports
impossible progress; this was observed and corrected during earlier analysis.
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_PROTOCOL = ROOT / "configs/research/budget_extension_screen_v1.yaml"
PREFIX_TOL = 1.0e-9


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(row, key, default=float("nan")):
    v = row.get(key)
    return float(default) if v in (None, "", "nan", "None") else float(v)


class RouteProjector:
    """Windowed polyline projection matching the task contract."""

    def __init__(self, task):
        self.P = np.asarray(task["points"], dtype=np.float64)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.CUM = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(seg.sum())
        self.back = float(task["projection_backtrack_distance"])
        self.fwd = float(task["projection_forward_distance"])

    def trace(self, rows, step=5):
        prev, out = 0.0, []
        for r in rows[::step]:
            pt = np.array([_num(r, "x"), _num(r, "y")])
            lo, hi = prev - self.back, prev + self.fwd
            best = (np.inf, prev)
            for i in range(len(self.P) - 1):
                a, b = self.P[i], self.P[i + 1]
                ab = b - a
                den = float(ab @ ab)
                ln = float(np.linalg.norm(ab))
                if self.CUM[i + 1] < lo or self.CUM[i] > hi:
                    continue
                s = 0.0 if den <= 1e-12 else float(np.clip((pt - a) @ ab / den, 0.0, 1.0))
                arc = self.CUM[i] + s * ln
                if arc < lo or arc > hi:
                    arc = float(np.clip(arc, lo, hi))
                    s = float(np.clip((arc - self.CUM[i]) / max(ln, 1e-9), 0.0, 1.0))
                d = float(np.linalg.norm(pt - (a + s * ab)))
                if d < best[0]:
                    best = (d, arc)
            prev = best[1]
            out.append(prev)
        return np.asarray(out)


def _prefix_matches(ref_rows, ext_rows, n):
    """Confirm the extended run reproduces the reference for its first n steps."""
    if len(ref_rows) < n or len(ext_rows) < n:
        return False, f"too few rows (ref {len(ref_rows)}, ext {len(ext_rows)})"
    for k in ("x", "y", "theta", "applied_v", "applied_omega"):
        a = np.array([_num(r, k) for r in ref_rows[:n]])
        b = np.array([_num(r, k) for r in ext_rows[:n]])
        worst = float(np.nanmax(np.abs(a - b)))
        if worst > PREFIX_TOL:
            return False, f"{k} diverges by {worst:.3e} within the first {n} steps"
    return True, "exact"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args(argv)

    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    seeds = [int(s) for s in protocol["design"]["seeds"]]
    ref_root = ROOT / protocol["design"]["reference_artifacts"]
    ref_steps = int(protocol["intervention"]["reference"])
    ext_steps = int(protocol["intervention"]["treatment"])
    root = args.root.resolve()

    results, missing = [], []
    for seed in seeds:
        ext = root / f"seed{seed}"
        ref = ref_root / f"seed{seed}" / "treatment"
        if not (ext / "metrics.json").exists():
            missing.append(seed)
            continue
        if not (ref / "metrics.json").exists():
            raise FileNotFoundError(f"reference artifact missing for seed {seed}")

        em = json.loads((ext / "metrics.json").read_text(encoding="utf-8"))
        rm = json.loads((ref / "metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((ext / "config_resolved.yaml").read_text(encoding="utf-8"))
        if int(cfg["experiment"]["max_steps"]) != ext_steps:
            raise ValueError(f"seed {seed}: extended max_steps is not {ext_steps}")
        if not bool(cfg["planner"]["known_static_map_segment_half_thickness"]):
            raise ValueError(f"seed {seed}: thickness flag is not enabled")

        er, rr = _rows(ext / "trajectory.csv"), _rows(ref / "trajectory.csv")
        ok, note = _prefix_matches(rr, er, min(ref_steps, len(rr)))

        proj = RouteProjector(cfg["task"])
        arc = proj.trace(er)
        gain_tail = float(arc[-1] - arc[-60]) if len(arc) > 60 else float("nan")
        collided = bool(em["collision"])
        success = bool(em["success"]) and not collided
        outcome = ("success" if success else "collision" if collided
                   else em["termination_reason"])
        tail = er[-300:]
        reasons = {}
        for r in tail:
            if _num(r, "safety_override", 0.0) > 0.0:
                key = str(r.get("safety_reason", ""))
                reasons[key] = reasons.get(key, 0) + 1
        top = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "none"

        results.append({
            "seed": seed,
            "prefix_reproduces_reference": ok,
            "prefix_note": note,
            "reference_outcome": ("collision" if rm["collision"]
                                  else rm["termination_reason"]),
            "reference_steps": int(rm["steps"]),
            "extended_outcome": outcome,
            "extended_steps": int(em["steps"]),
            "route_fraction": float(arc[-1] / proj.total),
            "route_remaining_m": float(proj.total - arc[-1]),
            "route_gain_final_300_steps_m": gain_tail,
            "final_goal_distance": float(em["final_goal_distance"]),
            "terminal_true_clearance_m": _num(er[-1], "clearance"),
            "terminal_safety_reason": top,
        })

    if missing:
        print(f"INCOMPLETE: {len(missing)} episodes missing: {missing}")
        return 1

    print("=" * 104)
    print("CHAPTER-1 BUDGET EXTENSION SCREEN   1200 -> 2050 steps, 3 continuation episodes")
    print("=" * 104)
    bad = [r for r in results if not r["prefix_reproduces_reference"]]
    for r in results:
        flag = "OK" if r["prefix_reproduces_reference"] else "DIVERGED"
        print(f"  seed {r['seed']}  prefix {flag} ({r['prefix_note']})")
    if bad:
        print("\n  PROTOCOL FAILURE: an extended run did not reproduce its reference "
              "prefix.\n  The continuation is not exact and the comparison is void.")
        return 1

    print()
    print(f"{'seed':>10s} {'ref':>10s} {'extended':>10s} {'steps':>6s} {'route%':>7s} "
          f"{'remain m':>9s} {'tail gain':>10s} {'clr':>6s} {'reason':>22s}")
    print("-" * 104)
    for r in results:
        print(f"{r['seed']:>10d} {r['reference_outcome']:>10s} {r['extended_outcome']:>10s} "
              f"{r['extended_steps']:>6d} {100 * r['route_fraction']:>6.1f}% "
              f"{r['route_remaining_m']:>9.2f} {r['route_gain_final_300_steps_m']:>+10.3f} "
              f"{r['terminal_true_clearance_m']:>6.3f} {r['terminal_safety_reason'][:20]:>22s}")
    print("-" * 104)

    n_success = sum(r["extended_outcome"] == "success" for r in results)
    n_coll = sum(r["extended_outcome"] == "collision" for r in results)
    still_moving = [r for r in results
                    if r["extended_outcome"] not in ("success", "collision")
                    and r["route_gain_final_300_steps_m"] > 0.30]
    arrested = [r for r in results
                if r["extended_outcome"] not in ("success", "collision")
                and r["route_gain_final_300_steps_m"] <= 0.30]

    print()
    print("=" * 104)
    print("READING")
    print("=" * 104)
    if n_success:
        print(f"  {n_success}/3 completed the route. Budget was the remaining constraint "
              f"for this mode.")
        print("  -> adopt the calibrated budget and re-run the full chapter-1 matrix.")
    if n_coll:
        print(f"  {n_coll}/3 collided. Extra time converted a timeout into a collision; "
              f"record as a safety finding.")
    if still_moving:
        print(f"  {len(still_moving)}/3 still gaining route at 2050 steps "
              f"(tail gain > 0.30 m). Progress is real but slower than the route "
              f"requires: the constraint is speed, not budget.")
    if arrested:
        print(f"  {len(arrested)}/3 became arrested (tail gain <= 0.30 m). "
              f"These entered a stall given more time; budget is not the constraint "
              f"and the static-stall recovery gap is the blocker.")
    print()
    print("  Three episodes is a diagnostic screen. No success rate and no statistical")
    print("  claim follows, whichever way it goes.")

    out = root / "budget_extension_result.json"
    out.write_text(json.dumps({
        "protocol": protocol["protocol"],
        "reference_max_steps": ref_steps,
        "extended_max_steps": ext_steps,
        "all_prefixes_reproduce": True,
        "n_success": n_success,
        "n_collision": n_coll,
        "n_still_moving": len(still_moving),
        "n_arrested": len(arrested),
        "episodes": results,
    }, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
