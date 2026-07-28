# Audit Record — Segment Half-Thickness Fix (flagged) and A/B Protocol

**Created:** 2026-07-28T06:53:17Z
**Repo:** `mobile-robot-mppi-study-single-v6`, branch `codex/complex-static-three-dynamic-v1`
**Status:** code change applied behind a flag; protocol frozen; **no episode executed**

---

## 1. The defect

`MppiController._known_static_map_clearance`, segment branch, previously:

```python
surface_distance = centerline_distance - float(obstacle.get("thickness", 0.10))
```

Scene `thickness` is the **full** MuJoCo box width. `model_factory.py:47` builds a
segment as a box with half-extents `(0.5 * length, 0.5 * thickness)`, and
`mujoco_plant.py:743`, `scene_feasibility.py:19` and `static_astar.py:54` all
measure against `0.5 * thickness`.

`static_astar.py:50` carries an explicit comment stating the rule:

> *"Scene segment `thickness` is the full MuJoCo box width. Keep A\* geometry
> identical to model_factory and scene_feasibility, both of which use half of that
> width as the centreline extent."*

The correction was made in A\* and never propagated to the MPPI cost path. The box
and cylinder branches of the same function are correct. Only the segment branch was
wrong.

**Effect:** the planner treated every segment wall's half-width as its full
thickness, over-estimating occupancy by `0.5 * thickness` — measured at exactly
**0.1000 m** at all five recorded chapter-1 deadlock states, against corridors
carrying ~0.17–0.19 m of true margin.

---

## 2. Change applied

| SHA256 (16) | path | change |
|---|---|---|
| `b1c758f3c9d9c09c` | `src/mobile_robot_mppi/planning/mppi.py` | **MODIFIED** — 3 additive edits |
| `18bb5409a385c0fd` | `tests/test_segment_half_thickness_regression.py` | new |
| `55dd0c95285332bb` | `configs/research/segment_half_thickness_ab_development_v1.yaml` | new |
| `41c881c85f07bc65` | `experiments/dynamic_uncertainty/run_segment_half_thickness_ab.py` | new |
| `9b07aa421f192493` | `experiments/dynamic_uncertainty/analyze_segment_half_thickness_ab.py` | new |
| `6ff9ce260a7db968` | `scripts/run_segment_half_thickness_ab.ps1` | new |

Edits to `mppi.py`:

1. `MppiConfig`: new field `known_static_map_segment_half_thickness: bool = False`.
2. `MppiConfig.from_mapping`: reads it, defaulting to `False`.
3. `_known_static_map_clearance` segment branch: extent multiplied by 0.5 **only**
   when the flag is set.

**The default preserves the legacy expression exactly.** No other geometry, filter,
cost term or default was touched.

---

## 3. Regression tests — 6, all passing

| test | result |
|---|---|
| config and `from_mapping` default to `False` | pass |
| `from_mapping` honours an override | pass |
| **default reproduces the legacy full-thickness clearance exactly** (atol 1e-12) | pass |
| **corrected mode equals the plant / MuJoCo convention exactly** (atol 1e-12) | pass |
| correction recovers exactly `0.5 * thickness`, never negative | pass |
| flag does not affect box or cylinder geometry | pass |

Measured clearance recovered per probe point: `0.10, 0.14, 0.14, 0.10, 0.14, 0.10,
0.11, 0.10` m — exactly half the thickness of whichever segment binds.

Run in the project venv with:
`.\.venv\Scripts\python.exe -m pytest tests\test_segment_half_thickness_regression.py -q`

(Executed here via a standalone runner; pytest is unavailable in the analysis
sandbox — no network.)

---

## 4. A/B protocol, frozen before execution

| | |
|---|---|
| map | chapter 1 (all 36 static obstacles are segments; the deadlock is here) |
| control | `known_static_map_segment_half_thickness: false` (legacy) |
| treatment | `true` (corrected) |
| held constant | `noise_basis: "ar1:2.0"` in **both** arms |
| design | paired CRN, 12 seeds, 24 episodes |
| seeds | 791101301–791101312, verified disjoint from all previously opened chapter-1 seeds |

**Why AR(1) in both arms:** the deadlock under test only appears reliably under
AR(1) — 6/12 AR(1) episodes ended in the barrier cluster versus 3/12 for i.i.d., and
only the AR(1) timeouts clustered. Holding it constant isolates the thickness
convention. Consequence, recorded in the protocol: results describe the AR(1)
configuration, not the frozen i.i.d. baseline.

### Two design corrections carried forward

**Confound fixed.** The continuous co-primary is now computed **only over pairs
whose outcome type is identical in both arms**. In the noise-basis screen the
unstratified version (0.283 m, threshold 0.25 m) was produced entirely by
collision→timeout conversion: +2.95 m median in the changed-outcome stratum versus
−0.025 m where the outcome type held. It measured survival time, not navigation.
The changed-outcome stratum is still reported but is not gating. This is
pre-specified.

**Safety rule strengthened.** This is the first change in this workstream that makes
the planner *less* conservative, so `FAIL_SAFETY` triggers on **either** ≥2
treatment-only collisions **or** a median true-clearance regression exceeding
0.05 m. A completion gain bought with clearance is rejected.

### Decision rule (frozen, safety evaluated first)

| rule | condition | action |
|---|---|---|
| `FAIL_SAFETY` | ≥2 treatment-only collisions **or** median clearance regression > 0.05 m | stop, seal, keep default `false`, no confirmatory |
| `PASS_TO_CONFIRMATORY` | no safety failure **and** (net success ≥ +2 pairs **or** same-outcome-stratum median distance reduction ≥ 0.25 m) | pre-register n ≥ 20 confirmatory |
| `FAIL_NO_EFFECT` | neither | correction is geometrically right but does not relieve the deadlock; look upstream |

A mechanism diagnostic — `known_static_map_candidate_feasible_fraction`, currently
0.0167 at the deadlock — is reported but is **not** gating.

**The flag default is not changed by this protocol regardless of outcome.**

---

## 5. Pre-execution validation — 5 checks, all passing

runner and analyzer parse; protocol frozen with correct arm assignment and 12 unique
fresh seeds; seeds disjoint from every previously opened chapter-1 seed; the runner's
own validator accepts the protocol; the analyzer evaluates safety before efficacy and
gates on the stratified endpoint.

MuJoCo-dependent paths are unvalidated here, so **the first episode is also a smoke
test**.

---

## 6. Honest limitations

1. **Effect size is unknown.** The bug is confirmed; its quantitative effect on the
   candidate feasible fraction was **not** established offline — the reconstruction
   in `static_filter_coverage_v1` was ~56× off the recorded value and its section 3
   is void. This experiment is the first real measurement.
2. **The fix may not be sufficient.** The static filter passes 1.67% at the
   deadlock. Recovering 0.10 m of clearance may or may not lift that enough to
   escape. `FAIL_NO_EFFECT` is a plausible outcome and is not a failure of the fix's
   correctness — the geometry is wrong either way and worth correcting.
3. **Other full-thickness sites left alone.** `legacy_pipeline.py:151` (dynamic-track
   matching, where over-inclusion is conservative in the safe direction) and
   `static_astar.py:149` (search-region padding, harmless). Neither changed.
4. **Scope of the flag when enabled.** It changes results on *any* map containing
   segment obstacles. Chapters 1–3 are affected; single-obstacle scenes should be
   checked before the default is ever flipped.
5. **Unread code remains.** Most of `safety/arbiter.py`, `evaluation/metrics.py` and
   `rl_driven_mppi.py`. A further attribution error is possible.

---

## 7. How to run

```powershell
cd D:\Projects\mobile-robot-mppi-study-single-v6

.\.venv\Scripts\python.exe -m pytest tests\test_segment_half_thickness_regression.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_noise_basis_regression.py -q

# smoke one episode first
.\.venv\Scripts\python.exe -m experiments.dynamic_uncertainty.run_segment_half_thickness_ab `
  --map chapter1 --seed 791101301 --arm control `
  --output research_artifacts\segment_half_thickness_ab_v1\seed791101301\control

# full screen
.\scripts\run_segment_half_thickness_ab.ps1
```

Commit the protocol **before** running so its timestamp provably precedes the
evidence.

```powershell
git add -- src/mobile_robot_mppi/planning/mppi.py `
           tests/test_segment_half_thickness_regression.py `
           configs/research/segment_half_thickness_ab_development_v1.yaml `
           experiments/dynamic_uncertainty/run_segment_half_thickness_ab.py `
           experiments/dynamic_uncertainty/analyze_segment_half_thickness_ab.py `
           scripts/run_segment_half_thickness_ab.ps1 `
           research_artifacts/segment_half_thickness_ab_v1/AUDIT_RECORD.md

git commit -m "fix(flagged): segment clearance used full thickness instead of half

Scene segment thickness is the full MuJoCo box width; model_factory builds the
geom with half-extent 0.5*thickness and mujoco_plant, scene_feasibility and
static_astar all measure against that half width. static_astar.py:50 documents
the rule explicitly. Only the MPPI cost path subtracted the full thickness,
over-estimating every segment wall by 0.5*thickness -- exactly 0.1000 m at all
five recorded chapter-1 deadlock states, against ~0.17-0.19 m of true corridor
margin.

Gated behind planner.known_static_map_segment_half_thickness, default false.
The default reproduces the legacy expression to 1e-12, so every frozen result
stays bit-reproducible. Box and cylinder geometry unaffected. 6 regression
tests.

Also freezes a 12-pair chapter-1 A/B, noise_basis held at ar1:2.0 in both arms.
Two design corrections carried forward from the noise-basis screen: the
continuous co-primary is restricted to pairs with identical outcome type (the
unstratified version there measured survival time, not navigation), and
FAIL_SAFETY additionally triggers on a median clearance regression, since this
is the first change that reduces planner conservatism.

Effect size is unmeasured; the offline reconstruction was 56x off and void.
No episode has been run."

git tag -a checkpoint/segment-half-thickness-flagged-fix-20260728 `
  -m "Segment half-thickness correction behind a flag; default legacy; A/B frozen"
```
