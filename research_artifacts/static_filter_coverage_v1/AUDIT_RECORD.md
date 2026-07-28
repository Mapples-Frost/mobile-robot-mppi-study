# Audit Record — Static-Map Filter Benchmark, and a Confirmed Geometry Bug

**Created:** 2026-07-28T06:42:18Z
**Status:** read-only analysis. No production code, config, or result modified.

---

## 1. Confirmed: segment-thickness convention bug in the planner

`MppiController._known_static_map_clearance` (`mppi.py:4105`) computes a segment
obstacle's surface distance as:

```python
surface_distance = centerline_distance - float(obstacle.get("thickness", 0.10))
```

It subtracts the **full thickness**. The simulated geometry is built as a MuJoCo box
with half-extents `(0.5 * length, 0.5 * thickness)` (`model_factory.py:47`), and the
plant measures clearance with `0.5 * thickness` (`mujoco_plant.py:743`).

`factories.py:814` injects `known_static_obstacles` as **deepcopies of the raw scene
obstacles**, so the planner receives the same authored `thickness` values the plant
uses. There is no compensating transformation anywhere in the path.

**The planner therefore treats every segment wall's half-width as its full thickness
and is over-conservative by exactly `0.5 * thickness`.**

### Evidence that this is a bug rather than a deliberate margin

* The function's docstring says *"Exact signed footprint clearance to frozen static
  geometry."* The intent is exactness.
* The **box** branch uses `obstacle["size"][:2]` as half-extents — correct.
* The **cylinder** branch uses `radius` — correct.
* Only the **segment** branch uses the full thickness.
* The defaults also disagree: planner `thickness` default `0.10`,
  `model_factory` and plant default `0.20`.

### Quantified

Chapter 1's static geometry is **36 segments**, thickness ∈ {0.20, 0.21, 0.22, 0.28}
→ predicted error 0.100–0.140 m.

| seed | plant clearance | planner clearance | gap |
|---|---|---|---|
| 791101204 | 0.1847 | 0.0847 | **0.1000** |
| 791101208 | 0.1753 | 0.0753 | **0.1000** |
| 791101209 | 0.1746 | 0.0746 | **0.1000** |
| 791101210 | 0.1787 | 0.0787 | **0.1000** |
| 791101211 | 0.1864 | 0.0864 | **0.1000** |

The geometry model used here was validated first: it reproduces the recorded plant
`clearance` to **0.000000 m** at every sampled step on all five seeds. The gap is not
a modelling artifact.

Chapter 1's corridors carry ~0.17–0.19 m of true margin. A systematic 0.10 m error
consumes more than half of it.

---

## 2. NOT established: the bug's quantitative effect on the feasible fraction

The benchmark's third section attempted to re-filter sampled candidates under both
conventions. **It does not reproduce the deployed filter and its output must not be
used.**

| | value |
|---|---|
| reconstructed feasible fraction, planner convention | 0.9012 |
| reconstructed feasible fraction, plant convention | 0.9918 |
| **fraction the system actually recorded** | **0.0160** |

A ~56× discrepancy. The injected geometry is identical, so the error lies in my
reconstruction of the planner's inputs — most likely the proposal prior (I held the
recorded `proposed_v`/`proposed_omega` constant across 36 steps, whereas the deployed
prior is a shifted multi-step solution) and/or the use of nominal rather than residual
rollout dynamics.

An earlier revision of the script printed a verdict from these numbers
("the convention does not by itself explain the starvation"). **That verdict was
unsupported and has been removed.** The script now reports section 3 as invalid.

**Consequence:** the bug is established; its magnitude of effect on candidate
starvation is not. Quantifying it requires the real prior, or an instrumented
in-process run that logs both conventions side by side.

---

## 3. Filter enumeration (answers "are there other parallel filters?")

Empirically, from the diagnostic columns the system itself emits, there are **four**
independent candidate filters, plus a per-source decomposition:

| filter | at the deadlock |
|---|---|
| `probabilistic_obstacle_candidate_feasible_fraction` (dynamic risk) | **1.000** |
| `known_static_map_candidate_feasible_fraction` (static map) | **0.0167** |
| `path_boundary_candidate_feasible_fraction` | **1.000** |
| `paper_same_cycle_guided_cost_filter` (guided cost margin) | n/a, guided ≈ 0.2% |

Per-source: `paper_gaussian_feasible_fraction`,
`paper_guided_{feasible,static,boundary,risk}_feasible_fraction`.

Only the static-map channel degrades: ~0.19–0.21 early in the episode → 0.0167 at the
deadlock. Dynamic and boundary hold at 1.000 throughout.

I had been reading only the dynamic channel for this entire investigation.

---

## 4. Recommended fix path — do NOT apply unilaterally

The one-line correction is `thickness` → `0.5 * thickness` at `mppi.py:4105`, plus
aligning the default (`0.10` → `0.20`).

It must not be applied casually:

1. It **changes every result on every map containing segment obstacles**, including
   the frozen chapter 1/2/3 artifacts and any single-obstacle scene using segments.
2. It **reduces** planner conservatism by 0.10–0.14 m. That is the correct value, but
   it will admit candidates that were previously rejected, and the collision
   consequences must be measured, not assumed.
3. It should ship behind a config flag defaulting to the current behaviour, exactly as
   `noise_basis` did, so existing results stay bit-reproducible.

Suggested sequence: add `known_static_map_segment_half_thickness: false` (default
preserves today's behaviour), add a regression test asserting planner clearance equals
plant clearance for segment geometry under the corrected setting, then run a paired
A/B on chapter 1 with a pre-registered decision rule.

---

## 5. Files

| SHA256 | path |
|---|---|
| see repo | `scripts/run_static_filter_coverage_benchmark.py` |
| see repo | `research_artifacts/static_filter_coverage_v1/static_filter_coverage_result.json` |

Reproduce: `.\.venv\Scripts\python.exe scripts\run_static_filter_coverage_benchmark.py`
