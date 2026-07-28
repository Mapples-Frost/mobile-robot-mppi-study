# Audit Record — Chapter 1 Is Not Completable Within Its Step Budget

**Created:** 2026-07-28T07:37:26Z
**Status:** read-only analysis of completed noise-basis artifacts. Nothing modified.
**Urgency:** the segment-half-thickness A/B is executing now (8 of 24 episodes). This
finding affects how that experiment must be read.

---

## 1. What reading `evaluation/metrics.py` and the task contract revealed

`metrics.py:2741`:

```python
success = bool(values[-1]["goal_distance"] <= self.goal_tolerance and not collision)
```

Success is the Euclidean distance from the **final** pose to the goal, against
`task.position_tolerance = 0.38 m`.

The chapter-1 task is not a point goal. It is a **24-waypoint polyline spiral**:

| property | value |
|---|---|
| total route length | **27.96 m** |
| start | (−6.19, −3.81) |
| goal | (0.01, −0.01) — the spiral centre |
| `position_tolerance` | 0.38 m |
| `max_steps` | 1200 (120 s at dt = 0.1) |
| **required mean speed** | **0.233 m/s** |

---

## 2. `final_goal_distance` is a poor progress measure on this map

The recorded deadlock at (1.2, −2.05) sits at **arc length 19.71 m — 70% of the
route — and only 0.01 m off the reference path.** It has **8.25 m of route still to
travel**, but its Euclidean distance to the goal is **2.36 m**.

The spiral compresses the final 8.25 m of route into 2.36 m of straight line. The
endpoint therefore understates remaining work by roughly 3.5× near the goal, and the
compression varies with position.

**This is a design error in both A/B protocols I wrote.** I pre-registered median
paired `final_goal_distance` reduction as the continuous co-primary, with a 0.25 m
threshold, on a map where 0.25 m of Euclidean change corresponds to wildly different
amounts of real progress depending on where the robot is. Route arc length is the
correct measure and is computable from the recorded trajectory.

That is my third endpoint-design error in this workstream, after the unstratified
distance confound and the choice of a zero-headroom map.

---

## 3. An error in my first attempt at the fix, caught and corrected

My initial route-progress metric used naive nearest-point projection onto the
polyline. On a spiral that passes close to itself this **aliases**: it reported seed
791101209 at 23.64 m of arc by step 110, which is physically impossible.

The task contract already specifies the correct approach —
`projection_backtrack_distance: 1.35`, `projection_forward_distance: 2.65` — a
*windowed* projection that cannot jump across spiral arms. Recomputing with that
window removed every anomaly and produced monotonic traces. All figures below use
the windowed projection.

---

## 4. Route progress, recomputed

| arm | median route progress | median remaining |
|---|---|---|
| control (i.i.d.) | **10.1%** | 25.14 m |
| treatment (AR(1)) | **43.6%** | 15.77 m |

Paired gain: **median +1.07 m, mean +6.26 m** of route.

The mean/median split reflects a bimodal treatment distribution: six episodes reach
70–71% (the deadlock barrier, tightly clustered) and six collide at 7–17%. Nothing
lands in between.

The Euclidean endpoint reported this same difference as a 0.283 m median gain.

---

## 5. The decisive finding: the budget forbids success

For each non-colliding treatment episode, the steps required to complete the full
27.96 m route **at the rate that episode itself achieved**:

| seed | achieved rate | steps needed | budget |
|---|---|---|---|
| 791101204 | 0.167 m/s | 1678 | 1200 |
| 791101205 | 0.164 m/s | 1703 | 1200 |
| 791101208 | 0.166 m/s | 1680 | 1200 |
| 791101209 | 0.165 m/s | 1693 | 1200 |
| 791101210 | 0.164 m/s | 1701 | 1200 |
| 791101211 | 0.164 m/s | 1703 | 1200 |

**Median 1697 steps needed against a 1200-step budget — short by ~40%.**

Those rates include the 30-second deadlock crawl. Excluding it, the pre-deadlock rate
is ~0.22 m/s, requiring ~1270 steps — **still over budget, by ~6%**.

**Chapter 1 is not completable in 1200 steps at any speed these episodes achieved,
with or without the deadlock.** The required mean speed of 0.233 m/s exceeds every
observed episode mean (0.096–0.344 m/s, most between 0.16 and 0.23).

---

## 6. Consequence for the experiment now running

The segment-half-thickness A/B will very likely return **0/12 success in both arms**,
and that will say nothing about whether the fix works. The primary endpoint is
structurally unable to move, for the same reason it could not move in the noise-basis
screen — but now the reason is quantified rather than suspected.

**Read the running experiment on these instead:**

1. **`known_static_map_candidate_feasible_fraction`** — the mechanism diagnostic
   already in the analyzer. Control is 0.0167 at the deadlock. This is the direct
   test of whether the thickness correction relieves the starvation.
2. **Route progress** (windowed projection). Control median 10.1%, AR(1) median
   43.6%. If the fix helps, the barrier cluster should move past 70%.
3. **Whether the timeouts still cluster at (1.2, −2.05).** If the cluster disperses,
   the deadlock is relieved even if no episode finishes.
4. **Collisions and minimum clearance** — the `FAIL_SAFETY` rule, unchanged and still
   the binding safety check.

Do **not** read `FAIL_NO_EFFECT` on the success endpoint as evidence the fix failed.

---

## 7. Recommended before any confirmatory

Calibrate the scenario so success is attainable. Two options, both cheap:

* **Raise `max_steps`** from 1200 to ≥ 1800 for chapter 1. Justified by measurement,
  not by convenience: the route is 27.96 m and achievable speeds are ~0.17–0.22 m/s.
* **Shorten the route** for a development variant.

Either is the "difficulty ladder" step I recommended in my first assessment and then
failed to apply when choosing this map. Until it is done, no chapter-1 experiment can
produce a non-zero success rate, and every success-gated decision rule will return the
same null.

Also: replace `final_goal_distance` with windowed route arc length as the continuous
endpoint in any future chapter-1 protocol.

---

## 8. Limitations

1. Route progress is computed from recorded `x`/`y` with a windowed projection that
   mirrors the task contract's parameters. It is not the planner's own internal
   progress variable, which was not logged.
2. "Steps needed" extrapolates each episode's own achieved rate linearly. Real
   completion could be faster (the remaining 30% may be less constrained) or slower
   (it re-enters the tight inner spiral). It is an estimate, not a prediction.
3. n = 12 pairs, one map, one seed band.
4. The 0.2% guided-candidate share remains unexplained; `rl_driven_mppi.py`'s actor
   path is still unread.
