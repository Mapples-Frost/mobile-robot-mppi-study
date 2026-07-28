# Chapter 1 Deadlock — Root Cause, and Why the Safety Module Is the Wrong Target

**Created:** 2026-07-28T06:33:04Z
**Status:** read-only analysis. No code, config or result modified.
**Supersedes:** the attribution in `POST_HOC_DIAGNOSIS.md` §3.

---

## 1. Correction to my own previous diagnosis

`POST_HOC_DIAGNOSIS.md` §3 concluded:

> "Chapter 1's zero success rate is gated by a scan-based proximity speed governor,
> not by sampling geometry, candidate coverage, risk thresholds, or the probabilistic
> arbitration path."

**That attribution is wrong.** The safety override is real (74–78% of deadlock steps)
but it is *downstream* of an already-crawling planner, and its magnitude is small.

Evidence: during the deadlock the planner's own proposed velocity is ~0.03 m/s and
the applied velocity is ~0.02 m/s. Safety is removing about 1 cm/s from a command
that was already 3 cm/s. It is not the binding constraint.

This is the second attribution error I have made on this same episode set. The first
(that the deadlock was a geometric chokepoint) was refuted by finding no static
obstacle within 1.2 m. Both errors came from reading one diagnostic channel and
inferring the mechanism rather than checking the parallel channels.

---

## 2. The actual binding constraint

There are **two** candidate filters running in parallel. I had been reading only one
of them for this entire investigation.

Median over the final 300 steps of every AR(1) timeout:

| episode | dynamic feasible | **static-map feasible** | static-map clearance | true clearance |
|---|---|---|---|---|
| 791101204/trt | 1.000 | **0.0167** | 0.0362 | 0.180 |
| 791101205/trt | 1.000 | **0.0167** | 0.0269 | 0.153 |
| 791101208/trt | 1.000 | **0.0167** | 0.0381 | 0.179 |
| 791101209/trt | 1.000 | **0.0167** | 0.0364 | 0.168 |
| 791101210/trt | 1.000 | **0.0167** | 0.0349 | 0.170 |
| 791101211/trt | 1.000 | **0.0200** | 0.0384 | 0.179 |
| 791101206/ctl | 1.000 | **0.0000** | −0.0081 | 0.097 |

`probabilistic_obstacle_candidate_feasible_fraction` — the **dynamic** risk filter I
have been studying since the beginning — passes **100%** of candidates.

`known_static_map_candidate_feasible_fraction` — the **static** map filter, which I
had never examined — passes **1.67%**, i.e. **10 of 600 candidates**.

The value is exactly 0.0167 on five independent seeds, which indicates a saturated
floor rather than a coincidence.

### Why the planner sees a corridor that reality does not

| quantity | value |
|---|---|
| true MuJoCo clearance at deadlock | ~0.17 m |
| `known_static_map_minimum_clearance` | ~0.036 m |
| `known_static_map_influence_m` | 0.12 m |
| `robot_radius` | 0.25 m |

The planner's static map is roughly 0.13 m more conservative than the plant, which
`known_static_map_influence_m = 0.12` accounts for. In an irregular-spiral corridor
that is the difference between a workable margin and 3.6 cm.

### Corrected causal chain

1. AR(1) fixes dynamic avoidance — confirmed: zero-feasible −60%, hard violations
   −58%, collisions 9/12 → 6/12. The robot now *survives* into the spiral corridor.
2. In the corridor, planner-map clearance falls to ~3.6 cm.
3. The static-map candidate filter rejects **98.3%** of candidates.
4. The ~10 survivors are all near-zero-velocity, so the planner proposes ~0.03 m/s.
5. The scan-guard governor scales that to ~0.02 m/s.
6. ~900 steps at 2 cm/s → timeout at 2.32 m from goal.

**AR(1) did not fail. It succeeded at its target and handed the robot to a second,
independent starvation problem in a filter nobody had measured.**

---

## 3. Chapter 1 is traversable — this is a local minimum, not infeasibility

Flood-fill connectivity over a 0.05 m grid spanning the deadlock and the goal region,
using the true static geometry:

| space | connected from deadlock to ≥2.2 m away? |
|---|---|
| physical free space (`clearance > 0`) | **YES** (1564 cells reachable) |
| planner space (`clearance > 0.12` inflation) | **YES** (970 cells reachable) |

A route exists **even in the inflated planner map**. The robot is not trapped by
geometry. It is starved of candidates that traverse a tight passage.

This is the same *class* of failure as RC-1 — candidate starvation — but in the
static filter rather than the sampler, and nothing done so far has addressed it.

---

## 4. Why I am declining the offer to relax or remove the safety module

You authorised substantially relaxing or removing safety. Based on this evidence I
recommend against it, and did not do it.

1. **It is not the binding constraint.** Safety removes ~1 cm/s from a 3 cm/s
   command. Remove it entirely and the planner still has only 10 feasible candidates,
   all near-zero-velocity. The predicted result is a timeout at perhaps 2.0 m instead
   of 2.32 m — not a success.
2. **It would likely convert timeouts into static collisions.** The static-map filter
   at 1.67% and the scan guard are the two things keeping a robot with 3.6 cm of
   planner clearance from driving into a wall. Removing the second while the first is
   starved is the same trade the margin-0.02 round already made and lost.
3. **It would damage the paper.** The nominal shield and independent safety layer are
   part of the claimed architecture and the source of the one statistically confirmed
   result you have (collision reduction). Results produced with safety removed would
   describe a different system.

A **bounded diagnostic ablation** of the safety layer remains legitimate later, as an
attribution measurement rather than a proposed fix — but it should come *after* the
static-filter starvation is addressed, because right now it would be measuring the
wrong variable.

---

## 5. Recommended next step

**Extend the offline coverage benchmark to the static-map filter.** This is the
method that already worked, applied to the filter that turns out to be binding.

Concretely, at the recorded deadlock states:

* what fraction of the maneuver space survives `known_static_map` candidate
  filtering, as a function of `known_static_map_influence_m`?
* does any sampling geometry produce a through-corridor maneuver that survives it?
* is the 0.0167 floor a minimum-candidate guarantee, and if so what are those 10
  candidates?

All of it is offline, runs in seconds against artifacts already in hand, and it
answers whether this is fixable by inflation tuning, by sampling, or neither —
before any further MuJoCo budget is spent.

The two parameters with the most leverage, to be measured rather than assumed:
`known_static_map_influence_m` (0.12) and the filter's minimum-candidate floor.

---

## 6. Status of the confirmatory

Unchanged from `POST_HOC_DIAGNOSIS.md`: **do not run it.** This analysis strengthens
that recommendation. The primary endpoint is capped by static-filter starvation, which
the noise-basis intervention does not touch. A larger sample would re-measure the
same ceiling.

---

## 7. What still stands

| claim | status |
|---|---|
| RC-1 — i.i.d. cannot generate committed maneuvers | stands |
| AR(1) reduces dynamic infeasibility in closed loop | **stands — now confirmed in MuJoCo** |
| P0.2 coverage ranking | stands (dynamic filter only; static filter unmeasured) |
| Chapter-1 deadlock is a safety-governor throttle | **corrected — it is static-filter starvation** |
| Chapter-1 deadlock is a geometric chokepoint | **corrected — route exists, incl. inflated map** |
| Do not run the confirmatory | stands, strengthened |
