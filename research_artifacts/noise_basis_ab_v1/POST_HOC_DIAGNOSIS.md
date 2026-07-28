# Post-Hoc Diagnosis — Noise-Basis A/B Screen (chapter 1, 12 pairs)

**Created:** 2026-07-28T05:55:19Z
**Status:** read-only analysis of completed artifacts. No code, config or result modified.
**Relationship to the frozen gate:** the gate verdict `PASS_TO_CONFIRMATORY` stands
as recorded. This document does not reinterpret it. It reports two things the gate
was not designed to detect.

---

## 1. The gate passed on a confounded endpoint

The pass came from a median paired final-goal-distance reduction of 0.2830 m against
a frozen 0.25 m threshold — a margin of 0.033 m.

Stratifying by whether the paired outcome *type* changed:

| stratum | n | median distance gain | mean |
|---|---|---|---|
| outcome type unchanged (both collide, or both time out) | 7 | **−0.0249 m** | +0.3637 m |
| outcome type changed (collision → timeout) | 5 | **+2.9521 m** | +2.4216 m |

**The entire gain lives in the outcome-type-changed stratum.** Where both arms end
the same way, AR(1) is marginally *worse* on final distance.

The mechanism is arithmetic, not navigational: an episode that collides at step 86
stops 6.05 m from goal; one that survives to step 1200 reaches 2.32 m. The endpoint
measures survival time, not navigation quality.

I chose this co-primary and did not pre-specify the stratified version. That is a
design error in the protocol I wrote, not a fault in the execution.

---

## 2. All six AR(1) timeouts deadlock at the same point

| seed | final (x, y) | goal distance |
|---|---|---|
| 791101204 | (0.996, −2.106) | 2.32 m |
| 791101205 | (1.291, −2.053) | 2.41 m |
| 791101208 | (1.019, −2.093) | 2.32 m |
| 791101209 | (1.173, −2.072) | 2.37 m |
| 791101210 | (1.256, −2.065) | 2.40 m |
| 791101211 | (1.290, −2.068) | 2.42 m |

Six independent seeds terminate inside a box roughly 0.30 m × 0.06 m. Over the final
300 steps (30 s) the robot covers a total path length of **0.67–0.75 m**.

Control-arm timeouts do **not** cluster there (4.53, 1.16), (2.72, −3.01),
(2.06, −1.86) — the i.i.d. arm mostly collides before reaching this point.

---

## 3. What is blocking it — not the planner

Diagnostics over the final 300 steps of every AR(1) timeout:

| quantity | value |
|---|---|
| candidate feasible fraction | **1.000** (median) |
| hard violations | **0%** of steps |
| forecast count > 0 | **0%** of steps |
| nearest dynamic obstacle | 1.39–3.38 m |
| zero-feasible steps | **0 / 300** |
| applied \|v\| | **0.022–0.028 m/s** |
| **safety_override** | **74–78% of steps** |

Every candidate is feasible. The risk layer is silent. No dynamic obstacle is within
1.3 m. And the robot is crawling at ~2 cm/s because **the safety arbiter is
overriding three quarters of all commands.**

Recorded `safety_reason` during those overrides, across all timeout tails:

* `front_obstacle_slow` — 1530 steps
* `near_body_hard_stop` — 284 steps

The governing configuration (`perception.scan_guard`):

```
front_soft_block_distance:   0.45     # front obstacle inside 0.45 m -> soft block
front_soft_block_max_speed:  0.04     # ... capped at 4 cm/s
front_slow_distance:         0.85
front_slow_min_scale:        0.45
front_stop_distance:         0.35
near_body_stop_radius:       0.35
front_angle_deg:             35.0
```

Observed median `clearance` at the deadlock is **0.15–0.18 m**, inside both
`front_stop_distance` and `near_body_stop_radius` (0.35 m). Observed speed
(0.022–0.028 m/s) is consistent with the 0.04 m/s soft-block cap, and 0.024 m/s over
300 steps predicts 0.72 m of travel — matching the observed 0.67–0.75 m.

**Conclusion: chapter 1's zero success rate is gated by a scan-based proximity speed
governor, not by sampling geometry, candidate coverage, risk thresholds, or the
probabilistic arbitration path.** Chapter 1 is an irregular-spiral map; where its
corridor front-clearance falls below 0.45 m, the robot is permanently throttled to
4 cm/s and cannot complete the route within 1200 steps.

---

## 4. What the screen did establish

The intervention worked at its intended target, and that should not be lost:

| indicator | i.i.d. | AR(1) | change |
|---|---|---|---|
| zero-feasible steps | 230 | 92 | **−60%** |
| hard violations | 443 | 187 | **−58%** |
| fallback activations | 1240 | 645 | **−48%** |
| collisions | 9/12 | 6/12 | −25 pp |
| stuck-step rate | 24.69% | 21.89% | −2.8 pp |

The offline diagnosis (RC-1, P0.2) transferred: correlated sampling substantially
reduces how often the planner exhausts its feasible candidate set, and converts early
collisions into survival. This is a real, mechanistically coherent, closed-loop
confirmation of the sampling-geometry hypothesis.

It then ran into a different wall.

---

## 5. Assessment of the confirmatory

**Recommendation: do not run the n ≥ 20 confirmatory as designed.**

`PASS_TO_CONFIRMATORY` is a permission, not an obligation. Declining to spend the
budget is not a protocol violation; reinterpreting the screen to claim efficacy would
be, and that is not proposed here.

Reasons:

1. The primary endpoint (`safe_success`) is 0/12 in both arms and is capped by the
   speed governor. A larger sample re-measures a quantity the intervention cannot
   move. Expected outcome: 0/N vs 0/N.
2. The co-primary that produced the pass is confounded (§1) and would remain
   confounded at n = 20. More seeds do not fix a mediated endpoint.
3. The known blocker is a fixed configuration threshold, diagnosable in hours from
   artifacts already in hand.

This is the failure mode I warned about in my first assessment of this project —
*"you cannot measure an improvement against a control at zero; build a difficulty
ladder and calibrate until the baseline succeeds 50–70%"* — and then reproduced when
designing this screen. I selected chapter 1 for worst-coverage and "headroom on
progress" without verifying that its baseline could ever succeed. The recorded
margin-0 control (1200 steps, 4.76 m from goal) was sufficient warning and I did not
act on it.

---

## 6. Recommended sequence

1. **Diagnose the governor deadlock.** Is the throttle escapable? Does
   `front_soft_block_max_speed = 0.04` admit any recovery, or is a robot inside
   0.45 m front clearance permanently trapped? Read `safety/arbiter.py` (1,641 lines,
   still unread) and the scan-guard producer. Cheap, offline.
2. **Establish whether chapter 1 is winnable at all** under the frozen safety
   contract. If its corridors are narrower than the governor's thresholds, no planner
   change can succeed there and it is unusable as a success-endpoint benchmark.
3. **Then choose the confirmatory endpoint deliberately.** If AR(1) is to be claimed,
   collision reduction (9/12 → 6/12) is the endpoint with a real effect and it matches
   the paper's already-confirmed strength. That requires re-registering the
   confirmatory with collision as primary, on fresh seeds, before seeing them — which
   is a legitimate use of a screen, not p-hacking.
4. **Investigate the 0.2% guided-candidate share** separately. The supervised actor is
   contributing 0.2% of candidates in the complex scene, so the learning component is
   close to inert there. This incidentally validates P0.2's Gaussian-only modelling
   (see `deep_read_corrections_v1` §Correction 2, which flagged the unmodelled actor
   source as a risk — the risk is small).

---

## 7. Open safety item

One treatment-only collision (seed 791101206: control timed out, AR(1) collided at
step 90). Below the `FAIL_SAFETY` threshold of 2 discordant pairs, so the gate
correctly did not trip. It should still be examined, since AR(1) produces larger
sustained excursions by construction and an early collision at step 90 is a
short-horizon failure rather than the late-window pattern seen elsewhere.

---

## 8. Deadline miss rate

Mean episode deadline-miss rate moved 94.10% → 98.88%, despite median paired
planner-P95 improving by 28.61 ms. Both arms miss the 100 ms control deadline on
essentially every cycle. Whatever is decided about the sampler, **no real-time claim
is defensible for the complex scene in its current configuration.**
