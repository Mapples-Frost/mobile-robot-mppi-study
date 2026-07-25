# B1 mechanism diagnosis

## Scope and integrity

- B1 is repeated, outcome-informed engineering development on the A1-A5 panel. It is not evidence for confirmatory effect estimation.
- All 16 paired jobs completed with zero runtime failures; stderr is empty.
- All 91 entries in `artifact_manifest.json` were independently re-hashed after execution; every SHA-256 matched.
- The implementation and protocol were frozen at `165da4d4b7aea7b78acd6f34bf550d438b67cb9e` before execution.
- The immutable artifact bundle SHA-256 is `68f2856f75c61b6e6ff611b5a3c5678ce9c318ef9aa42895d3a14d2dd78b69ba`.

## Frozen gate outcome

B1 failed the prospective, pre-execution development gate. The gate is not relaxed after observing these outcomes.

| Metric | V4 | B1 candidate | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 7/8 | +3 successes |
| Collisions | 2/8 | 1/8 | -1 collision; no new paired collision |
| Mean minimum clearance | 0.194 m | 0.259 m | +0.065 m |
| Mean final goal distance | - | - | 0.374 m reduction |
| Mean stuck steps | 23.50 | 22.75 | 3.19% reduction |
| Mean zero-speed risk steps | 19.75 | 21.75 | 10.13% increase |
| Direction switches | - | - | +13 total |
| Three-phase oscillations | - | - | +6 total |
| Eligible challenge conversions | - | 2 | required 2 |

All V4 successes were preserved and every controller decision retained exactly 600 rollouts. B1 passed the frozen task, paired-safety, clearance, final-progress, stuck-step, and rollout-contract checks. It failed the zero-speed-risk bound by 0.13 percentage points and failed both no-increase oscillation checks.

## Per-seed attribution

- `740200006`: the deadline supervisor activated for four steps and converted a V4 safe timeout into a safe success, ending 0.278 m from the goal.
- `740200011`: the supervisor activated for twelve steps and converted a V4 collision into a safe success, ending 0.275 m from the goal.
- `740300251`: converted from safe timeout to safe success, but the deadline supervisor never activated. Direction switches rose from 3 to 7 and three-phase oscillations from 0 to 3.
- `740300001`: both arms remained safe successes, the deadline supervisor never activated, but direction switches rose from 1 to 7 and three-phase oscillations from 0 to 1.
- `740200011`: despite its safety and completion conversion, direction switches rose from 3 to 6 and three-phase oscillations from 0 to 2.
- `740200191`: both arms retained the inherited collision; B1 introduced no new paired collision.

## Mechanism isolation conclusion

The online deadline-feasibility supervisor has a strong task/safety signal on this development panel, but B1 does not isolate that mechanism. Its candidate also inherits the A5 post-conflict recovery overrides. Most oscillation regression occurs in seeds where the deadline supervisor has zero active steps, so the observed regression cannot be attributed primarily to deadline intervention.

B2 therefore changes experimental structure, not deadline parameters: it uses the same frozen V4 Full control and adds only the eleven `dynamic_deadline_*` fields. Every A5 `dynamic_recovery_*` candidate override is removed. The gate, panel, common random numbers, rollout budget, horizon, control period, episode deadline, goal tolerance, prediction stack, risk thresholds, HSS, Actor, ICODE, and safety boundaries remain unchanged.

This is the second attempt in the deadline-feasibility mechanism family. If B2 fails, the result must be retained. A third local attempt is allowed only if B2 provides a specific, falsifiable attribution that does not require broad retuning. Three non-improving attempts or two safety regressions trigger another global mechanism review.
