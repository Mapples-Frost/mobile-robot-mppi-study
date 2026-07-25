# A4 mechanism diagnosis

## Scope and integrity

- A4 is outcome-informed engineering development and cannot be used for confirmatory effect estimation.
- All 16 paired jobs completed with zero runtime failures.
- All 91 files in `artifact_manifest.json` were independently re-hashed; every SHA-256 matched.
- The implementation was frozen at `7c5f665d1201beb1495e84e202036d4c9c1474a4` before execution.

## Gate outcome

A4 failed the frozen development gate, despite a material task improvement.

| Metric | V4 | A4 V5 | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 6/8 | +2 successes |
| Collisions | 2/8 | 2/8 | no aggregate reduction; one new OOD collision |
| Mean minimum clearance | 0.194 m | 0.249 m | +0.056 m |
| Mean stuck steps | 23.5 | 24.125 | 2.7% worse |
| Mean zero-speed risk steps | 19.75 | 24.125 | 22.2% worse |
| Eligible challenge conversions | - | 1 | below required 2 |

No V4 success was lost and all decisions retained 600 rollouts.

## Per-seed findings

- `740200006` became a safe success (final distance 0.294 m), with 16 fewer stuck steps than V4. This confirms that additional post-conflict mobility can convert at least one persistent safe non-completion.
- `740200011` changed from a V4 collision to a safe success (final distance 0.268 m). This is a strong mechanism success, although the sentinel is not counted in the challenge-conversion gate.
- `740200191` still collided at 24.2 s. The 8 s recovery intervals improved forward mobility but ended before the final closing encounter; the last six cycles show existing active escape and hard-risk logic reacting, but the collision was unavoidable from the resulting state.
- `740300251` regressed from a safe non-completion to a collision at 16.8 s. The first recovery stayed in advance mode from 2.1 to 9.9 s and moved goal distance from 8.94 to 6.48 m. The later collision occurred after recovery release, showing that unconditional long commitment changed encounter timing adversely.
- All four V4 successes were preserved.

## Mechanism conclusion

The evidence rejects an unconditional 8 s commitment for every recovery. It also shows that the original 0.8 s release is too short in stalled trajectories. The appropriate next mechanism is conditional, not another fixed-duration compromise:

1. restore the short, safety-abortable recovery commitment;
2. after planner release, monitor online goal progress over a frozen finite window;
3. re-enter a short recovery only when progress is below a frozen threshold and all existing front-clear, temporal-closing, TTC, and predicted-risk guards are clear;
4. cap the number and duration of re-engagements;
5. cancel the watch on any renewed hazard so the existing dynamic escape remains authoritative.

This uses only online controller state and does not expose conflict-certificate truth or future obstacle trajectories to the controller. A5 is the final attempt in this mechanism family before a mandatory broader state-machine review.
