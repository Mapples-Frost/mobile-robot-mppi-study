# A5 mechanism diagnosis

## Scope and integrity

- A5 is repeated, outcome-informed engineering development. It is not evidence for confirmatory effect estimation.
- All 16 paired jobs completed with zero runtime failures.
- All 91 entries in `artifact_manifest.json` were independently re-hashed after execution; every SHA-256 matched.
- The implementation was frozen at `d3cfbda67994dbc0925a7536b11b3b7fa79b74d3` before execution.
- The immutable artifact bundle SHA-256 is `fddefc576455c148f13289874a839bd7a4d981571b5f9986ccd39315ea79bafd`.

## Frozen gate outcome

A5 failed the pre-execution development gate.

| Metric | V4 | A5 V5 | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 5/8 | +1 success |
| Collisions | 2/8 | 1/8 | -1 collision |
| Mean minimum clearance | 0.194 m | 0.259 m | +0.065 m |
| Mean stuck steps | 23.5 | 22.75 | 3.2% reduction |
| Mean zero-speed risk steps | 19.75 | 21.75 | 10.1% increase |
| Eligible challenge conversions | - | 1 | below required 2 |

All V4 successes were preserved and every controller decision retained exactly 600 rollouts. The failed checks were the absolute-zero-collision requirement, two challenge conversions, 20% stuck-step reduction, and 20% zero-speed-risk reduction.

## Per-seed findings

- `740300251` changed from a safe non-completion to a safe success. A5 therefore removed the new OOD collision introduced by A4 while keeping the mobility benefit on this seed. Its final distance improved by 0.840 m, stuck steps fell by 21, and minimum clearance increased by 0.024 m.
- `740200006` remained a safe timeout but ended only 0.319 m from the goal, versus 0.411 m for V4. Stuck steps fell by 18 and minimum clearance increased by 0.038 m. At the final sample it was still moving toward the goal at 0.236 m/s; this was a deadline miss, not a terminal standstill.
- `740200011` changed from a V4 collision to a safe non-completion with 0.364 m minimum clearance. It ended 0.384 m from the goal while still moving toward it at 0.239 m/s. A5 preserved the safety conversion seen in A4 but no longer completed before the 40 s deadline.
- `740200191` remained a collision in both arms. A5 improved final progress by 0.107 m and minimum clearance by 0.013 m but did not change the inherited collision outcome.
- The four V4-success sentinels were all preserved.

## Progress-watch falsification

The intended A5 mechanism did not activate on this panel:

- progress-watch exposure ranged from 60 to 167 steps per V5 episode;
- `dynamic_recovery_progress_watch_trigger_count` was zero in all eight V5 episodes;
- `dynamic_recovery_progress_reentry_count_max` was zero in all eight V5 episodes.

This is not evidence that the re-entry thresholds need another local adjustment. The two most relevant safe timeouts were making more than 0.04 m of online goal-distance progress per 0.5 s window, so a binary stall detector correctly regarded them as moving. Their actual failure was insufficient remaining-time-normalized progress. A fixed-duration recovery and a stall-triggered recovery therefore miss opposite sides of the same state distinction.

## Global mechanism conclusion

A3-A5 have exhausted the fixed/conditional post-conflict commitment family:

1. short unconditional commitment is safe but too weak for some deadlines;
2. long unconditional commitment improves completion but can alter encounter timing and create a collision;
3. stall-triggered re-entry does not activate when the robot is moving safely but too slowly to finish before the frozen deadline.

No A6 parameter tuning should be performed in this family. The next candidate must change mechanism class to an online deadline-feasibility supervisor: compare conservative remaining travel time against remaining episode time, and request bounded mobility only when the existing risk, TTC, closing, heading, and clearance guards certify it. The supervisor must relinquish authority immediately on renewed hazard and must not use truth conflict windows or future obstacle trajectories.

The global review also identifies a development-gate mismatch. Requiring zero absolute V5 collisions asks a recovery package to eliminate inherited V4 collisions, whereas the preregistered scientific safety claim is collision non-inferiority. Any next development protocol should distinguish (a) no newly introduced paired collision, (b) aggregate collision count no worse than V4, and (c) the unchanged held-out/formal non-inferiority criterion. This is an outcome-informed development amendment and cannot be applied retroactively to make A5 pass.

Before any held-out qualification, the new mechanism, diagnostics, gate, seeds, and hashes must be frozen. The held-out seeds must be new, disjoint from A1-A5 and the future 360-seed registry, and remain unopened until execution is complete.
