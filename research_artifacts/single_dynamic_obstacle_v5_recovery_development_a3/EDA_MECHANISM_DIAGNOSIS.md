# A3 mechanism diagnosis

## Scope and integrity

- This is outcome-informed engineering development, not confirmatory effect estimation.
- All 16 paired episode jobs completed without runtime failures.
- The 91 files listed in `artifact_manifest.json` were independently re-hashed after execution; all SHA-256 values matched.
- The frozen V4 arm remained unchanged at implementation commit `8c677f2cb218b7ecca58b29d07c866366f65ea0e`.

## Gate outcome

A3 failed the frozen development gate.

| Metric | V4 | A3 V5 | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 4/8 | no gain |
| Collisions | 2/8 | 1/8 | one collision avoided |
| Mean minimum clearance | 0.194 m | 0.228 m | +0.034 m |
| Mean stuck steps | 23.5 | 30.875 | 31.4% worse |
| Mean zero-speed risk steps | 19.75 | 25.25 | 27.8% worse |
| Recovery-challenge conversions | - | 0/3 eligible failures | below the required 2 |

No V4 success was lost and the rollout budget remained exactly 600 per decision.

## Per-seed findings

- `740300251`: A3 reduced final goal distance from 1.126 m to 0.425 m and stuck steps from 33 to 19, but the stricter 0.15 rad alignment tolerance added delay and the episode timed out immediately before arrival. A1/A2 had converted this seed.
- `740200006`: A3 avoided A2's new collision and reduced stuck/zero-risk steps by two relative to V4, but finished at 0.536 m. Its last recovery interval ended at 21.5 s; the controller then needed the rest of the 40 s horizon to approach the goal.
- `740200011`: A3 converted the V4 collision into a safe trajectory and finished at 0.446 m, but this sentinel is not counted as a recovery-challenge conversion. Its last recovery interval ended at 34.6 s, leaving too little time for nominal terminal completion.
- `740200191`: the remaining collision occurred at 24.1 s, around seven seconds after the last recovery interval ended. At impact the temporal TTC was 0.233 s and the predicted maximum probability was already 1.0, too late for a new recovery cycle. This is consistent with lingering in the conflict corridor after premature release, not with unsafe motion during recovery.

## Mechanism conclusion

The clearance-gated alignment creep prevented A2's collision regression on `740200006`, but it reintroduced near-zero-speed alignment. Across the four principal difficult seeds, A3 contained 53 alignment-mode steps, 51 of which did not use the gated creep. More importantly, the successful recovery intervals were released after short local progress, while the robot was still exposed to a later encounter or had insufficient horizon remaining to finish.

The next attempt should therefore change mechanism level rather than continue tuning the same creep threshold:

1. preserve all existing hard-risk, temporal-closing, and front-clear abort conditions;
2. retain clearance-trend-gated low-speed alignment to avoid unsafe unconditional creep;
3. restore the 0.20 rad alignment tolerance used by A1/A2;
4. add a bounded, safety-abortable post-conflict corridor-clearance commitment long enough to avoid immediate release back into nominal stagnation;
5. keep every frozen component and the 600-rollout/H=36/dt=0.1 contract unchanged.

The A3 panel remains a contaminated development panel and cannot support paper claims. Any selected implementation must pass a new disjoint held-out qualification before a formal registry is prepared.
