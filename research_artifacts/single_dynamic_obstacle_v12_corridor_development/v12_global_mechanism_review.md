# v12 global mechanism review

## Frozen development result

- Paired development jobs: 16 seeds × 2 arms.
- V4: 10 successes, 5 collisions.
- V12: 10 successes, 6 collisions.
- Safe-success gain: 0.
- Prevented collisions: 0.
- New paired collisions: 1.
- Lost V4 successes: 1.
- Mean minimum-clearance delta: -0.0641 m.
- Corridor mechanism exercise: 221 active steps, including 103 turn steps.

## Paired transitions

- OOD seed 750400003 changed from safe non-completion to success.
- ID seed 750300029 changed from V4 success/no collision to V12
  failure/collision.

## Interpretation

The corridor state machine was exercised extensively, so this is not an
inactive-feature or missing-diagnostic failure. A finite turn-then-straight
commit can recover one completion, but the same unconditional side commitment
can carry a previously safe trajectory into collision. The mechanism therefore
does not provide a valid safety-preserving improvement over frozen V4.

## Decision

Stop the v12 reachable-side-corridor mechanism family. Do not retune turn
steps, commit steps, speed, or seed selection. Do not run a fresh held-out
qualification and do not start a formal server experiment. Preserve V4 as the
current defensible single-obstacle result unless a future study is explicitly
authorized to replace the controller architecture rather than tune this
escape family.
