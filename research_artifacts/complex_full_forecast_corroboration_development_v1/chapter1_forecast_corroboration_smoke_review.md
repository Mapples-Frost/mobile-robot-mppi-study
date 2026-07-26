# Chapter 1 forecast-corroboration smoke review

## Frozen run

- Scene: `chapter1`
- Seed: `790200025`
- Deliberately truncated horizon: 400 of 1500 steps
- Result: no collision, max-steps termination
- Actual minimum clearance: 0.2268 m
- Final goal distance: 6.5172 m
- Trajectory length: 3.5290 m
- Maximum route progress: 0.9220 m

## Targeted mechanism result

The robot cleared the entrance region instead of looping around the start:

- y advanced from -3.20 m to -1.11 m;
- goal distance improved by 0.594 m;
- route progress more than doubled relative to the previous 1500-step run
  (0.9220 m versus 0.4348 m);
- the final 20 decisions were low-risk, non-emergency forward motion.

The probability contract also behaved as intended:

- 94 steps were forecast-corroborated;
- critical-distance override was not needed;
- low-risk emergency selections fell to 68/205, with the remainder attributable
  to the finite intent hold following a corroborated conflict.

## Interpretation boundary

This is a positive mechanism smoke test, not an episode completion result. No
new parameter change is justified. The next run must use a fresh seed, the same
shared configuration, and the map's full 1500-step horizon.
