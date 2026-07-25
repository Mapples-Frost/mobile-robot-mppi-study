# Single-dynamic-obstacle v5 held-out qualification

## Purpose and evidential scope

This is a one-shot engineering generalization gate for the C3 dual-horizon
deadline supervisor. It is not formal effect estimation and does not replace
the future preregistered v5 formal experiment. The C3 candidate was frozen
after the repeated development panel passed at commit
`eeb676e8b8bb60fdbe752270504e688d3b41a579` with development bundle
`a277f21b2ca344a56c792939c73fb88904ea19155368156761b13da32d35ead2`.
No further tuning on the C3 development seeds is allowed.

## Units, arms, blocking, and seed selection

- Independent unit: one complete episode seed.
- Cohort: 32 new paired seeds, ID/OOD 16 each, for 64 episode jobs.
- Arms: `V4_full_frozen` and `V5_dual_horizon_full`.
- Each seed is a complete block and uses common random numbers. Both arms run
  in one worker; arm order alternates within split, is exactly 8/8 in both
  directions, and the 32 blocks are shuffled with schedule seed `752699041`.
- ID candidates begin at `750100001`; OOD candidates begin at `750200001`.
  Within each split, the first 16 ascending seeds that pass the frozen ghost
  constant-speed geometric conflict certificate are selected.
- Selection executes no controller and reads no controller outcome. All prior
  program seeds (`730000000` through `749999999`) are excluded. Future v5
  formal ranges `760200001...760201999` and `760300001...760301999` are
  reserved and excluded.

The sealed registry is
`configs/seeds/single_dynamic_obstacle_v5_held_out_qualification_sealed.yaml`.
Its file SHA-256 is
`3a606928a5621128a411c91eae075b9ffc1a808124fcf04c7b839a57a0b07eea` and
its canonical schedule SHA-256 is
`fcd0ddbbc2eb1489085f77a6557380caa8bcd0a4c3e4746595e99160ef51f87e`.

## Frozen candidate

Only the C3 deadline-supervisor fields differ from v4 Full: urgency reserve is
5 steps, authority reserve is 0, clear hold is 5 cycles, the speed floor is
0.30 m/s, the required-speed margin is 1.10, and the heading gate is 0.20 rad.
The supervisor cannot change steering, cannot exceed 0.35 m/s, and cannot
override risk, TTC, closing, clearance, heading, or renewed-hazard vetoes.
The predictor, collision risk, threshold, safety margin, obstacle generator,
Actor, ICODE, HSS, 36-step horizon, 0.1 s period, 400-step deadline, and 600
rollouts per decision remain frozen.

## Qualification gates

Every gate below must pass without post-outcome amendment:

1. paired safe-success net gain at least 2/32;
2. at least two V4 safe non-completions converted to safe success;
3. no V4 success lost, with non-negative net success change in ID and OOD;
4. no new paired collision and total candidate collisions no greater than V4;
5. mean minimum-clearance and conflict-window q05-clearance loss each no more
   than 0.03 m;
6. aggregate zero-speed-under-risk and stuck-step reductions each at least
   -10% (a bounded non-inferiority health check);
7. total direction switches do not increase and three-phase oscillations do
   not increase;
8. every controller decision uses exactly 600 rollouts.

Passing authorizes preparation and preregistration of a future formal v5
registry, not execution of that experiment. Failure closes the single-obstacle
optimization path; the failed bundle is retained and work moves to the
multi-obstacle study without outcome-driven replacement of qualification seeds.

## Execution and outcome lock

The complete registry, protocol, gates, analyzer, tests, and hashes are committed
and tagged before execution. Progress files expose structural completion only.
No episode metric is read until all 32 blocks and 64 episode jobs are complete.
There is no automatic algorithm-failure retry and no seed replacement. Formal
server execution remains unauthorized; this engineering qualification runs only
on the local configured environment.
