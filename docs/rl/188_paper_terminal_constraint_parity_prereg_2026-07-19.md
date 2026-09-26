# L188 paper-controller terminal-constraint parity repair

Date: 2026-07-19  
Status: frozen before implementation and before seeds 556--557 are run  
Scope: deterministic compatibility repair; sealed seeds 561--565 stay closed

## Triggering evidence

The L187 terminal-weight candidate passed its seed-553 diagnostic but failed
the pre-registered confirmation Gate because Full Proposed succeeded on seed
554 and failed on seed 555.  It nevertheless had zero collisions, 18.3% lower
mean cross-track RMSE than ordinary fixed, and 0.3% lower mean control jerk.
The only failed criterion was two-of-two endpoint success.

Inspection of the saved seed-555 terminal state exposed a controller parity
defect:

```text
terminal phase:                 true
final goal distance:            0.3373 m
true endpoint bearing error:   +0.3740 rad
reported alignment active:      false
executed angular velocity:     -0.9088 rad/s
```

The base MPPI and legacy RL-driven controller apply the configured local
terminal-bearing law after slew-rate clipping.  The paper-faithful
`PaperRLDrivenMppiController` called the shared sampling constraint but did not
apply that final action constraint or report its bearing diagnostics.  The
negative angular command above therefore turned away from the endpoint,
creating a terminal limit cycle.

## Frozen implementation change

No cost, tolerance, horizon, rollout budget, network, checkpoint, or safety
parameter may change.  The repair will:

1. extract the already-tested final terminal-action rule into one shared
   `RLDrivenMppiController` helper;
2. use that helper in both the legacy and paper-faithful RL-driven solvers;
3. retain the configured terminal speed limit and heading gate;
4. preserve rotate-in-place control when translation is gated;
5. recompute the returned predicted trajectory after the final action;
6. expose identical bearing/alignment diagnostics from both solvers.

The expected behavior is:

```text
large bearing error -> v_cmd is gated, omega_cmd turns toward endpoint
small bearing error -> bounded forward translation resumes
```

## Required tests

The change is accepted only if:

1. a paper-controller unit test fails on the old implementation and passes on
   the repaired implementation;
2. a misaligned terminal state produces zero first-step translation and an
   angular command with the correct sign;
3. disabling the terminal constraints retains previous behavior;
4. the existing paper-fidelity tests still pass;
5. the full repository test suite passes.

## Development confirmation

After unit tests, the frozen L187 terminal weight 50 is evaluated on previously
unused development seeds 556--557 with all four randomized factorial arms.
The L188 Gate passes only if:

1. Full Proposed succeeds on both seeds;
2. Full Proposed has zero collisions;
3. its mean cross-track RMSE is no more than 5% worse than ordinary fixed;
4. its mean control jerk is no more than 10% worse than ordinary fixed.

No result from seeds 551--557 may be described as sealed confirmation.  The
two L186 holdout geometries and seeds 561--565 remain untouched until this Gate
passes.
