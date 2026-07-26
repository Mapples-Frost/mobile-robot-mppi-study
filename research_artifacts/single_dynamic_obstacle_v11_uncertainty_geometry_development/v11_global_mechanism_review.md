# v11 global mechanism review

## Evidence

- Paired development jobs: 16 seeds × 2 arms.
- V4: 10 successes, 5 collisions.
- V11: 11 successes, 5 collisions.
- Paired safe-success gain: +1 episode; no lost V4 successes.
- Paired collision change: 0; no prevented collisions and no new paired collisions.
- Mean minimum-clearance delta: -0.0506 m.
- Emergency mechanism exercise: passed.
- Candidate-clearance nonnegative audit: failed.

## Interpretation

The uncertainty-fused geometric escape contract is exercised and can convert at least one
previous non-completion into a success, but it does not reduce collision count on this
development set. The remaining failures are not explained by a missing emergency trigger
alone. Several paired failures still enter collision with negative minimum clearance after
the geometric escape is active, indicating that the current forward-turn command is not a
complete closed-loop avoidance policy. The lower clearance and unchanged collision count
also make a fresh held-out qualification scientifically unjustified.

## Decision

Do not run a fresh held-out qualification for v11 and do not start the formal server run.
Freeze the v11 code and evidence. Any future single-obstacle attempt must change the
closed-loop mechanism family (candidate generation/side-selection and state-machine
coordination), rather than retuning the current uncertainty threshold, turn magnitude, or
commit duration.
