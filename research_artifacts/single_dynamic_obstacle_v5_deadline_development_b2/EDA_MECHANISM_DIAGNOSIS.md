# B2 mechanism diagnosis

## Scope and integrity

- B2 is repeated, outcome-informed engineering development, not confirmatory effect estimation.
- All 16 paired jobs completed with zero runtime failures and empty stderr.
- All 91 manifest entries independently re-hashed successfully after execution.
- The protocol and implementation were frozen at `a405c6c59da0f77dbc24f5652185a8eadb7ae7c6` before execution.
- Artifact bundle SHA-256: `4a479fa2c1e785a3537586f34e04d260e41b7ef6816b3513a2d194de577ac0b7`.

## Frozen gate outcome

B2 failed the unchanged prospective gate solely because it converted one rather than two eligible challenge episodes. No gate is relaxed.

| Metric | V4 | B2 isolated deadline | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 5/8 | +1 success |
| Collisions | 2/8 | 2/8 | no change; no new paired collision |
| Eligible challenge conversions | - | 1 | required 2 |
| Mean minimum clearance | 0.194 m | 0.219 m | +0.025 m |
| Mean final goal distance | - | - | 0.115 m reduction |
| Mean stuck steps | 23.50 | 20.50 | 12.77% reduction |
| Mean zero-speed risk steps | 19.75 | 18.50 | 6.33% reduction |
| Direction switches | - | - | no increase |
| Three-phase oscillations | - | - | no increase |

Every decision retained exactly 600 rollouts, all V4 successes were preserved, and all safety and mechanism-regression checks passed. Removing the A5 overrides eliminated the B1 oscillation regression, confirming that B1's extra direction switches and three-phase oscillations did not originate in the deadline supervisor.

## Falsifiable terminal-step diagnosis

- `740300251` was converted from safe timeout to safe success. The isolated supervisor activated for one step, reduced final goal distance by 0.836 m, increased minimum clearance by 0.201 m, reduced stuck steps by 24, and reduced zero-speed-risk steps by 10.
- `740200006` remained a safe timeout but ended at 0.322691 m, only 0.022691 m outside the frozen 0.300 m success tolerance. The supervisor was active for 30 steps and introduced no direction switch or oscillation.
- At control step 398, `740200006` was still safely deadline-supervised at the 0.35 m/s cap. At steps 399 and 400, `dynamic_deadline_available_steps` became zero because the B2 protocol reserved two terminal steps, so the supervisor relinquished authority despite a clear guard and continued goal alignment.
- The final two applied speeds were approximately 0.330 and 0.300 m/s. Retaining bounded deadline authority for those two clear terminal cycles provides about 0.06 m commanded travel, exceeding the observed 0.022691 m shortfall without changing steering, risk gates, or speed cap.

## B3 decision and stopping rule

B3 is a single-parameter correction: `dynamic_deadline_reserve_steps` changes from 2 to 0. All other candidate fields, the repeated panel, arm order, common random numbers, frozen V4 control, safety guards, gate, rollout budget, and episode deadline remain identical to B2.

This is the third attempt in the deadline-feasibility mechanism family. If B3 fails any frozen gate, local deadline tuning stops and a new global mechanism review is mandatory. If B3 passes, it may advance only to a new, disjoint, unopened held-out qualification; the repeated development panel cannot support an effect claim.
