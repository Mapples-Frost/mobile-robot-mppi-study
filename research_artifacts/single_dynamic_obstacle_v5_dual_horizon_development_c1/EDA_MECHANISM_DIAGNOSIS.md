# C1 mechanism diagnosis

## Scope and integrity

- C1 is repeated, outcome-informed engineering development, not confirmatory evidence.
- All 16 paired jobs completed with zero runtime failures and empty stderr.
- All 91 manifest entries independently re-hashed successfully.
- Protocol and implementation freeze commit: `4e0445f86d49ebb3ababefef189af22e063388f7`.
- Artifact bundle SHA-256: `761f058d4f09c270e92ee2cc3cb6018b7d8a4aada6a77c7758eaedf906408870`.

## Frozen gate outcome

C1 failed solely because it converted one rather than two eligible challenge episodes. It passed all other frozen checks: no new paired collision, collisions not worse, no lost V4 success, exact 600-rollout decisions, preserved clearance, improved final progress, improved stuck and zero-speed exposure, and no added direction switch or three-phase oscillation.

| Metric | V4 | C1 | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 5/8 | +1 success |
| Collisions | 2/8 | 2/8 | no change |
| Eligible challenge conversions | - | 1 | required 2 |
| Mean minimum-clearance delta | - | - | +0.025 m |
| Mean final-distance delta | - | - | -0.116 m |
| Relative stuck-step reduction | - | - | 12.77% |
| Relative zero-speed-risk reduction | - | - | 6.33% |
| Direction-switch increase | - | - | 0 |
| Three-phase-oscillation increase | - | - | 0 |

## Dual-horizon evidence

The structural separation worked as intended. On `740200006`, C1 retained B2's early urgency behavior and remained safely active through the final two cycles. Its final five cycles all used the unchanged 0.35 m/s cap under clear guards. Active steps increased from B2's 30 to C1's 32, and final goal distance improved from 0.322691 m to 0.314809 m.

The episode still missed the 0.300 m tolerance by 0.014809 m. Observed terminal goal-distance reduction per capped clear cycle was about 0.03 m, so one additional earlier safe cycle is sufficient in scale. This is not evidence to alter the speed cap, tolerance, deadline, or safety guards.

## C2 bounded amendment

C2 is the second attempt in the dual-horizon family. It changes only the urgency reserve from 2 to 3 steps while retaining zero authority reserve. This makes the same online feasibility calculation urgent approximately one cycle earlier but does not extend the episode, change terminal authority, alter steering, increase the 0.35 m/s cap, or weaken any risk/TTC/closing/clearance/heading guard.

All other candidate fields, the repeated panel, arm order, common random numbers, gate, and frozen components remain identical to C1. If C2 fails, its evidence is retained; one further dual-horizon attempt is allowed only with another specific falsifiable diagnosis. Three failed attempts or two safety regressions close this family.
