# B3 mechanism diagnosis

## Scope and integrity

- B3 is repeated, outcome-informed engineering development, not confirmatory evidence.
- All 16 paired jobs completed with zero runtime failures and empty stderr.
- All 91 manifest entries independently re-hashed successfully.
- Protocol and implementation freeze commit: `6632a300d83fda06c69843e307b41aac05050299`.
- Artifact bundle SHA-256: `ac28a3c81f9f1beb6b13178fed869fc8e13a8dd6d214256c28371e579230cec7`.

## Frozen gate outcome

B3 failed the unchanged gate solely because it converted one rather than two eligible challenge episodes. All other checks passed: no new paired collision, aggregate collisions not worse, no lost V4 success, exact 600-rollout decisions, preserved clearance, improved final progress, improved stuck and zero-speed exposure, and no added direction switch or three-phase oscillation.

| Metric | V4 | B3 | Effect |
|---|---:|---:|---:|
| Successes | 4/8 | 5/8 | +1 success |
| Collisions | 2/8 | 2/8 | no change |
| Eligible challenge conversions | - | 1 | required 2 |
| Mean minimum-clearance delta | - | - | +0.025 m |
| Mean final-distance delta | - | - | -0.113 m |
| Relative stuck-step reduction | - | - | 12.77% |
| Relative zero-speed-risk reduction | - | - | 6.33% |
| Direction-switch increase | - | - | 0 |
| Three-phase-oscillation increase | - | - | 0 |

## B3 falsification

The B2 hypothesis that removing the two-step reserve would simply extend terminal authority was false. On `740200006`, the candidate ended at 0.340931 m rather than B2's 0.322691 m, and active supervisor steps fell from 30 to 18.

The reason is structural: `dynamic_deadline_reserve_steps` serves two coupled roles in the implementation. It both shortens the time horizon used to compute required speed and decides whether any authority remains. Reducing it from 2 to 0 made required speed less urgent earlier in the episode, delaying activation, even though it permitted nominal terminal eligibility. A single reserve parameter therefore cannot simultaneously provide early urgency and final-cycle authority.

## Mandatory family stop

B1, B2, and B3 constitute three attempts in the single-horizon deadline-feasibility family. Per the frozen stopping rule, no B4 parameter adjustment is allowed. The family is closed and a global mechanism review is required before further development.

The next admissible direction is a structurally different dual-horizon supervisor: retain the B2 urgency horizon (two-step reserve) but decouple it from an independently guarded terminal-authority horizon (zero-step reserve). This is a code-level factor separation, not another threshold tune. It must receive a new mechanism-family name, protocol, tests, freeze commit, and unchanged safety/task gate before execution. If pursued, renewed hazard, risk, TTC, closing, front-clearance, heading, 0.35 m/s speed cap, 600 rollouts, and all V4 components remain unchanged.
