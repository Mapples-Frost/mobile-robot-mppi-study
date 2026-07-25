# v5 recovery development A1 mechanism diagnosis

This is a post-hoc engineering diagnosis of the sealed A1 development bundle. It
is not part of the A1 artifact manifest and is not confirmatory evidence.

- Source bundle SHA-256: `2aae03584f1649868129096e2168fa8e595c628f1645b955292959dc3dcb5a03`
- Episodes: 16 (8 paired blocks); missing episode artifacts: 0
- Manifest entries verified: 91/91; hash failures: 0
- Episode implementation Git SHA: `412a221e32c73cc79d98d9fe7dc503e1dbbe81f3`

## A1 outcome summary

| Endpoint | V4 frozen | V5 A1 | Direction |
|---|---:|---:|---|
| Safe successes | 4/8 | 5/8 | +1 episode |
| Collisions | 2/8 | 1/8 | -1 episode |
| Mean minimum clearance | 0.194 m | 0.249 m | +0.055 m |
| Mean stuck steps | 23.5 | 38.9 | worse |
| Mean zero-speed-risk steps | 19.8 | 25.5 | worse |

A1 converted one of the four predeclared recovery challenges and lost none of
the four locally reproduced V4 successes. It nevertheless failed the frozen
engineering gate: the minimum was two conversions, the absolute V5 collision
limit was zero, and both mobility diagnostics had to improve by at least 20%.

## Mechanism localization

Across the eight V5 trajectories, recovery produced 157 alignment cycles and
129 advance cycles. During 142 recovery-active cycles the applied translational
speed remained below 0.02 m/s. Forty-four of those alignment stops coincided
with the predeclared risk-or-raw-closing condition. The long alignment segments
occurred after the front guard was clear and the selected predicted collision
probability was typically near zero; examples included target-bearing errors of
about 1.4--1.8 rad followed by saturated in-place rotation.

This shows that A1's progressive acceleration and forward commitment work once
the robot is aligned, but the preceding rotation-only alignment phase recreates
the very post-conflict immobility that v5 is intended to remove. It also explains
why one safety-gain sentinel changed from an early collision to a safe timeout:
the longer surviving episode accumulated many stationary alignment steps.

## A2 engineering hypothesis

When all existing recovery-entry conditions remain satisfied (front guard clear,
temporal closing clear, and predicted risk below the frozen entry threshold),
replace rotation-only alignment with a tightly turning 0.10 m/s forward creep.
Any new guard, temporal-closing, or abort-risk event still cancels recovery before
translation is applied. The feature remains disabled by default, so V4 behavior
is unchanged. No predictor, risk threshold, rollout, checkpoint, HSS, or safety
boundary is modified.
