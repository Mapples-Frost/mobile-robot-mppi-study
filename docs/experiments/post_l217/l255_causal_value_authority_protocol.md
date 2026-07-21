# L255 Causal Value-Authority Development Protocol

## Purpose

L254 rejected the weighted-action cancellation hypothesis and isolated a
cross-layer authority contradiction in Full Proposed: the causal
ICODE/innovation reliability was zero while the terminal critic retained
unit authority. L255 tests one implementation change only: terminal-value
authority is upper-bounded by the causal dynamics confidence already used by
the residual-conditioned proposal layer.

This is an implementation of the frozen method contract
`Reliability-Weighted Value/HSS`; it does not change ICODE, the Actor, MPPI
cost weights, map geometry, MuJoCo physics, rollout budget, or safety chain.

## Frozen design

- Development only; no sealed seed is used.
- Scene: `tracking_grand_s_chicane_l234`.
- Physics: `nominal_seen`.
- Seed: `923301001`.
- Arms: `icode_mppi`, `full_proposed`.
- Budget: 100 rollouts per control step; 350 control steps maximum.
- Full uses the frozen BC Actor checkpoint and all L254 settings.
- ICODE-MPPI is an unchanged paired control arm.
- MuJoCo execution and all per-step artifacts are required.

## Pre-specified engineering checks

1. Exactly two unique completed episodes with complete provenance.
2. `qualification=1`, MuJoCo 3.2.3, matching frozen Git/config/checkpoint
   fingerprints, and no Traceback/Exception/NaN.
3. In Full rows where causal dynamics confidence is zero, terminal raw
   authority may remain nonzero but the applied terminal authority and causal
   cap must be zero.
4. ICODE completion must remain within 0.01 absolute of its L254 paired value,
   confirming that the intervention is isolated to Full.

## Development efficacy gate

L255 passes this short probe only if all conditions hold:

- zero collision and zero path-boundary violation in both arms;
- Full boundary no-feasible fraction is at most 0.10;
- Full completion is at least 0.22 within 350 steps;
- Full stuck steps are at most 100;
- the terminal causal cap is active on at least one Full control step.

Passing this probe permits a full-budget S-Chicane replication followed by a
three-map development matrix. Failure keeps only raw data and a concise Gate
status; no figure or detailed report is produced. No seed selection, failed
episode deletion, or post-outcome threshold changes are allowed.
