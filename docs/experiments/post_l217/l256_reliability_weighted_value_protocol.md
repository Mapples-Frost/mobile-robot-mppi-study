# L256 Reliability-Weighted Terminal Value Development Protocol

## Root-cause correction

L255 completed 2/2 but is an engineering-invalid no-treatment duplicate of
L254: its resolved Full configuration did not enable
`paper_rl_driven.conservative_terminal`, so the new authority cap could not
execute. L255 raw artifacts are retained and must not be interpreted as a
method result.

L256 activates the already implemented and previously calibrated conservative
terminal evaluator using the frozen L198 thresholds. It also requires the
current causal ICODE/innovation confidence to upper-bound the candidate-level
critic authority. This is the intended `Reliability-Weighted Value/HSS`
component of the frozen core method, not a new paper direction.

## Frozen design

- Development scene: `tracking_grand_s_chicane_l234`.
- Physics/seed: `nominal_seen`, `923301001`.
- Paired arms: unchanged `icode_mppi` and treated `full_proposed`.
- Equal budget: 100 rollouts per decision, maximum 350 steps.
- MuJoCo, map, Actor, ICODE ensemble, MPPI running cost, boundary filter and
  safety chain are unchanged from L255.
- Conservative-terminal thresholds are copied exactly from the prior L198
  calibration; no L254/L255 outcome was used to tune them.
- A manifest runtime contract must fail before MuJoCo starts unless Full has
  HSS, conservative terminal value, terminal weight 1.0 and diagnostics
  enabled in the resolved configuration.

## Gate

The probe passes only when:

1. exactly two unique, provenance-complete MuJoCo episodes finish without
   Traceback/Exception/NaN;
2. both arms have zero collision and zero boundary violations;
3. Full has no-feasible-decision fraction at most 0.10;
4. Full reaches completion at least 0.22 with at most 100 stuck steps;
5. conservative terminal is enabled on every Full step, the causal cap is
   active on at least one step, and zero causal confidence implies zero
   applied terminal authority;
6. unchanged ICODE completion differs from L254 by at most 0.01.

A positive result proceeds to a full-budget S-Chicane replication. A negative
result retains only raw data and a concise Gate status; no plot or detailed
report is produced. No sealed seed, seed selection, failed-run deletion or
post-outcome threshold change is allowed.
