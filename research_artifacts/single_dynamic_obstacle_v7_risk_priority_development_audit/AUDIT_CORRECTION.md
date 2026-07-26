# v7 development-gate audit correction

The original v7 execution is preserved unchanged in
`single_dynamic_obstacle_v7_risk_priority_development/` and was archived with
its original `development_result.json` and bundle hash.

The original gate reported `emergency_mechanism_exercised: false` because the
v6 audit only counted the legacy emergency-candidate diagnostic. The v7 source
contract intentionally uses the normal active-avoidance fallback and records
its preservation at
`probabilistic_obstacle_speed_governor_bypassed_for_active_avoidance_steps`.
The audit now counts either diagnostic; no episode data or control code was
changed.

The corrected audit is therefore an analysis-only replay of the exact 16
sealed episode artifacts. It is not a new experiment and does not authorize
formal execution.
