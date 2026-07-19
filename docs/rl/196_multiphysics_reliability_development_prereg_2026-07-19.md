# L196 multi-physics reliability development preregistration

Date frozen: 2026-07-19
Status: frozen before running seeds 569--571
Parent: passed L194 source-relative Actor competence remediation

## Question

Under fixed compute, does the separated
\((c_{\mathrm{dyn}},c_\pi)\) reliability mechanism preserve safety while
improving or maintaining path tracking across seen and unseen MuJoCo dynamics
shifts?

This is the final development Gate for reliability-calibrated hybrid sampling.
It is not sealed confirmation and it does not include the final
critic-reliability term \(c_Q\).

## Design

The experimental unit is one complete closed-loop episode.  Every
scene--domain--seed block contains all four arms in seeded randomized order:

1. ordinary ICODE with fixed hybrid sampling;
2. value-aligned ICODE with fixed hybrid sampling;
3. ordinary ICODE with adaptive \((c_{\mathrm{dyn}},c_\pi)\);
4. value-aligned ICODE with adaptive
   \((c_{\mathrm{dyn}},c_\pi)\), called the current Full Proposed development
   arm.

Frozen factors:

- scene: `high_dynamic_reverse_s_terminal_dev_l187`;
- domains:
  `high_mass_seen`, `low_friction_seen`, `long_delay_seen`,
  `combined_unseen`;
- development seeds: 569, 570, 571;
- 4 domains x 3 seeds x 4 arms = 48 episodes;
- `K=100`, two paper iterations, horizon 36;
- episode limit 360, success tolerance 0.25 m;
- terminal guidance radius 1.26 m and floor 0.30;
- same Actor, ordinary ICODE ensemble, value-aligned ICODE ensemble, L192
  dynamics calibration, and L194 source-competence configuration;
- memory disabled and the complete LaserScan, local obstacle layer,
  `scan_guard`, and safety-arbitration chain unchanged.

Domains may be executed as independent filesystem shards for wall-clock
efficiency.  Each shard has its own seeded four-arm schedule.  Shards are
combined only after all 48 episodes finish.  Parallel execution does not share
controller state or random-number generators.

The independent inferential cluster remains the seed.  Physics domains are
repeated strata within seed, not twelve independent replicates.

## Additional diagnostics frozen before execution

Episode summaries must include:

- dynamics confidence mean/min/max;
- Actor support confidence mean/min/max;
- source-relative Actor competence mean/min/max;
- raw and applied guided fractions;
- low/medium/high authority fractions;
- equal rollout and iteration budgets.

No MuJoCo ground-truth disturbance label enters the online controller.  Domain
labels are used only after execution for stratified analysis.

## Development Gate

The Gate passes only if all conditions hold:

1. Full Proposed succeeds in all 12 shifted-domain episodes;
2. Full Proposed has zero collisions;
3. pooled Full Proposed mean cross-track RMSE is no greater than pooled
   ordinary-fixed mean cross-track RMSE;
4. Full Proposed is within 5% of ordinary fixed in at least three of four
   individual domains;
5. in `combined_unseen`, Full Proposed cross-track RMSE is no greater than
   ordinary fixed;
6. pooled Full Proposed control jerk is no more than 10% above ordinary fixed;
7. Full Proposed dynamics confidence in `combined_unseen` is lower than the
   mean over the three seen-shift domains;
8. every adaptive episode reports finite, nonconstant source competence and
   changes its raw guided allocation away from a constant 0.60;
9. every arm uses exactly `K=100` and two paper iterations.

The per-domain noninferiority allowance is a stability requirement; the pooled
and combined-unseen comparisons require directional improvement, not merely a
wide tolerance.

## Decision rule

- Pass: freeze \((c_{\mathrm{dyn}},c_\pi)\) and proceed to the separately
  preregistered critic-reliability/conservative-terminal Gate.
- Fail only the dynamics-confidence ordering: diagnose whether the ICODE
  ensemble already covers `combined_unseen`; do not alter the controller using
  domain labels.
- Fail safety, pooled tracking, or unseen tracking: reliability-calibrated HSS
  does not advance to sealed testing in its current form.

The L186 geometry and seeds 561--565 remain sealed throughout L196.
