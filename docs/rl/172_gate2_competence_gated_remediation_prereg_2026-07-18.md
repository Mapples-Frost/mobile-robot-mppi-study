# Gate 2 remediation: competence-gated value alignment

Date frozen: 2026-07-18

## Why remediation is necessary

The clean L188 paired development experiment completed all 18 paired cells but
failed the frozen closed-loop Gate. Relative to ordinary task fine-tuning,
unconditional value alignment:

- reduced mean control jerk by 1.72%;
- increased minimum clearance by 1.09%;
- produced no collisions in either condition;
- increased mean final goal distance by 11.29%;
- changed success from 8/18 to 7/18.

The failure is retained and must not be excluded from the research record.
L186/L187 are separate invalid runs because duplicate background processes
wrote to the same directories; neither is evidence.

L188 also exposed an assay floor: both checkpoints achieved 0/9 success in the
narrow-corridor strata at the frozen 100-rollout budget. The terminal critic
mean was strongly separated by task competence: approximately 11--13 in the
clean scene and approximately -0.5 in the corridor. This diagnostic is
development evidence, not a confirmatory subgroup result.

## Mechanistic correction

The existing confidence term measures observation support and twin-critic
agreement, but it does not ask whether the critic represents a competent
policy region. Consequently, a mutually agreeing critic can still shape ICODE
in trajectories where the Actor does not solve the task.

L189 adds a training-only competence authority:

```text
value authority
  = support confidence
  * twin-critic agreement
  * task-competence weight
```

The competence weight is zero below a failure-value reference, one above a
success-value reference, and smoothstep-interpolated between them. Both
references are calibrated once from the L183 **training split only**:

- off value: median critic value of transitions from failed episodes;
- on value: median critic value of transitions from successful episodes.

Episode outcomes, rather than timesteps, define the labels. Validation, test,
unseen, L188 endpoints, scene names and simulator truth at deployment do not
enter this gate. Runtime control still uses a single frozen ICODE checkpoint;
there is no online outcome leakage.

## Frozen L189 settings

- same L183 data, base ICODE, Actor/critics, H=10 objective and optimizer family;
- `lambda_value=5`, retained from L185;
- no value-weight grid;
- failure quantile = 0.5;
- success quantile = 0.5;
- random seed = 20260720;
- model selection remains validation terminal-value RMSE under the 3% rollout
  degradation constraint.

## Development evaluation

If L189 passes the offline mechanism Gate, compare L184 versus L189 using:

- scenes: clean single obstacle and narrow corridor;
- domains: nominal-seen, long-delay-seen and combined-unseen;
- new seeds: 24, 25 and 26;
- 100 total prediction rollouts, two MPPI iterations;
- one randomized checkpoint order inside every block.

The closed-loop criterion is unchanged: improve final goal distance or control
jerk without worsening the other, success, or collision.

If this development comparison passes, run an independent confirmation on
new seeds 27--31. If it fails, do not tune competence quantiles on those
outcomes; reconsider critic calibration or retain value consistency as an
offline-only result.
