# L273 Critic Capacity and Action-Representation Diagnosis

## Question and boundary

L272 showed that scene interference is real but not sufficient: neither a
scene-conditioned Critic nor six equal-compute scene-specific Critics repaired
held-out action ranking. L273 asks the next narrower question: can the frozen
69D state and two-dimensional action be mapped to the already measured H40
return ranking by the current Critic function class, and if not, is the
bottleneck network capacity or raw action representation?

This is a supervised **diagnostic of the Critic function class**, not policy
training and not a replacement for RL. Oracle returns are never supplied to an
Actor, replay buffer, MPPI, or deployed controller. Actor training remains
unauthorized.

## Frozen data and leakage controls

- Use only existing Windows-native L263 constant-continuation H40 rollouts and
  the independently frozen L268 held-out recovery diagnostic. All files are
  SHA256 pinned in the configuration.
- L263's 18 training-role states (27 actions each) are the only fitting data.
  Its eight validation-role states are never sampled for updates. State—not an
  action row—is the split unit.
- The 36 L268 states (three actions each) form a second external evaluation and
  are never sampled for updates.
- Within each state, H40 returns are centered and divided by that state's
  standard deviation. This removes state-level return offsets and tests the
  action ranking that the Actor needs from the Critic.
- L258, final Hairpin/S-Chicane/Infinity, and sealed seeds are forbidden.

## Frozen 2 x 2 causal design

Three paired seeds run four fixed arms for 3,000 updates, batch 256. Sampling
first chooses training states uniformly and then actions uniformly, preventing
one state's grid from dominating.

| Arm | Hidden widths | Action input |
| --- | --- | --- |
| `baseline_raw` | 256, 256 | `(v, omega)` |
| `capacity_wide_raw` | 512, 512 | `(v, omega)` |
| `baseline_polynomial` | 256, 256 | fixed degree-3 action basis |
| `capacity_wide_polynomial` | 512, 512 | fixed degree-3 action basis |

The degree-3 basis is exactly `(v, omega, v^2, omega^2, v*omega, v^3,
omega^3, v^2*omega, v*omega^2)`. It adds no learned or privileged state. All
arms use the same 25-quantile objective, optimizer, learning rate, batch size,
updates, observations, targets, and paired random seeds. Checkpoints at 1,000
and 3,000 updates are engineering artifacts, not selection candidates.

## Metrics and frozen decision tree

Report mean within-state Spearman, top-1 action agreement, and standardized
target error on the L263 train/validation states. On L268 also report
three-action Spearman, recovery-versus-fast-forward accuracy, and six-scene
coverage. The paired seed is the inference unit.

Decisions use only the frozen thresholds in the configuration:

1. If `baseline_raw` itself fits training and generalizes to both evaluations,
   the function class is sufficient and the remaining cause is Bellman/target
   learning rather than capacity or action encoding.
2. If polynomial encoding consistently improves the baseline-width arm and
   external scenes, classify action representation as limiting.
3. If width consistently improves the raw-action arm, classify capacity as
   limiting.
4. If only the combined arm passes, classify a combined bottleneck.
5. Otherwise classify observation/target generalization as limiting and return
   to the unresolved privileged/history observation probe.

No result authorizes Actor training automatically. All arms and failures are
retained; there is no seed, checkpoint, or threshold selection after viewing
outcomes.

