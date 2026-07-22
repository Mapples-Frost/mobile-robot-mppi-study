# L264–L266 Critic Failure Diagnosis Protocol

Status: preregistered after L263 and before L264–L266 outcome tables are read.

## Question and stopping rule

L263 established that the frozen L262 Critic has poor H40 action ranking. This
diagnosis asks, in order, whether the error is caused by (L264) Bellman-target
implementation, (L265) action extrapolation outside replay support, or (L266)
missing off-path recovery experience. This is a read-only diagnostic stage.
No Actor, Critic, reward, observation, normalizer, map, MPPI cost,
Actor/Traditional fusion, or safety parameter may be changed, and no gradient
update may be retained. L267 observation-aliasing and any repair are separately
preregistered only after this stage stops.

## Frozen artifacts

- Windows-native repository: `D:\Projects\mobile-robot-mppi-study`.
- L263 code commit: `6b7c5c9c3f8694e0980b4c1e4caf0e82c0927d33`.
- L262 checkpoint: `l262_coverage_gated_value_seed20262611_6k`, step 6000.
- Checkpoint SHA256:
  `48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.
- L263 state library: 26 accepted states plus the unchanged rejected state
  `scene_07_state_02`.
- L263 action library: 702 unique state-action pairs and frozen H40 rollouts.
- Diagnostic random seed: 20262864.
- L258 and final Hairpin, S-Chicane, Infinity artifacts remain forbidden.

## L264: Bellman-target integrity

The audit selects 120 replay transitions before reading their diagnostics: 20
per scene, with up to four true terminals per scene followed by evenly spaced
nonterminal transition IDs. For an independently reloaded temporary agent, it
captures the exact target tensor passed by `SACAgent.update` while retaining no
optimizer or parameter mutation. From the same frozen target action and log
probability, NumPy independently recomputes

`r + gamma * (1-done) * (min(target_q1,target_q2) - alpha*log_pi)`.

Pass requires maximum absolute difference at most `1e-6`. The audit also
records reward, bootstrap, entropy, full target, online Q, absolute TD error,
quantile spread, twin disagreement, terminal rate, and outcome labels for all
6,000 replay transitions by scene. Static provenance records that replay stores
`terminated`, not `terminated or truncated`; time-limit truncation therefore
bootstraps. Reward scaling must occur only in the environment and not again in
the SAC update. The configured horizon is one-step and target critics must be
loaded separately and soft-updated with the frozen `tau`.

## L265: action-support and extrapolation

For each L263 state, Euclidean distance in the frozen normalized 69D space
selects the 64 nearest replay observations. Candidate support distance is the
minimum normalized-action L2 distance to those local replay actions. Local
thresholds use only replay data: each neighbor's nearest-other-action distance
within the same 64 actions is computed, with its median and 90th percentile as
the state-specific thresholds.

- `in_support`: distance no greater than the local median;
- `near_support`: above the median and no greater than local q90;
- `out_of_support`: above local q90.

The audit additionally records local observation distance, local action
density within q90, deterministic Actor-action distance, exact squashed-policy
log probability, and action-boundary margin. Frozen H40 Actor-follow returns are
joined without resimulation. Per support class it reports within-state
Spearman, Top-1, Top-3, standardized Q/return error, and recovery-versus-forward
pair accuracy. A recovery/forward pair is assigned the worse support class of
its two actions.

Interpretation:

- in-support mean Spearman below 0.20 or pair accuracy below 0.60 is evidence
  that extrapolation alone cannot explain the Critic failure;
- in-support credible while out-of-support Spearman is lower by at least 0.20
  or pair accuracy is lower by at least 0.15 supports an
  extrapolation-dominant cause;
- fewer than three states with a valid metric is reported as insufficient, not
  converted to a causal conclusion.

## L266: replay coverage and recovery chains

True absolute CTE is recovered as `sqrt(constraint_cost)`; the encoded signed
CTE is not used for magnitude bins. Fixed bins are `<0.25`, `0.25–0.75`,
`0.75–1.5`, `1.5–3.0`, and `>3.0` metres. Curvature bins use raw feature 14:
straight `<0.05`, gentle `0.05–0.20`, sharp left/right `>=0.20`, with a
sign-changing adjacent transition labelled reversal when both magnitudes exceed
0.05.

Replay episode continuity requires consecutive transition IDs, identical scene
group, previous `done=0`, and maximum error between previous
`next_observation` and current `observation` at most `1e-5`; this also detects
unrecorded time-limit resets. A recovery transition starts above 0.75 m and
reduces CTE by at least 0.05 m at the next continuous state. A complete recovery
chain contains an off-path state above 0.75 m followed in the same reconstructed
episode by re-entry below 0.75 m. Counts are reported, never inferred from
validation trajectories. `loop_exit` is marked unavailable because the frozen
replay contains only the six training scenes.

Coverage is called sparse when fewer than six complete recovery chains exist in
total or any training scene has zero. This threshold is descriptive, not a
claim that six chains are sufficient for learning.

Every CTE/curvature/scene/stage cell records count, action moments, positive
recovery fraction, reward, TD error, quantile spread, and Actor–replay action
gap. Empty cells remain explicit.

## Outputs and decision rule

Raw CSV/JSON, integrity checks, a concise Chinese diagnostic report, and
reproducible PDF/PNG diagnostic figures are stored under
`results/research_platform/rl/l264_l266_critic_failure_diagnosis`. The one
recommended next repair is selected lexicographically:

1. Bellman mismatch if L264 exceeds `1e-6` or a static contract fails;
2. support-constrained Critic/Actor repair if L265 is extrapolation-dominant;
3. targeted recovery data collection if in-support ranking is poor and L266
   has no or very sparse complete recovery chains;
4. Critic representation/target propagation audit if in-support ranking is
   poor despite material recovery coverage;
5. L267 observation aliasing only after the preceding causes are insufficient.

The stage then pauses for human review. No training follows automatically.
