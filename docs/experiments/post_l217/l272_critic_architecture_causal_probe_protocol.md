# L272 Equal-Compute Critic Architecture Causal Probe

## Question and authorization boundary

L271 found that real six-scene Critic gradients were more destructive than
equal-sized shuffled partitions in two of three frozen checkpoint blocks. L272
tests whether removing or explicitly representing scene identity causally
improves held-out action ranking. This is a Critic-only diagnostic. It does not
authorize Actor training, reward changes, observation changes, or deployment.

## Frozen data, model semantics, and exclusions

- Windows-native CUDA execution under `D:\Projects\mobile-robot-mppi-study`.
- Every input artifact is SHA256 pinned in the configuration: the L262 source
  checkpoint, L268 formal recovery dataset/config, the 36-state held-out H40
  diagnostic, all three shared-Critic checkpoints, and the positive L271
  summary.
- Three paired seed blocks use the exact deterministic L268
  recovery-balanced replay for that seed. Replay contents, reward, action
  scaling, normalizer, target rule, gamma, tau, batch size, hidden widths,
  activation, quantile count, optimizer, and learning rate are unchanged.
- Actor and alpha remain frozen and are hash-audited. ICODE, MPPI, HSS,
  Actor/Traditional fusion, maps, and safety chain are untouched.
- L258, final Hairpin/S-Chicane/Infinity artifacts, and sealed seeds are
  forbidden.

## Paired equal-compute arms

The independent replicate is the paired training seed. All arms start from the
same L262 Critic state and use the corresponding frozen L268 treatment replay.

1. `shared_69d`: the already completed L268 recovery-balanced Critic, 6,000
   updates of batch 256, is hash-pinned and reevaluated in the common L272
   evaluator.
2. `per_scene_equal_compute`: six independent copies of the source Critic each
   see only one scene for 1,000 updates of batch 256. Across the ensemble this
   is exactly 6,000 optimizer steps and 1,536,000 sampled transitions, equal to
   the shared arm. Each scene receives 256,000 sampled exposures in both arms.
   Scene execution order is independently seeded within each paired block.
3. `scene_conditioned_equal_compute`: one Critic receives the frozen normalized
   69D observation followed by a six-way one-hot scene context. Its first-layer
   observation and action columns, all later layers, and target networks are
   copied exactly from L262; the six new context columns are zero initialized.
   It receives 6,000 scene-balanced updates of batch 256.

Arm execution order is preregistered and fixed. Per-scene checkpoints are
written at 500 and 1,000 updates; scene-conditioned checkpoints at 3,000 and
6,000. These are engineering recovery points, not checkpoint candidates.

## Evaluation and unit of inference

Only the already frozen L268 held-out recovery diagnostic is used: 36 states,
six per training scene, with exactly three H40 action returns per state
(`recorded_recovery`, `source_actor`, `fast_forward`). No outcome enters
training. The common evaluator reports per seed and per scene:

- recovery-versus-forward pair accuracy;
- mean within-state three-action Spearman correlation;
- top-1 action agreement;
- Q range and quantile spread.

Repeated actions within one state and states within one seed are descriptive,
not independent replicates. Gate direction is assessed over the three paired
seed blocks, with scene coverage as a robustness condition.

## Frozen Gate and decision

An intervention arm passes only if all conditions hold relative to
`shared_69d`:

- aggregate pair accuracy at least 0.65 and paired median improvement at least
  0.10;
- aggregate three-action Spearman at least 0.15 and paired median improvement
  at least 0.10;
- aggregate top-1 agreement at least 0.45 and paired median improvement at
  least 0.05;
- at least two of three seed blocks improve both pair accuracy and Spearman;
- at least four of six scenes improve pair accuracy and no scene decreases by
  more than 0.166667;
- all values remain finite, absolute mean Q is at most 500, quantile spread at
  most 100, and Actor/alpha hashes do not change.

If both arms pass, the single deployable scene-conditioned Critic is selected.
A conditioned pass authorizes only preregistration of a small-budget Actor
probe. A per-scene-only pass confirms interference but requires a deployable
conditioning design before any Actor update. Failure sends the diagnosis to
Critic capacity/action representation. All failures and checkpoints are kept;
there is no seed or checkpoint selection by outcome.
