# L271 Critic Scene-Interference Diagnosis Protocol

## Question and scope

L263 showed that realized MuJoCo returns usually prefer recovery while the
Critic misranks actions. L268 recovery-balanced replay improved all three
paired Spearman blocks but failed the frozen Critic Gate. L269 found delayed
recovery value without passing its long-horizon Gate. L270 found close 69D
oracle-action conflicts, but none of the preregistered candidate observables
decoded recovery steering well enough to authorize an observation change.

L271 asks one narrower question: are the gradients produced by the six formal
training scenes more mutually destructive than gradients from equally sized
random partitions of the exact same transitions? This is a read-only
diagnostic of the already trained L268 recovery-balanced Critics. It cannot
train or initialize an Actor or Critic.

## Frozen inputs and exclusions

- Windows-native CUDA execution under `D:\Projects\mobile-robot-mppi-study`.
- The L262 source checkpoint, L268 formal recovery dataset, L268 Critic-only
  integrity/results, all three recovery-balanced 6k checkpoints, and the L270
  negative summary are SHA256 pinned in the config.
- The exact deterministic L268 replay constructor is reused. Each seed must
  recreate 6,000 transitions, exactly 1,000 per scene.
- Actor, alpha, target Critics, reward, 69D observation, network structure,
  replay contents, ICODE, HSS, MPPI, Actor/Traditional fusion, maps, and safety
  chain remain frozen.
- L258 and final Hairpin, S-Chicane, and Infinity artifacts are forbidden.

## Gradient audit

For each of the three frozen L268 recovery-balanced 6k checkpoints, select 128
transitions per scene without replacement using the preregistered seed. Compute
the ordinary frozen one-step distributional Critic loss. Actor sampling is
performed once, in stable transition order, before any real or null grouping;
the resulting Bellman targets are frozen and reused by every partition so
policy noise cannot masquerade as group conflict. The Actor, alpha, online
Critics, and target Critics are never stepped. Concatenate twin-Critic
gradients and report cosine matrices, gradient norms, negative-pair fractions,
and cancellation ratios for:

1. all Critic parameters;
2. observation columns of the first layer;
3. the two action columns of the first layer;
4. the output layer.

The primary null keeps the identical 768 selected transitions but randomly
permutes their scene labels into six groups of 128. Twenty-four independently
seeded balanced null partitions are evaluated. This distinguishes
scene-associated conflict from generic finite-batch gradient disagreement.

## Frozen Gate

For each checkpoint, real scene interference is an extreme block only when:

- the real action-input median off-diagonal cosine is below the fifth
  percentile of its null distribution;
- the real all-parameter cancellation ratio is above the 95th percentile of
  its null distribution;
- the null-minus-real action-input median-cosine effect is at least 0.05;
- the real-minus-null all-parameter cancellation effect is at least 0.05;
- the largest-to-smallest scene gradient-norm ratio is at most 20, preventing
  one numerically dominant scene from being mislabeled as directional conflict.

The Gate supports scene interference only if at least two of three checkpoint
blocks meet every condition and every computed value is finite. A pass
authorizes only preregistration of an equal-compute shared versus per-scene
versus scene-conditioned Critic probe. A failure sends the diagnosis to Critic
capacity/action representation. No L271 result authorizes Actor training or a
formal algorithm change. All outcomes are retained.
